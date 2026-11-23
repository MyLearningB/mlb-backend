from contextlib import asynccontextmanager
from datetime import datetime, timedelta, date
from typing import Optional
import asyncio
import json
import os

import firebase_admin
from firebase_admin import credentials
from fastapi import (
    FastAPI,
    HTTPException,
    status,
    Depends,
    Header,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select

from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

# ============================================================
# DATABASE
# ============================================================

from database import (
    create_db_and_tables,
    get_session,
    engine,
)

# ============================================================
# MODELS
# ============================================================

from models import (
    User,
    StudyPlan,
    StudySession,
    Quest,
    Feedback,
)

# ============================================================
# SECURITY
# ============================================================

from security import get_current_user

# ============================================================
# SCHEMAS
# ============================================================

from schemas import (
    DashboardResponse,
    UserDashboardInfo,
    PetDashboardInfo,
    StreakInfo,
    SolveRequest,
    SolveResponse,
    SolveFeedbackRequest,
    TodayPlanSession,
    OnboardingQuizSubmit,
    StudyProfileUpdate,
    normalize_mode,
)

# ============================================================
# AI SERVICE
# ============================================================

from ai_service import generate_gemini_solution

# ============================================================
# ROLLOVER
# ============================================================

from plan_rollover import plan_rollover_loop

# ============================================================
# ROUTERS
# ============================================================

from routers import (
    auth,
    canvas,
    collections,
    community,
    notes,
    notifications,
    onboarding,
    plan,
    profile,
    study,
    trophies,
)

from routers.auth import limiter
from routers.onboarding import onboarding_step
from gamification import (
    get_pet,
    pet_view,
    streak_view,
    xp_history_last_7,
)
from utils import (
    ALLOWED_GOAL_TYPES,
    get_local_now,
)


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()

    firebase_json_str = os.getenv("FIREBASE_CREDENTIALS_JSON")

    try:
        if not firebase_admin._apps:
            if firebase_json_str:
                cred_dict = json.loads(firebase_json_str)
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred)
                print("🔥 Firebase initialized from FIREBASE_CREDENTIALS_JSON.")
            elif os.path.exists("firebase-credentials.json"):
                cred = credentials.Certificate("firebase-credentials.json")
                firebase_admin.initialize_app(cred)
                print("🔥 Firebase initialized from local credentials file.")
            else:
                print("⚠️ Firebase credentials not found. Push notifications disabled.")
    except Exception as e:
        print(f"❌ Error initializing Firebase: {e}")

    rollover_task = asyncio.create_task(plan_rollover_loop())

    yield

    rollover_task.cancel()
    try:
        await rollover_task
    except asyncio.CancelledError:
        pass


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="myLB API",
    version="1.0",
    lifespan=lifespan,
)


# ============================================================
# RATE LIMITER
# ============================================================

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ============================================================
# CORS
# ============================================================

origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,https://app.mylb.com",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROUTER REGISTRATION
# ============================================================

app.include_router(auth.router)
app.include_router(study.router)
app.include_router(notes.router)
app.include_router(canvas.router)
app.include_router(collections.router)
app.include_router(notifications.router)
app.include_router(community.router)
app.include_router(trophies.router)

# Onboarding state, pet catalog, adopt + rename pet
app.include_router(onboarding.router)

# Study plan (generate / approve / adjust / complete)
app.include_router(plan.router)

app.include_router(
    profile.router,
    prefix="/users/me",
    tags=["Profile"],
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/ping", tags=["Health"])
def keep_alive():
    return {"status": "myLB is awake and ready for beta testing!"}


# ============================================================
# ONBOARDING QUIZ
# ============================================================

@app.patch(
    "/users/me/quiz",
    status_code=status.HTTP_200_OK,
    tags=["Profile"],
)
def submit_onboarding_quiz(
    request: OnboardingQuizSubmit,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    if request.goal_type not in ALLOWED_GOAL_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid goal type.",
        )

    if not current_user.is_first_session:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Onboarding already completed.",
        )

    current_user.goal_type = request.goal_type

    if request.study_goal is not None:
        current_user.study_goal = request.study_goal

    if request.target_date is not None:
        current_user.target_date = request.target_date

    session.add(current_user)
    try:
        session.commit()
    except Exception:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save quiz preferences.",
        )

    return {
        "message": "Study profile updated successfully",
        "goal_type": current_user.goal_type,
    }


# ============================================================
# STUDY PROFILE
# ============================================================

@app.patch(
    "/users/me/study-profile",
    status_code=status.HTTP_200_OK,
    tags=["Profile"],
)
def update_study_profile(
    request: StudyProfileUpdate,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    x_timezone: str = Header("UTC"),
):
    if request.goal_type is not None:
        if request.goal_type not in ALLOWED_GOAL_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid goal type.",
            )
        current_user.goal_type = request.goal_type

    if request.study_goal is not None:
        goal = request.study_goal.strip()
        current_user.study_goal = goal[:200] or None

    if request.target_date is not None:
        if request.target_date < get_local_now(x_timezone).date():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Target date can't be in the past.",
            )
        current_user.target_date = request.target_date

    if request.clear_target_date:
        current_user.target_date = None

    if request.daily_goal_mins is not None:
        current_user.daily_goal_mins = request.daily_goal_mins

    session.add(current_user)
    try:
        session.commit()
    except Exception:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not update study profile.",
        )

    return {
        "message": "Study profile updated.",
        "goal_type": current_user.goal_type,
        "study_goal": current_user.study_goal,
        "target_date": current_user.target_date,
        "daily_goal_mins": current_user.daily_goal_mins,
    }


# ============================================================
# PET ADOPTION — moved to routers/onboarding.py
# ============================================================


# ============================================================
# DASHBOARD
# ============================================================

@app.get(
    "/users/me/dashboard",
    response_model=DashboardResponse,
    status_code=status.HTTP_200_OK,
)
def get_dashboard(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    x_timezone: str = Header("UTC"),
):
    local_now = get_local_now(x_timezone)
    hour = local_now.hour

    if hour < 12:
        greeting = "Good morning"
    elif hour < 18:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"

    first_name = (
        current_user.name.split()[0]
        if current_user.name
        else "Student"
    )

    user_info = UserDashboardInfo(
        first_name=first_name,
        is_first_session=current_user.is_first_session,
    )

    today = local_now.date()
    today_str = today.isoformat()

    # ---- Streak: one implementation (gamification.streak_view),
    #      the same numbers the profile screen shows.
    streak = streak_view(current_user, today)
    streak_info = StreakInfo(
        days=streak["days"],
        active_today=streak["active_today"],
    )

    # ---- Pet: level, evolution, mood and trait from one place ----
    pet = get_pet(session, current_user.id)
    pet_info = PetDashboardInfo(
        **pet_view(
            pet,
            current_user,
            today,
            xp_history_last_7(session, current_user.id, today),
        )
    )

    plans_by_id = {
        p.id: p
        for p in session.exec(
            select(StudyPlan).where(
                StudyPlan.user_id == current_user.id,
                StudyPlan.is_archived == False,  # noqa: E712
            )
        ).all()
    }

    today_sessions = session.exec(
        select(StudySession)
        .where(
            StudySession.user_id == current_user.id,
            StudySession.date == today_str,
            StudySession.completed == False,  # noqa: E712
        )
        .order_by(StudySession.id)
    ).all()

    real_today_plan = []
    for s in today_sessions:
        plan = plans_by_id.get(s.plan_id)
        if plan is None:
            continue
        real_today_plan.append(
            TodayPlanSession(
                id=str(s.id),
                plan_id=plan.id,
                plan_title=plan.title or plan.subject,
                subject=s.subject,
                topic=s.topic,
                duration_mins=s.duration_mins,
                mode=normalize_mode(s.mode),
                emoji=plan.emoji,
                color_hex=plan.color_hex,
                completed=s.completed,
            )
        )
        if len(real_today_plan) >= 8:
            break

    db_quests = session.exec(
        select(Quest).where(Quest.user_id == current_user.id).limit(3)
    ).all()

    real_quests = [
        {
            "id": str(q.id),
            "title": q.title,
            "type": q.type,
            "progress": q.progress,
            "target": q.target,
            "members_count": q.members_count,
        }
        for q in db_quests
    ]

    return DashboardResponse(
        user=user_info,
        pet=pet_info,
        quests=real_quests,
        today_plan=real_today_plan,
        streak=streak_info,
        greeting=greeting,
        goal_type=current_user.goal_type,
        study_goal=current_user.study_goal,
        target_date=current_user.target_date,
        daily_goal_mins=current_user.daily_goal_mins,
        onboarding_step=onboarding_step(session, current_user),
        has_pet=pet is not None,
    )


# ============================================================
# AI SOLVE
# ============================================================
@app.post(
    "/solve",
    response_model=SolveResponse,
    status_code=status.HTTP_200_OK,
)
@limiter.limit("15/minute")
async def solve_question(
    request: Request,
    payload: SolveRequest,
    current_user: User = Depends(get_current_user),
):
    if not payload.question_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide a question_text.",
        )

    # Use the new Gemini function here
    solution_data = await generate_gemini_solution(payload.question_text)

    if not solution_data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Couldn't generate a solution. Try again.",
        )

    return SolveResponse(**solution_data)


# ============================================================
# SOLUTION FEEDBACK
# ============================================================

@app.post(
    "/solve/{solution_id}/feedback",
    status_code=status.HTTP_200_OK,
)
def submit_solution_feedback(
    solution_id: str,
    request: SolveFeedbackRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    new_feedback = Feedback(
        solution_id=solution_id,
        user_id=current_user.id,
        helpful=request.helpful,
        flag_reason=request.flag_reason,
    )

    session.add(new_feedback)
    session.commit()

    print(
        f"✅ Feedback saved for {solution_id}: "
        f"Helpful? {request.helpful}. "
        f"Reason: {request.flag_reason}"
    )

    return {"acknowledged": True}