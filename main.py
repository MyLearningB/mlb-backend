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
    Pet,
    StudyPlan,
    StudySession,
    DailyActivity,
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
    PetAdoptionRequest,
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

from ai_service import generate_deepseek_solution

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
    plan,
    profile,
    study,
    trophies,
)

from routers.auth import limiter
from utils import (
    ALLOWED_GOAL_TYPES,
    get_local_now,
    get_pet_evolution_data,
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
):
    if request.goal_type is not None:
        if request.goal_type not in ALLOWED_GOAL_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid goal type.",
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
            detail="Could not update study profile.",
        )

    return {
        "message": "Study profile updated.",
        "goal_type": current_user.goal_type,
        "study_goal": current_user.study_goal,
        "target_date": current_user.target_date,
    }


# ============================================================
# PET ADOPTION
# ============================================================

@app.post(
    "/users/me/pet",
    status_code=status.HTTP_200_OK,
)
def adopt_pet(
    pet_data: PetAdoptionRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    if not current_user.id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user session.",
        )

    pet_name = pet_data.pet_name.strip()

    if not pet_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pet name cannot be empty.",
        )

    if len(pet_name) > 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pet name cannot exceed 20 characters.",
        )

    allowed_pet_types = {"nova", "pip", "luna", "zap"}

    if pet_data.pet_type not in allowed_pet_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid pet type.",
        )

    existing_pet = session.exec(
        select(Pet).where(Pet.user_id == current_user.id)
    ).first()

    if existing_pet:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Already adopted!",
        )

    new_pet = Pet(
        user_id=current_user.id,
        pet_type=pet_data.pet_type,
        pet_name=pet_name,
    )

    session.add(new_pet)

    current_user.is_first_session = False
    session.add(current_user)

    try:
        session.commit()
    except Exception:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save your buddy. Please try again.",
        )

    session.refresh(new_pet)
    session.refresh(current_user)

    return {
        "pet_id": new_pet.id,
        "pet_type": new_pet.pet_type,
        "pet_name": new_pet.pet_name,
        "level": new_pet.level,
        "xp": new_pet.xp,
        "is_first_session": current_user.is_first_session,
    }


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

    last_7_days = [
        (today - timedelta(days=i)).isoformat()
        for i in range(6, -1, -1)
    ]

    activities = session.exec(
        select(DailyActivity).where(
            DailyActivity.user_id == current_user.id,
            DailyActivity.date.in_(last_7_days),
        )
    ).all()

    xp_map = {a.date: a.xp_earned for a in activities}

    real_xp_history = [xp_map.get(day, 0) for day in last_7_days]

    streak_active = xp_map.get(today_str, 0) > 0

    sixty_days_ago = (today - timedelta(days=60)).isoformat()

    all_active_dates = session.exec(
        select(DailyActivity.date)
        .where(
            DailyActivity.user_id == current_user.id,
            DailyActivity.xp_earned > 0,
            DailyActivity.date >= sixty_days_ago,
        )
        .distinct()
        .order_by(DailyActivity.date.desc())
    ).all()

    streak_count = 0
    check_date = today

    if not streak_active:
        check_date -= timedelta(days=1)

    for active_date_str in all_active_dates:
        if active_date_str == check_date.isoformat():
            streak_count += 1
            check_date -= timedelta(days=1)
        elif active_date_str > check_date.isoformat():
            continue
        else:
            break

    streak_info = StreakInfo(
        days=streak_count,
        active_today=streak_active,
    )

    pet = session.exec(
        select(Pet).where(Pet.user_id == current_user.id)
    ).first()

    pet_type = pet.pet_type if pet else "nova"
    pet_level = pet.level if pet else 1
    pet_xp = pet.xp if pet else 0
    pet_name = pet.pet_name if pet else "Nova"
    pet_mood = getattr(pet, "mood", "happy") if pet else "happy"

    evolution_data = get_pet_evolution_data(pet_type, pet_level, pet_xp)

    pet_info_dict = {
        "name": pet_name,
        "type": pet_type,
        "level": pet_level,
        "xp": pet_xp,
        "mood": pet_mood,
        "xp_history": real_xp_history if pet else [0] * 7,
        **evolution_data,
    }

    pet_info = PetDashboardInfo(**pet_info_dict)

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
                subject=s.subject,
                duration_mins=s.duration_mins,
                mode=normalize_mode(s.mode),
                emoji=plan.emoji,
                color_hex=plan.color_hex,
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

    solution_data = await generate_deepseek_solution(payload.question_text)

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