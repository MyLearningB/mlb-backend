from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Optional
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
)
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select

from sqlalchemy import text

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
    PlanResponse,
    PlanGenerateRequest,
    SessionUpdateRequest,
    PlanApproveRequest,
    SolveRequest,
    SolveResponse,
    SolveFeedbackRequest,
    PlanGoal,
    TodayPlanSession,
    PlanStats,
    WeekDay,
    SessionDetail,
    OnboardingQuizSubmit,
)

# ============================================================
# AI SERVICE
# ============================================================

from ai_service import (
    generate_deepseek_solution,
    generate_deepseek_study_plan,
)

# ============================================================
# ROUTERS
# ============================================================

from routers import (
    collections,
    community,
    notifications,
    study,
    notes,
    canvas,
    profile,
    trophies,
    auth,
)

from routers.auth import limiter

# NEW: Import the evolution math
from utils import get_pet_evolution_data

# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    create_db_and_tables()

    # --------------------------------------------------------
    # FIREBASE
    # --------------------------------------------------------

    firebase_json_str = os.getenv(
        "FIREBASE_CREDENTIALS_JSON"
    )

    try:

        if not firebase_admin._apps:

            # -----------------------------------------------
            # RENDER / PRODUCTION
            # -----------------------------------------------

            if firebase_json_str:

                cred_dict = json.loads(
                    firebase_json_str
                )

                cred = credentials.Certificate(
                    cred_dict
                )

                firebase_admin.initialize_app(
                    cred
                )

                print(
                    "🔥 Firebase initialized from "
                    "FIREBASE_CREDENTIALS_JSON."
                )

            # -----------------------------------------------
            # LOCAL DEVELOPMENT
            # -----------------------------------------------

            elif os.path.exists(
                "firebase-credentials.json"
            ):

                cred = credentials.Certificate(
                    "firebase-credentials.json"
                )

                firebase_admin.initialize_app(
                    cred
                )

                print(
                    "🔥 Firebase initialized from "
                    "local credentials file."
                )

            # -----------------------------------------------
            # NO FIREBASE
            # -----------------------------------------------

            else:

                print(
                    "⚠️ Firebase credentials not found. "
                    "Push notifications disabled."
                )

    except Exception as e:

        print(
            f"❌ Error initializing Firebase: {e}"
        )

    yield


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

app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,
)


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

app.include_router(
    profile.router,
    prefix="/users/me",
    tags=["Profile"],
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/ping",
    tags=["Health"],
)
def keep_alive():

    return {
        "status": "myLB is awake and ready for beta testing!"
    }


# ============================================================
# TIMEZONE HELPER
# ============================================================

def get_local_now(
    timezone_str: str = "UTC",
) -> datetime:

    try:

        tz = ZoneInfo(timezone_str)

    except ZoneInfoNotFoundError:

        tz = ZoneInfo("UTC")

    return datetime.now(tz)

# ============================================================
# ONBOARDING QUIZ 
# ============================================================

@app.patch(
    "/users/me/quiz",
    status_code=status.HTTP_200_OK,
    tags=["Profile"]
)
def submit_onboarding_quiz(
    request: OnboardingQuizSubmit,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    current_user.goal_type = request.goal_type
    current_user.study_goal = request.study_goal
    current_user.target_date = request.target_date
    
    session.add(current_user)
    
    try:
        session.commit()
    except Exception:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save quiz preferences."
        )
        
    return {
        "message": "Study profile updated successfully",
        "goal_type": current_user.goal_type
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

    # --------------------------------------------------------
    # CHECK AUTHENTICATED USER
    # --------------------------------------------------------

    if not current_user.id:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user session.",
        )

    # --------------------------------------------------------
    # VALIDATE PET NAME
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # VALIDATE PET TYPE
    # --------------------------------------------------------

    allowed_pet_types = {
        "nova",
        "pip",
        "luna",
        "zap",
    }

    if pet_data.pet_type not in allowed_pet_types:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid pet type.",
        )

    # --------------------------------------------------------
    # CHECK EXISTING PET
    # --------------------------------------------------------

    existing_pet = session.exec(
        select(Pet).where(
            Pet.user_id == current_user.id
        )
    ).first()

    if existing_pet:

        # ----------------------------------------------------
        # ENSURE ONBOARDING STATE IS CORRECT
        # ----------------------------------------------------

        if current_user.is_first_session:

            current_user.is_first_session = False

            session.add(current_user)
            session.commit()

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Already adopted!",
        )

    # --------------------------------------------------------
    # CREATE PET
    # --------------------------------------------------------

    new_pet = Pet(
        user_id=current_user.id,
        pet_type=pet_data.pet_type,
        pet_name=pet_name,
    )

    session.add(new_pet)

    # --------------------------------------------------------
    # COMPLETE ONBOARDING
    # --------------------------------------------------------

    current_user.is_first_session = False

    session.add(current_user)

    # --------------------------------------------------------
    # COMMIT
    # --------------------------------------------------------

    try:

        session.commit()

    except Exception:

        session.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save your buddy. Please try again.",
        )

    # --------------------------------------------------------
    # REFRESH
    # --------------------------------------------------------

    session.refresh(new_pet)
    session.refresh(current_user)

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # LOCAL TIME
    # --------------------------------------------------------

    local_now = get_local_now(
        x_timezone
    )

    hour = local_now.hour

    if hour < 12:

        greeting = "Good morning"

    elif hour < 18:

        greeting = "Good afternoon"

    else:

        greeting = "Good evening"

    # --------------------------------------------------------
    # USER
    # --------------------------------------------------------

    first_name = (
        current_user.name.split()[0]
        if current_user.name
        else "Student"
    )

    user_info = UserDashboardInfo(
        first_name=first_name,
        is_first_session=current_user.is_first_session,
    )

    # --------------------------------------------------------
    # DATES
    # --------------------------------------------------------

    today = local_now.date()

    today_str = today.isoformat()

    last_7_days = [
        (
            today - timedelta(days=i)
        ).isoformat()
        for i in range(6, -1, -1)
    ]

    # --------------------------------------------------------
    # XP HISTORY
    # --------------------------------------------------------

    activities = session.exec(
        select(DailyActivity).where(
            DailyActivity.user_id == current_user.id,
            DailyActivity.date.in_(last_7_days),
        )
    ).all()

    xp_map = {
        activity.date: activity.xp_earned
        for activity in activities
    }

    real_xp_history = [
        xp_map.get(day, 0)
        for day in last_7_days
    ]

    # --------------------------------------------------------
    # TODAY STREAK
    # --------------------------------------------------------

    streak_active = (
        xp_map.get(today_str, 0) > 0
    )

    # --------------------------------------------------------
    # ALL ACTIVE DAYS
    # --------------------------------------------------------

    all_active_dates = session.exec(
        select(DailyActivity.date)
        .where(
            DailyActivity.user_id == current_user.id,
            DailyActivity.xp_earned > 0,
        )
        .distinct()
        .order_by(
            DailyActivity.date.desc()
        )
    ).all()

    # --------------------------------------------------------
    # CALCULATE STREAK
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PET
    # --------------------------------------------------------

    pet = session.exec(
        select(Pet).where(
            Pet.user_id == current_user.id
        )
    ).first()

    pet_type = pet.pet_type if pet else "nova"
    pet_level = pet.level if pet else 1
    pet_xp = pet.xp if pet else 0
    pet_name = pet.pet_name if pet else "Nova"
    pet_mood = getattr(pet, 'mood', "happy") if pet else "happy"

    # 1. Get the dynamic evolution data
    evolution_data = get_pet_evolution_data(pet_type, pet_level, pet_xp)

    # 2. Combine base data with evolution data
    pet_info_dict = {
        "name": pet_name,
        "type": pet_type,
        "level": pet_level,
        "xp": pet_xp,
        "mood": pet_mood,
        "xp_history": real_xp_history if pet else [0] * 7,
        **evolution_data
    }

    pet_info = PetDashboardInfo(**pet_info_dict)

    # --------------------------------------------------------
    # TODAY'S STUDY SESSIONS
    # --------------------------------------------------------

    today_sessions = session.exec(
        select(StudySession)
        .where(
            StudySession.user_id == current_user.id,
            StudySession.date == today_str,
            StudySession.completed == False,
        )
        .order_by(
            StudySession.id
        )
        .limit(4)
    ).all()

    real_today_plan = [
        TodayPlanSession(
            id=str(study_session.id),
            subject=study_session.subject,
            duration_mins=study_session.duration_mins,
            mode=study_session.mode,
        )
        for study_session in today_sessions
    ]

    # --------------------------------------------------------
    # QUESTS
    # --------------------------------------------------------

    db_quests = session.exec(
        select(Quest)
        .where(
            Quest.user_id == current_user.id
        )
        .limit(3)
    ).all()

    real_quests = [
        {
            "id": str(quest.id),
            "title": quest.title,
            "type": quest.type,
            "progress": quest.progress,
            "target": quest.target,
            "members_count": quest.members_count,
        }
        for quest in db_quests
    ]

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

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
# GET STUDY PLAN
# ============================================================

@app.get(
    "/users/me/plan",
    response_model=Optional[PlanResponse],
    status_code=status.HTTP_200_OK,
)
def get_study_plan(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    x_timezone: str = Header("UTC"),
):

    # --------------------------------------------------------
    # GET MOST RECENT PLAN
    # --------------------------------------------------------

    statement = (
        select(StudyPlan)
        .where(
            StudyPlan.user_id == current_user.id
        )
        .order_by(
            StudyPlan.id.desc()
        )
    )

    db_plan = session.exec(
        statement
    ).first()

    if not db_plan:

        return None

    # --------------------------------------------------------
    # GET PLAN SESSIONS
    # --------------------------------------------------------

    sessions_statement = (
        select(StudySession)
        .where(
            StudySession.plan_id == db_plan.id
        )
    )

    db_sessions = session.exec(
        sessions_statement
    ).all()

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    today = get_local_now(
        x_timezone
    ).date()

    if db_plan.deadline:
        days_remaining = (db_plan.deadline - today).days
        days_remaining = max(days_remaining, 0)
    else:
        days_remaining = -1

    # --------------------------------------------------------
    # PLAN STATS
    # --------------------------------------------------------

    total_duration = sum(
        study_session.duration_mins
        for study_session in db_sessions
    )

    daily_target = (
        total_duration // len(db_sessions)
        if db_sessions
        else 60
    )

    stats = PlanStats(
        days_remaining=days_remaining,
        daily_target_mins=daily_target,
        topics_count=len(db_sessions),
    )

    # --------------------------------------------------------
    # WEEK
    # --------------------------------------------------------

    week = []

    for i in range(7):

        current_date = (
            today + timedelta(days=i)
        )

        date_str = current_date.isoformat()

        day_session = next(
            (
                study_session
                for study_session in db_sessions
                if study_session.date == date_str
            ),
            None,
        )

        week.append(
            WeekDay(
                date=date_str,
                day_label=current_date.strftime(
                    "%a"
                ).upper(),
                has_session=bool(day_session),
                session_type=(
                    "study"
                    if day_session
                    else "rest"
                ),
            )
        )

    # --------------------------------------------------------
    # SESSION DETAILS
    # --------------------------------------------------------

    formatted_sessions = [
        SessionDetail(
            id=str(study_session.id),
            date=study_session.date,
            time=study_session.time or "16:00",
            subject=study_session.subject,
            duration_mins=study_session.duration_mins,
            mode=study_session.mode,
            priority=study_session.priority,
            completed=study_session.completed,
        )
        for study_session in db_sessions
    ]

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return PlanResponse(
        goal=PlanGoal(
            subject=db_plan.subject,
            deadline=db_plan.deadline,
        ),
        stats=stats,
        week=week,
        sessions=formatted_sessions,
        nudge=None,
    )


# ============================================================
# GENERATE STUDY PLAN
# ============================================================

@app.post(
    "/users/me/plan/generate",
    response_model=PlanResponse,
    status_code=status.HTTP_200_OK,
)
async def generate_study_plan(
    request: PlanGenerateRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    x_timezone: str = Header("UTC"),
):

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    today = get_local_now(
        x_timezone
    ).date()

    days_remaining = None

    if request.deadline:
        days_remaining = (request.deadline - today).days
        if days_remaining < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Deadline passed.",
            )

    # --------------------------------------------------------
    # AI GENERATION
    # --------------------------------------------------------

    ai_plan_data = (
        await generate_deepseek_study_plan(
            goal=request.goal,
            target_date=request.deadline,
            days_remaining=days_remaining,
        )
    )

    if not ai_plan_data:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate plan.",
        )

    # --------------------------------------------------------
    # CREATE PLAN
    # --------------------------------------------------------

    db_plan = StudyPlan(
        user_id=current_user.id,
        subject=request.goal,
        deadline=request.deadline,
        is_approved=False,
    )

    session.add(db_plan)
    session.commit()
    session.refresh(db_plan)

    # --------------------------------------------------------
    # CREATE SESSIONS
    # --------------------------------------------------------

    for ai_session in ai_plan_data.get(
        "sessions",
        [],
    ):

        db_session = StudySession(
            plan_id=db_plan.id,
            user_id=current_user.id,
            date=ai_session["date"],
            time=ai_session.get(
                "time",
                "12:00",
            ),
            subject=ai_session["subject"],
            duration_mins=ai_session[
                "duration_mins"
            ],
            mode=ai_session["mode"],
            priority=ai_session["priority"],
            completed=False,
        )

        session.add(db_session)
        session.commit()
        session.refresh(db_session)

        # ----------------------------------------------------
        # FRONTEND ID
        # ----------------------------------------------------

        ai_session["id"] = str(
            db_session.id
        )

        ai_session["completed"] = False

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return PlanResponse(
        goal=PlanGoal(
            subject=request.goal,
            deadline=request.deadline,
        ),
        stats=ai_plan_data["stats"],
        week=ai_plan_data["week"],
        sessions=ai_plan_data["sessions"],
        nudge=None,
    )


# ============================================================
# UPDATE STUDY SESSION
# ============================================================

@app.patch(
    "/users/me/plan/session/{session_id}",
    status_code=status.HTTP_200_OK,
)
def update_session(
    session_id: int,
    request: SessionUpdateRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):

    db_session = session.exec(
        select(StudySession).where(
            StudySession.id == session_id,
            StudySession.user_id == current_user.id,
        )
    ).first()

    if not db_session:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found.",
        )

    # --------------------------------------------------------
    # UPDATE TIME
    # --------------------------------------------------------

    if request.scheduled_time is not None:

        db_session.time = (
            request.scheduled_time
        )

    # --------------------------------------------------------
    # UPDATE DURATION
    # --------------------------------------------------------

    if request.duration_mins is not None:

        if request.duration_mins <= 0:

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Duration must be greater than zero.",
            )

        db_session.duration_mins = (
            request.duration_mins
        )

    # --------------------------------------------------------
    # UPDATE SKIPPED
    # --------------------------------------------------------

    if request.skipped is not None:

        db_session.skipped = (
            request.skipped
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    session.add(db_session)
    session.commit()
    session.refresh(db_session)

    return db_session


# ============================================================
# APPROVE STUDY PLAN
# ============================================================

@app.patch(
    "/users/me/plan/{plan_id}/approve",
    status_code=status.HTTP_200_OK,
)
def approve_study_plan(
    plan_id: int,
    request: PlanApproveRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):

    # --------------------------------------------------------
    # FIND PLAN BELONGING TO USER
    # --------------------------------------------------------

    db_plan = session.exec(
        select(StudyPlan).where(
            StudyPlan.id == plan_id,
            StudyPlan.user_id == current_user.id,
        )
    ).first()

    if not db_plan:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Plan not found.",
        )

    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    db_plan.is_approved = request.approved

    session.add(db_plan)
    session.commit()
    session.refresh(db_plan)

    return {
        "message": (
            f"Plan {plan_id} "
            f"approved status updated"
        ),
        "approved": db_plan.is_approved,
    }


# ============================================================
# AI SOLVE
# ============================================================

@app.post(
    "/solve",
    response_model=SolveResponse,
    status_code=status.HTTP_200_OK,
)
async def solve_question(
    request: SolveRequest,
    current_user: User = Depends(get_current_user),
):

    if not request.question_text:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide a question_text.",
        )

    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    solution_data = (
        await generate_deepseek_solution(
            request.question_text
        )
    )

    if not solution_data:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Couldn't generate a solution. Try again.",
        )

    return SolveResponse(
        **solution_data
    )


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

    # --------------------------------------------------------
    # CREATE FEEDBACK
    # --------------------------------------------------------

    new_feedback = Feedback(
        solution_id=solution_id,
        user_id=current_user.id,
        helpful=request.helpful,
        flag_reason=request.flag_reason,
    )

    session.add(new_feedback)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    session.commit()

    print(
        f"✅ Feedback saved for {solution_id}: "
        f"Helpful? {request.helpful}. "
        f"Reason: {request.flag_reason}"
    )

    return {
        "acknowledged": True
    }