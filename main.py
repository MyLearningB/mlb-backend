from contextlib import asynccontextmanager
from datetime import datetime, timedelta, date
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
    PlanResponse,
    PlanGoal,
    PlanStats,
    PlanGenerateRequest,
    PlanUpdateRequest,
    PlanArchiveRequest,
    PlanListItem,
    PlanListResponse,
    SessionUpdateRequest,
    SessionDetail,
    PlanApproveRequest,
    SolveRequest,
    SolveResponse,
    SolveFeedbackRequest,
    TodayPlanSession,
    WeekDay,
    OnboardingQuizSubmit,
    StudyProfileUpdate,
    ManualSessionCreate,
    # NEW: shared normalizers
    normalize_mode,
    normalize_priority,
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
from utils import get_pet_evolution_data


# ============================================================
# CONSTANTS
# ============================================================

ALLOWED_GOAL_TYPES = {
    "university",
    "professional",
    "high_school",
    "self_improvement",
}

# Per-persona default emoji + accent colour.
# Used when the client doesn't supply their own, so that
# a uni student and a chess learner see visually different
# chips on the dashboard.
GOAL_TYPE_DEFAULTS = {
    "university":        {"emoji": "🎓", "color_hex": "#7C6EE6"},
    "professional":      {"emoji": "💼", "color_hex": "#3CFFC8"},
    "high_school":       {"emoji": "📚", "color_hex": "#FF8A65"},
    "self_improvement":  {"emoji": "✨", "color_hex": "#C8F53C"},
}


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    # --------------------------------------------------------
    # DATABASE
    # --- NOTE: Transition to Alembic for migrations before v1.1 ---
    # --------------------------------------------------------

    create_db_and_tables()

    # --------------------------------------------------------
    # FIREBASE
    # --------------------------------------------------------

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
# TIMEZONE HELPER
# ============================================================

def get_local_now(timezone_str: str = "UTC") -> datetime:
    try:
        tz = ZoneInfo(timezone_str)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo("UTC")
    return datetime.now(tz)


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _persona_defaults(goal_type: Optional[str]) -> dict:
    """Return the default emoji / color_hex for a goal_type."""
    return GOAL_TYPE_DEFAULTS.get(
        goal_type or "",
        {"emoji": "🎯", "color_hex": "#7C6EE6"},
    )


def _build_plan_response(
    db_plan: StudyPlan,
    db_sessions: list,
    local_today: date,
) -> PlanResponse:
    """Shared PlanResponse builder used by GET /plan and GET /plan/{id}."""
    if db_plan.deadline:
        days_remaining = max((db_plan.deadline - local_today).days, 0)
    else:
        days_remaining = -1

    total_duration = sum(s.duration_mins for s in db_sessions)
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

    week = []
    for i in range(7):
        current_date = local_today + timedelta(days=i)
        date_str = current_date.isoformat()
        day_session = next(
            (s for s in db_sessions if s.date == date_str),
            None,
        )
        week.append(
            WeekDay(
                date=date_str,
                day_label=current_date.strftime("%a").upper(),
                has_session=bool(day_session),
                session_type="study" if day_session else "rest",
            )
        )

    formatted_sessions = [
        SessionDetail(
            id=str(s.id),
            date=s.date,
            time=s.time or "16:00",
            subject=s.subject,
            duration_mins=s.duration_mins,
            mode=normalize_mode(s.mode),
            priority=normalize_priority(s.priority),
            completed=s.completed,
        )
        for s in db_sessions
    ]

    return PlanResponse(
        plan_id=db_plan.id,
        title=db_plan.title or db_plan.subject,
        goal_type=db_plan.goal_type or "self_improvement",
        emoji=db_plan.emoji,
        color_hex=db_plan.color_hex,
        is_approved=db_plan.is_approved,
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
# ONBOARDING QUIZ (first-session only)
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
    """
    Save the user's onboarding quiz answer. Only allowed once
    (guarded by is_first_session). For later edits, use
    PATCH /users/me/study-profile.
    """
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
# STUDY PROFILE (always available)
# ============================================================
#
# NEW. This is the endpoint the client should call when a user
# wants to change their default persona AFTER onboarding. It is
# intentionally NOT gated by is_first_session.

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
        if current_user.is_first_session:
            current_user.is_first_session = False
            session.add(current_user)
            session.commit()

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

    # ------------------------------------------------
    # Today's sessions (excludes archived plans)
    # ------------------------------------------------
    # NEW: join StudyPlan to skip archived plans and pull
    # emoji / color_hex / plan_id for the chip UI.

    plans_by_id = {
        p.id: p
        for p in session.exec(
            select(StudyPlan).where(
                StudyPlan.user_id == current_user.id,
                StudyPlan.is_archived == False,
            )
        ).all()
    }

    today_sessions = session.exec(
        select(StudySession)
        .where(
            StudySession.user_id == current_user.id,
            StudySession.date == today_str,
            StudySession.completed == False,
        )
        .order_by(StudySession.id)
    ).all()

    real_today_plan = []
    for s in today_sessions:
        plan = plans_by_id.get(s.plan_id)
        if plan is None:
            # session belongs to an archived/deleted plan → skip
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
# LIST ALL PLANS (multi-goal switcher)
# ============================================================

@app.get(
    "/users/me/plans",
    response_model=PlanListResponse,
    status_code=status.HTTP_200_OK,
)
def list_study_plans(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    x_timezone: str = Header("UTC"),
    include_archived: bool = False,
):
    today = get_local_now(x_timezone).date()

    query = select(StudyPlan).where(StudyPlan.user_id == current_user.id)

    if not include_archived:
        query = query.where(StudyPlan.is_archived == False)

    plans = session.exec(
        query.order_by(StudyPlan.created_at.desc())
    ).all()

    items: list[PlanListItem] = []

    for plan in plans:
        sessions = session.exec(
            select(StudySession).where(StudySession.plan_id == plan.id)
        ).all()

        completed_count = sum(1 for s in sessions if s.completed)

        days_remaining = None
        if plan.deadline:
            days_remaining = max((plan.deadline - today).days, 0)

        items.append(
            PlanListItem(
                plan_id=plan.id,
                title=plan.title or plan.subject,
                subject=plan.subject,
                goal_type=plan.goal_type or "self_improvement",
                deadline=plan.deadline,
                emoji=plan.emoji,
                color_hex=plan.color_hex,
                is_approved=plan.is_approved,
                is_archived=plan.is_archived,
                days_remaining=days_remaining,
                session_count=len(sessions),
                completed_count=completed_count,
            )
        )

    return PlanListResponse(plans=items, total_count=len(items))


# ============================================================
# GET LATEST PLAN (kept for Flutter compat)
# ============================================================

@app.get(
    "/users/me/plan",
    response_model=Optional[PlanResponse],
    status_code=status.HTTP_200_OK,
)
def get_latest_study_plan(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    x_timezone: str = Header("UTC"),
):
    db_plan = session.exec(
        select(StudyPlan)
        .where(
            StudyPlan.user_id == current_user.id,
            StudyPlan.is_archived == False,
        )
        .order_by(StudyPlan.id.desc())
    ).first()

    if not db_plan:
        return None

    db_sessions = session.exec(
        select(StudySession).where(StudySession.plan_id == db_plan.id)
    ).all()

    return _build_plan_response(
        db_plan,
        db_sessions,
        get_local_now(x_timezone).date(),
    )


# ============================================================
# GET PLAN BY ID
# ============================================================

@app.get(
    "/users/me/plan/{plan_id}",
    response_model=PlanResponse,
    status_code=status.HTTP_200_OK,
)
def get_study_plan_by_id(
    plan_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    x_timezone: str = Header("UTC"),
):
    db_plan = session.exec(
        select(StudyPlan).where(
            StudyPlan.id == plan_id,
            StudyPlan.user_id == current_user.id,
        )
    ).first()

    if not db_plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Study plan not found.",
        )

    db_sessions = session.exec(
        select(StudySession).where(StudySession.plan_id == db_plan.id)
    ).all()

    return _build_plan_response(
        db_plan,
        db_sessions,
        get_local_now(x_timezone).date(),
    )


# ============================================================
# GENERATE STUDY PLAN
# ============================================================

@app.post(
    "/users/me/plan/generate",
    response_model=PlanResponse,
    status_code=status.HTTP_200_OK,
)
@limiter.limit("5/minute")
async def generate_study_plan(
    request: Request,
    payload: PlanGenerateRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    x_timezone: str = Header("UTC"),
):
    today = get_local_now(x_timezone).date()

    days_remaining = None
    if payload.deadline:
        days_remaining = (payload.deadline - today).days
        if days_remaining < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Deadline passed.",
            )

    # --------------------------------------------------------
    # PERSONA RESOLUTION
    # --------------------------------------------------------
    # Prefer the request's goal_type (per-plan persona);
    # fall back to the user's onboarding choice.
    resolved_goal_type = (
        payload.goal_type
        or current_user.goal_type
        or "self_improvement"
    )

    if resolved_goal_type not in ALLOWED_GOAL_TYPES:
        resolved_goal_type = "self_improvement"

    defaults = _persona_defaults(resolved_goal_type)

    # --------------------------------------------------------
    # AI GENERATION (persona-aware)
    # --------------------------------------------------------
    # ai_service.generate_deepseek_study_plan must accept
    # `goal_type` and inject the persona hint into the prompt.
    # See the "ai_service.py" note below this file.
    ai_plan_data = await generate_deepseek_study_plan(
        goal=payload.goal,
        target_date=payload.deadline,
        days_remaining=days_remaining,
        goal_type=resolved_goal_type,
    )

    if not ai_plan_data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate plan.",
        )

    # --------------------------------------------------------
    # PERSIST PLAN
    # --------------------------------------------------------

    db_plan = StudyPlan(
        user_id=current_user.id,
        title=payload.title or payload.goal,
        subject=payload.goal,
        goal_type=resolved_goal_type,
        deadline=payload.deadline,
        emoji=payload.emoji or defaults["emoji"],
        color_hex=payload.color_hex or defaults["color_hex"],
        is_approved=False,
    )

    session.add(db_plan)
    session.commit()
    session.refresh(db_plan)

    # --------------------------------------------------------
    # PERSIST SESSIONS
    # --------------------------------------------------------

    formatted_sessions: list[SessionDetail] = []

    for ai_session in ai_plan_data.get("sessions", []):
        db_session = StudySession(
            plan_id=db_plan.id,
            user_id=current_user.id,
            date=ai_session.get("date", today.isoformat()),
            time=ai_session.get("time", "12:00"),
            subject=ai_session.get("subject", payload.goal),
            duration_mins=ai_session.get("duration_mins", 30),
            # FIX: normalize AI output so DB never stores
            # "Review"/"Medium" variants that break the Literal.
            mode=normalize_mode(ai_session.get("mode", "review")),
            priority=normalize_priority(ai_session.get("priority", "normal")),
            completed=False,
        )

        session.add(db_session)
        session.commit()
        session.refresh(db_session)

        formatted_sessions.append(
            SessionDetail(
                id=str(db_session.id),
                date=db_session.date,
                time=db_session.time,
                subject=db_session.subject,
                duration_mins=db_session.duration_mins,
                mode=db_session.mode,
                priority=db_session.priority,
                completed=False,
            )
        )

    # Rebuild the whole response from the DB so we return exactly
    # what a subsequent GET would return — no AI-shape leakage.
    return _build_plan_response(
        db_plan,
        session.exec(
            select(StudySession).where(StudySession.plan_id == db_plan.id)
        ).all(),
        today,
    )


# ============================================================
# UPDATE PLAN METADATA
# ============================================================

@app.patch(
    "/users/me/plan/{plan_id}",
    status_code=status.HTTP_200_OK,
)
def update_plan_metadata(
    plan_id: int,
    payload: PlanUpdateRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
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

    if payload.title is not None:
        db_plan.title = payload.title
    if payload.deadline is not None:
        db_plan.deadline = payload.deadline
    if payload.emoji is not None:
        db_plan.emoji = payload.emoji
    if payload.color_hex is not None:
        db_plan.color_hex = payload.color_hex
    if payload.goal_type is not None:
        if payload.goal_type not in ALLOWED_GOAL_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid goal type.",
            )
        db_plan.goal_type = payload.goal_type

    session.add(db_plan)
    session.commit()
    session.refresh(db_plan)

    return {
        "message": "Plan updated.",
        "plan_id": db_plan.id,
        "title": db_plan.title,
    }


# ============================================================
# ARCHIVE / UNARCHIVE PLAN
# ============================================================

@app.post(
    "/users/me/plan/{plan_id}/archive",
    status_code=status.HTTP_200_OK,
)
def archive_plan(
    plan_id: int,
    payload: PlanArchiveRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
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

    db_plan.is_archived = payload.archived
    session.add(db_plan)
    session.commit()

    return {
        "message": "Plan archived." if payload.archived else "Plan restored.",
        "plan_id": db_plan.id,
        "is_archived": db_plan.is_archived,
    }


# ============================================================
# APPROVE PLAN — legacy route (Flutter currently hits this)
# ============================================================
#
# Flutter sends PATCH /users/me/plan { approved: true }.
# We approve the user's latest non-archived plan.
# Once Flutter switches to /plan/{plan_id}/approve, this stays
# around as a backwards-compatible convenience.

@app.patch(
    "/users/me/plan",
    status_code=status.HTTP_200_OK,
)
def approve_latest_plan_legacy(
    request: PlanApproveRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    db_plan = session.exec(
        select(StudyPlan)
        .where(
            StudyPlan.user_id == current_user.id,
            StudyPlan.is_archived == False,
        )
        .order_by(StudyPlan.id.desc())
    ).first()

    if not db_plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No plan to approve.",
        )

    db_plan.is_approved = request.approved
    session.add(db_plan)
    session.commit()
    session.refresh(db_plan)

    return {
        "message": f"Plan {db_plan.id} approved status updated",
        "approved": db_plan.is_approved,
        "plan_id": db_plan.id,
    }


# ============================================================
# APPROVE PLAN — canonical per-plan route
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

    db_plan.is_approved = request.approved
    session.add(db_plan)
    session.commit()
    session.refresh(db_plan)

    return {
        "message": f"Plan {plan_id} approved status updated",
        "approved": db_plan.is_approved,
    }


# ============================================================
# CREATE MANUAL STUDY SESSION
# ============================================================

@app.post(
    "/users/me/plan/{plan_id}/session",
    status_code=status.HTTP_201_CREATED,
)
def add_manual_session(
    plan_id: int,
    payload: ManualSessionCreate,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    db_plan = session.exec(
        select(StudyPlan).where(
            StudyPlan.id == plan_id,
            StudyPlan.user_id == current_user.id,
        )
    ).first()

    if not db_plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Study plan not found.",
        )

    new_session = StudySession(
        plan_id=db_plan.id,
        user_id=current_user.id,
        date=payload.date,
        time=payload.time,
        subject=payload.subject,
        duration_mins=payload.duration_mins,
        mode=normalize_mode(payload.mode),
        priority=normalize_priority(payload.priority),
        completed=False,
    )

    session.add(new_session)
    session.commit()
    session.refresh(new_session)

    return {
        "message": "Session added successfully",
        "session": {
            "id": str(new_session.id),
            "date": new_session.date,
            "subject": new_session.subject,
            "duration_mins": new_session.duration_mins,
            "mode": new_session.mode,
        },
    }


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

    if request.scheduled_time is not None:
        db_session.time = request.scheduled_time

    if request.duration_mins is not None:
        if request.duration_mins <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Duration must be greater than zero.",
            )
        db_session.duration_mins = request.duration_mins

    if request.skipped is not None:
        db_session.skipped = request.skipped

    session.add(db_session)
    session.commit()
    session.refresh(db_session)

    return db_session


# ============================================================
# DELETE STUDY SESSION
# ============================================================

@app.delete(
    "/users/me/plan/session/{session_id}",
    status_code=status.HTTP_200_OK,
)
def delete_session(
    session_id: int,
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

    session.delete(db_session)
    session.commit()

    return {"message": "Session removed from plan."}


# ============================================================
# COMPLETE STUDY SESSION & GRANT XP
# ============================================================

@app.post(
    "/users/me/plan/session/{session_id}/complete",
    status_code=status.HTTP_200_OK,
)
def complete_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    x_timezone: str = Header("UTC"),
):
    db_session = session.exec(
        select(StudySession).where(
            StudySession.id == session_id,
            StudySession.user_id == current_user.id,
        )
    ).first()

    if not db_session or db_session.completed:
        raise HTTPException(
            status_code=400,
            detail="Session invalid or already completed.",
        )

    # 1. Mark complete
    db_session.completed = True
    session.add(db_session)

    # 2. Grant XP based on duration
    xp_gained = db_session.duration_mins

    # 3. Add to Pet
    pet = session.exec(
        select(Pet).where(Pet.user_id == current_user.id)
    ).first()

    if pet:
        pet.xp += xp_gained
        session.add(pet)

    # 4. Update Daily Activity (powers the Streak)
    local_today = get_local_now(x_timezone).date().isoformat()

    daily_activity = session.exec(
        select(DailyActivity).where(
            DailyActivity.user_id == current_user.id,
            DailyActivity.date == local_today,
        )
    ).first()

    if daily_activity:
        daily_activity.xp_earned += xp_gained
        daily_activity.study_time_mins += db_session.duration_mins
    else:
        daily_activity = DailyActivity(
            user_id=current_user.id,
            date=local_today,
            xp_earned=xp_gained,
            study_time_mins=db_session.duration_mins,
        )
    session.add(daily_activity)

    session.commit()

    return {"message": "Session completed!", "xp_gained": xp_gained}


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