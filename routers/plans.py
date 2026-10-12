from datetime import date, timedelta
from math import ceil
from typing import Optional, Tuple
import json

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Request,
    status,
)
from sqlmodel import Session, select

from database import get_session
from models import (
    DailyActivity,
    Pet,
    StudyPlan,
    StudySession,
    User,
)
from schemas import (
    ManualSessionCreate,
    PlanApproveRequest,
    PlanArchiveRequest,
    PlanGenerateRequest,
    PlanGoal,
    PlanListItem,
    PlanListResponse,
    PlanResponse,
    PlanStats,
    PlanUpdateRequest,
    SessionDetail,
    SessionUpdateRequest,
    WeekDay,
    normalize_mode,
    normalize_priority,
)
from security import get_current_user
from ai_service import (
    generate_deepseek_phase_plan,
    generate_deepseek_study_plan,
)
from routers.auth import limiter
from utils import (
    get_local_now,
    persona_defaults,
    resolve_goal_type,
)


router = APIRouter(tags=["Study Plan"])


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_WINDOW_DAYS = 7
DEFAULT_STUDY_DAYS_PER_WEEK = 6
DEFAULT_MINUTES_PER_DAY = 60


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _compute_arc_metadata(
    db_plan: StudyPlan,
    local_today: date,
) -> Tuple[
    Optional[int],
    Optional[int],
    Optional[str],
    Optional[str],
]:
    """
    Return (total_weeks, current_week, phase_name, phase_milestone).
    All None when the plan has no deadline.
    """
    if not db_plan.start_date or not db_plan.deadline:
        return None, None, None, None

    span_days = (db_plan.deadline - db_plan.start_date).days
    total_weeks = max(1, ceil(span_days / 7))
    current_week = ((local_today - db_plan.start_date).days // 7) + 1
    current_week = min(max(current_week, 1), total_weeks)

    phase_name = None
    phase_milestone = None

    if db_plan.phase_plan:
        try:
            phases = json.loads(db_plan.phase_plan).get("phases", [])
            for p in phases:
                if p.get("start_week", 0) <= current_week <= p.get("end_week", 0):
                    phase_name = p.get("name")
                    phase_milestone = p.get("milestone")
                    break
        except (ValueError, TypeError):
            pass

    return total_weeks, current_week, phase_name, phase_milestone


def _build_plan_response(
    db_plan: StudyPlan,
    db_sessions: list,
    local_today: date,
) -> PlanResponse:
    """Shared PlanResponse builder."""
    total_weeks, current_week, phase_name, phase_milestone = (
        _compute_arc_metadata(db_plan, local_today)
    )

    if db_plan.deadline:
        days_remaining = max((db_plan.deadline - local_today).days, 0)
    else:
        days_remaining = -1

    today_str = local_today.isoformat()
    upcoming = [s for s in db_sessions if s.date >= today_str]
    upcoming_sorted = sorted(
        upcoming,
        key=lambda s: (s.date, s.time or "00:00"),
    )

    total_duration = sum(s.duration_mins for s in upcoming)
    distinct_days = len({s.date for s in upcoming}) or 1
    distinct_topics = len({s.subject for s in upcoming})

    daily_target = (
        total_duration // distinct_days
        if upcoming
        else DEFAULT_MINUTES_PER_DAY
    )

    stats = PlanStats(
        days_remaining=days_remaining,
        daily_target_mins=daily_target,
        topics_count=distinct_topics,
    )

    week = []
    for i in range(7):
        current_date = local_today + timedelta(days=i)
        date_str = current_date.isoformat()
        day_session = next(
            (s for s in upcoming_sorted if s.date == date_str),
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
            skipped=getattr(s, "skipped", False),
        )
        for s in upcoming_sorted
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
        total_weeks=total_weeks,
        current_week=current_week,
        phase_name=phase_name,
        phase_milestone=phase_milestone,
        updated_at=getattr(db_plan, "updated_at", None),
    )


# ============================================================
# LIST ALL PLANS
# ============================================================

@router.get(
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
        query = query.where(StudyPlan.is_archived == False)  # noqa: E712

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
# GET LATEST PLAN
# ============================================================

@router.get(
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
            StudyPlan.is_archived == False,  # noqa: E712
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

@router.get(
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

@router.post(
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

    resolved_goal_type = resolve_goal_type(
        payload.goal_type,
        current_user.goal_type,
    )
    defaults = persona_defaults(resolved_goal_type)

    minutes_per_day = (
        payload.availability.minutes_per_day
        if payload.availability and payload.availability.minutes_per_day
        else (current_user.daily_goal_mins or DEFAULT_MINUTES_PER_DAY)
    )
    study_days_per_week = (
        payload.availability.study_days_per_week
        if payload.availability and payload.availability.study_days_per_week
        else DEFAULT_STUDY_DAYS_PER_WEEK
    )

    window_days = DEFAULT_WINDOW_DAYS
    if days_remaining is not None and days_remaining < DEFAULT_WINDOW_DAYS:
        window_days = max(days_remaining, 1)

    total_wk = (
        max(1, ceil(days_remaining / 7))
        if days_remaining is not None
        else -1
    )

    phase_data = None
    if total_wk >= 2:
        phase_data = await generate_deepseek_phase_plan(
            goal=payload.goal,
            goal_type=resolved_goal_type,
            total_weeks=total_wk,
            minutes_per_day=minutes_per_day,
            study_days_per_week=study_days_per_week,
        )

    first_phase = (
        phase_data["phases"][0]
        if phase_data and phase_data.get("phases")
        else None
    )

    window_data = await generate_deepseek_study_plan(
        goal=payload.goal,
        target_date=payload.deadline,
        days_remaining=days_remaining,
        goal_type=resolved_goal_type,
        window_start=today,
        window_days=window_days,
        phase_context=first_phase,
        minutes_per_day=minutes_per_day,
        study_days_per_week=study_days_per_week,
    )

    if not window_data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate plan.",
        )

    try:
        db_plan = StudyPlan(
            user_id=current_user.id,
            title=payload.title or payload.goal,
            subject=payload.goal,
            goal_type=resolved_goal_type,
            deadline=payload.deadline,
            emoji=payload.emoji or defaults["emoji"],
            color_hex=payload.color_hex or defaults["color_hex"],
            is_approved=False,
            start_date=today,
            window_start=today,
            window_days=window_days,
            phase_plan=json.dumps(phase_data) if phase_data else None,
        )

        session.add(db_plan)
        session.commit()
        session.refresh(db_plan)

        for ai_session in window_data.get("sessions", []):
            session.add(
                StudySession(
                    plan_id=db_plan.id,
                    user_id=current_user.id,
                    date=ai_session["date"],
                    time=ai_session["time"],
                    subject=ai_session["subject"],
                    duration_mins=ai_session["duration_mins"],
                    mode=ai_session["mode"],
                    priority=ai_session["priority"],
                    completed=False,
                )
            )

        session.commit()

    except Exception:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save your study plan. Please try again.",
        )

    db_sessions = session.exec(
        select(StudySession).where(StudySession.plan_id == db_plan.id)
    ).all()

    return _build_plan_response(db_plan, db_sessions, today)


# ============================================================
# UPDATE PLAN METADATA
# ============================================================

@router.patch(
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
        db_plan.goal_type = resolve_goal_type(payload.goal_type, None)

    session.add(db_plan)
    session.commit()
    session.refresh(db_plan)

    return {
        "message": "Plan updated.",
        "plan_id": db_plan.id,
        "title": db_plan.title,
    }


# ============================================================
# ARCHIVE / UNARCHIVE
# ============================================================

@router.post(
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
# APPROVE — legacy route
# ============================================================

@router.patch(
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
            StudyPlan.is_archived == False,  # noqa: E712
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
# APPROVE — canonical per-plan
# ============================================================

@router.patch(
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
# MANUAL SESSION
# ============================================================

@router.post(
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
# UPDATE SESSION
# ============================================================

@router.patch(
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

    return {
        "id": str(db_session.id),
        "plan_id": db_session.plan_id,
        "date": db_session.date,
        "time": db_session.time,
        "subject": db_session.subject,
        "duration_mins": db_session.duration_mins,
        "mode": normalize_mode(db_session.mode),
        "priority": normalize_priority(db_session.priority),
        "completed": db_session.completed,
        "skipped": db_session.skipped,
    }


# ============================================================
# DELETE SESSION
# ============================================================

@router.delete(
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
# COMPLETE SESSION & GRANT XP
# ============================================================

@router.post(
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
        select(StudySession)
        .where(
            StudySession.id == session_id,
            StudySession.user_id == current_user.id,
        )
        .with_for_update()
    ).first()

    if not db_session or db_session.completed:
        raise HTTPException(
            status_code=400,
            detail="Session invalid or already completed.",
        )

    db_session.completed = True
    session.add(db_session)

    xp_gained = db_session.duration_mins

    pet = session.exec(
        select(Pet).where(Pet.user_id == current_user.id)
    ).first()

    if pet:
        pet.xp += xp_gained
        session.add(pet)

    local_today_date = get_local_now(x_timezone).date()
    local_today = local_today_date.isoformat()

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

    if current_user.last_active_date != local_today_date:
        yesterday = local_today_date - timedelta(days=1)

        if current_user.last_active_date == yesterday:
            current_user.current_streak += 1
        else:
            current_user.current_streak = 1

        if current_user.current_streak > current_user.longest_streak:
            current_user.longest_streak = current_user.current_streak

        current_user.last_active_date = local_today_date
        session.add(current_user)

    session.commit()

    return {"message": "Session completed!", "xp_gained": xp_gained}