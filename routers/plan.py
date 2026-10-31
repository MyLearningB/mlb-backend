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
    PlanSubject,
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
    PlanSubjectInput,
    PlanSubjectResponse,
    PlanUpdateRequest,
    SessionDetail,
    SessionUpdateRequest,
    WeekDay,
    normalize_mode,
    normalize_priority,
)
from security import get_current_user
from ai_service import (
    build_fallback_schedule,
    generate_gemini_multisubject_plan,
    generate_gemini_phase_plan,
    generate_gemini_study_plan,
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
    """Return (total_weeks, current_week, phase_name, phase_milestone)."""
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
    db_subjects: Optional[list] = None,
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

    # ----------------------------------------------------------
    # Week strip — richer session_type
    # ----------------------------------------------------------
    week = []
    for i in range(7):
        current_date = local_today + timedelta(days=i)
        date_str = current_date.isoformat()

        day_sessions = [s for s in upcoming_sorted if s.date == date_str]

        if not day_sessions:
            session_type = "rest"
        elif any(
            normalize_priority(s.priority) == "weak_area"
            for s in day_sessions
        ):
            session_type = "weak_area"
        elif any(normalize_mode(s.mode) == "review" for s in day_sessions):
            session_type = "review"
        else:
            session_type = "study"

        week.append(
            WeekDay(
                date=date_str,
                day_label=current_date.strftime("%a").upper(),
                has_session=bool(day_sessions),
                session_type=session_type,
            )
        )

    formatted_sessions = [
        SessionDetail(
            id=str(s.id),
            date=s.date,
            time=s.time or "16:00",
            subject=s.subject,
            topic=getattr(s, "topic", None),
            subject_id=getattr(s, "subject_id", None),
            duration_mins=s.duration_mins,
            mode=normalize_mode(s.mode),
            priority=normalize_priority(s.priority),
            completed=s.completed,
            skipped=getattr(s, "skipped", False),
        )
        for s in upcoming_sorted
    ]

    subjects_resp = []
    for ps in (db_subjects or []):
        ps_sessions = [s for s in db_sessions if s.subject_id == ps.id]
        try:
            topics_list = json.loads(ps.topics or "[]")
        except (ValueError, TypeError):
            topics_list = []
        try:
            weak_list = json.loads(ps.weak_topics or "[]")
        except (ValueError, TypeError):
            weak_list = []

        subjects_resp.append(
            PlanSubjectResponse(
                id=ps.id,
                name=ps.name,
                exam_date=ps.exam_date,
                weight=ps.weight,
                current_level=ps.current_level,
                topics=topics_list,
                weak_topics=weak_list,
                session_count=len(ps_sessions),
                completed_count=sum(1 for s in ps_sessions if s.completed),
            )
        )

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
        subjects=subjects_resp,
        plan_type=db_plan.plan_type or "single_goal",
        notes=db_plan.notes,
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

    db_subjects = session.exec(
        select(PlanSubject).where(PlanSubject.plan_id == db_plan.id)
    ).all()

    return _build_plan_response(
        db_plan,
        db_sessions,
        get_local_now(x_timezone).date(),
        db_subjects=db_subjects,
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

    db_subjects = session.exec(
        select(PlanSubject).where(PlanSubject.plan_id == db_plan.id)
    ).all()

    return _build_plan_response(
        db_plan,
        db_sessions,
        get_local_now(x_timezone).date(),
        db_subjects=db_subjects,
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

    # ── 1. Normalise input ─────────────────────────────────
    subjects_input = payload.subjects or []
    if not subjects_input:
        if not payload.goal:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provide either `goal` or `subjects[]`.",
            )
        subjects_input = [PlanSubjectInput(name=payload.goal)]

    if len(subjects_input) > 30:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 30 subjects per plan.",
        )

    for s in subjects_input:
        if s.exam_date and s.exam_date < today:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Exam date for '{s.name}' is in the past.",
            )

    # Plan deadline = latest exam date, or explicit payload.deadline
    latest_exam = max(
        (s.exam_date for s in subjects_input if s.exam_date),
        default=None,
    )
    deadline = payload.deadline or latest_exam
    if deadline and (deadline - today).days < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Deadline passed.",
        )
    days_remaining = (deadline - today).days if deadline else None

    # ── 2. Persona + availability ──────────────────────────
    goal_type = resolve_goal_type(payload.goal_type, current_user.goal_type)
    defaults = persona_defaults(goal_type)
    avail = payload.availability
    minutes_per_day = (
        avail.minutes_per_day
        if avail and avail.minutes_per_day
        else (current_user.daily_goal_mins or DEFAULT_MINUTES_PER_DAY)
    )
    study_days = (
        avail.study_days_per_week
        if avail and avail.study_days_per_week
        else DEFAULT_STUDY_DAYS_PER_WEEK
    )
    session_pref = (
        avail.session_length_pref
        if avail and avail.session_length_pref
        else 45
    )
    earliest = (avail.earliest if avail and avail.earliest else "07:00")
    latest = (avail.latest if avail and avail.latest else "22:00")

    window_days = DEFAULT_WINDOW_DAYS
    if days_remaining is not None and days_remaining < DEFAULT_WINDOW_DAYS:
        window_days = max(days_remaining, 1)

    # ── 3. Phase skeleton (only if multi-week) ────────────
    total_wk = (
        max(1, ceil(days_remaining / 7)) if days_remaining is not None else -1
    )
    phase_data = None
    if total_wk >= 2:
        label = (
            subjects_input[0].name
            if len(subjects_input) == 1
            else (payload.title or payload.goal or "multi-subject plan")
        )
        phase_data = await generate_gemini_phase_plan(
            goal=label,
            goal_type=goal_type,
            total_weeks=total_wk,
            minutes_per_day=minutes_per_day,
            study_days_per_week=study_days,
        )
    first_phase = (
        phase_data["phases"][0]
        if phase_data and phase_data.get("phases")
        else None
    )

    # ── 4. Persist plan + subjects ─────────────────────────
    try:
        title = payload.title or (
            subjects_input[0].name
            if len(subjects_input) == 1
            else f"{len(subjects_input)}-subject plan"
        )

        db_plan = StudyPlan(
            user_id=current_user.id,
            title=title[:80],
            subject=subjects_input[0].name,
            goal_type=goal_type,
            deadline=deadline,
            emoji=payload.emoji or defaults["emoji"],
            color_hex=payload.color_hex or defaults["color_hex"],
            is_approved=False,
            start_date=today,
            window_start=today,
            window_days=window_days,
            phase_plan=json.dumps(phase_data) if phase_data else None,
            plan_type=payload.plan_type,
            notes=payload.notes,
        )
        session.add(db_plan)
        session.commit()
        session.refresh(db_plan)

        db_subjects: list[PlanSubject] = []
        for s in subjects_input:
            ps = PlanSubject(
                plan_id=db_plan.id,
                name=s.name[:100],
                exam_date=s.exam_date,
                weight=s.weight,
                current_level=s.current_level,
                topics=json.dumps(s.topics or []),
                weak_topics=json.dumps(s.weak_topics or []),
            )
            session.add(ps)
            db_subjects.append(ps)
        session.commit()
        for ps in db_subjects:
            session.refresh(ps)

    except Exception:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save your study plan. Please try again.",
        )

    # ── 5. Generate sessions (retry → fallback) ───────────
    subjects_dicts = [
        {
            "name": ps.name,
            "exam_date": ps.exam_date,
            "weight": ps.weight,
            "current_level": ps.current_level,
            "topics": json.loads(ps.topics or "[]"),
            "weak_topics": json.loads(ps.weak_topics or "[]"),
        }
        for ps in db_subjects
    ]
    subject_id_by_name = {ps.name: ps.id for ps in db_subjects}

    window_data = None
    for _attempt in (1, 2):
        window_data = await generate_gemini_multisubject_plan(
            subjects=subjects_dicts,
            plan_title=title,
            plan_type=payload.plan_type,
            goal_type=goal_type,
            window_start=today,
            window_days=window_days,
            minutes_per_day=minutes_per_day,
            study_days_per_week=study_days,
            earliest=earliest,
            latest=latest,
            session_length_pref=session_pref,
            phase_context=first_phase,
            notes=payload.notes,
        )
        if window_data and window_data.get("sessions"):
            break

    if not window_data or not window_data.get("sessions"):
        window_data = build_fallback_schedule(
            subjects=subjects_dicts,
            window_start=today,
            window_days=window_days,
            minutes_per_day=minutes_per_day,
            study_days_per_week=study_days,
            session_length_pref=session_pref,
        )

    try:
        for s in window_data["sessions"]:
            session.add(
                StudySession(
                    plan_id=db_plan.id,
                    user_id=current_user.id,
                    date=s["date"],
                    time=s["time"],
                    subject=s["subject_name"],
                    subject_id=subject_id_by_name.get(s["subject_name"]),
                    topic=s.get("topic"),
                    duration_mins=s["duration_mins"],
                    mode=s["mode"],
                    priority=s["priority"],
                    completed=False,
                )
            )
        session.commit()
    except Exception:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save sessions. Please try again.",
        )

    db_sessions = session.exec(
        select(StudySession).where(StudySession.plan_id == db_plan.id)
    ).all()

    return _build_plan_response(
        db_plan,
        db_sessions,
        today,
        db_subjects=db_subjects,
    )


# ============================================================
# EXTEND PLAN WINDOW
# ============================================================
#
# Tops up the rolling window by `window_days` from the day after
# the current last session. This is what fixes the "plan looks
# empty after a week" complaint — the client calls this when the
# user taps "Generate next 14 days".

@router.post(
    "/users/me/plan/{plan_id}/extend",
    response_model=PlanResponse,
    status_code=status.HTTP_200_OK,
)
async def extend_study_plan(
    plan_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    x_timezone: str = Header("UTC"),
):
    today = get_local_now(x_timezone).date()

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
    if db_plan.is_archived:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot extend an archived plan.",
        )

    db_subjects = session.exec(
        select(PlanSubject).where(PlanSubject.plan_id == plan_id)
    ).all()
    if not db_subjects:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Plan has no subjects to extend.",
        )

    # Window starts the day after the latest existing session
    last = session.exec(
        select(StudySession)
        .where(StudySession.plan_id == plan_id)
        .order_by(StudySession.date.desc())
    ).first()

    window_start = today
    if last:
        try:
            tail = date.fromisoformat(last.date)
            window_start = max(today, tail + timedelta(days=1))
        except (ValueError, TypeError):
            pass

    window_days = db_plan.window_days or DEFAULT_WINDOW_DAYS
    if db_plan.deadline:
        remaining = (db_plan.deadline - window_start).days
        if remaining <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Plan deadline has passed.",
            )
        window_days = min(window_days, remaining)

    minutes_per_day = current_user.daily_goal_mins or DEFAULT_MINUTES_PER_DAY
    study_days = DEFAULT_STUDY_DAYS_PER_WEEK

    subjects_dicts = [
        {
            "name": ps.name,
            "exam_date": ps.exam_date,
            "weight": ps.weight,
            "current_level": ps.current_level,
            "topics": json.loads(ps.topics or "[]"),
            "weak_topics": json.loads(ps.weak_topics or "[]"),
        }
        for ps in db_subjects
    ]
    subject_id_by_name = {ps.name: ps.id for ps in db_subjects}

    recent = session.exec(
        select(StudySession)
        .where(StudySession.plan_id == plan_id)
        .order_by(StudySession.date.desc())
        .limit(20)
    ).all()
    recent_topics = [
        s.topic or s.subject for s in recent if (s.topic or s.subject)
    ]

    window_data = await generate_gemini_multisubject_plan(
        subjects=subjects_dicts,
        plan_title=db_plan.title or db_plan.subject,
        plan_type=db_plan.plan_type or "single_goal",
        goal_type=db_plan.goal_type,
        window_start=window_start,
        window_days=window_days,
        minutes_per_day=minutes_per_day,
        study_days_per_week=study_days,
        notes=db_plan.notes,
        recent_topics=recent_topics,
    )
    if not window_data or not window_data.get("sessions"):
        window_data = build_fallback_schedule(
            subjects=subjects_dicts,
            window_start=window_start,
            window_days=window_days,
            minutes_per_day=minutes_per_day,
            study_days_per_week=study_days,
        )

    try:
        for s in window_data["sessions"]:
            session.add(
                StudySession(
                    plan_id=plan_id,
                    user_id=current_user.id,
                    date=s["date"],
                    time=s["time"],
                    subject=s["subject_name"],
                    subject_id=subject_id_by_name.get(s["subject_name"]),
                    topic=s.get("topic"),
                    duration_mins=s["duration_mins"],
                    mode=s["mode"],
                    priority=s["priority"],
                    completed=False,
                )
            )
        db_plan.window_start = window_start
        session.add(db_plan)
        session.commit()
    except Exception:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not extend plan. Please try again.",
        )

    db_sessions = session.exec(
        select(StudySession).where(StudySession.plan_id == plan_id)
    ).all()

    return _build_plan_response(
        db_plan,
        db_sessions,
        today,
        db_subjects=db_subjects,
    )


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
        "message": (
            "Plan archived." if payload.archived else "Plan restored."
        ),
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

    # Auto-link to a PlanSubject row if the name matches.
    subject_row = session.exec(
        select(PlanSubject).where(
            PlanSubject.plan_id == db_plan.id,
            PlanSubject.name == payload.subject,
        )
    ).first()

    new_session = StudySession(
        plan_id=db_plan.id,
        user_id=current_user.id,
        date=payload.date,
        time=payload.time,
        subject=payload.subject,
        subject_id=subject_row.id if subject_row else None,
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
            "subject_id": new_session.subject_id,
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
        "topic": db_session.topic,
        "subject_id": db_session.subject_id,
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