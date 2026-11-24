from datetime import date, timedelta
from typing import List, Optional
import json

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Header,
    HTTPException,
    Request,
    status,
)
from sqlmodel import Session, select

from database import get_session
from models import (
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
    PlanListItem,
    PlanListResponse,
    PlanReplanRequest,
    PlanResponse,
    PlanScheduleUpdate,
    PlanUpdateRequest,
    SessionUpdateRequest,
    UpcomingResponse,
    UpcomingSession,
    normalize_mode,
    normalize_priority,
)
from security import get_current_user
from routers.auth import limiter
from gamification import award_xp
from utils import (
    get_local_now,
    persona_defaults,
    resolve_goal_type,
)
import plan_scheduler as ps
import plan_service as svc
from plan_rollover import refine_if_needed


router = APIRouter(tags=["Study Plan"])


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_MINUTES_PER_DAY = 45
MAX_ACTIVE_PLANS = 8


# ============================================================
# HELPERS
# ============================================================

def _get_owned_plan(session: Session, plan_id: int, user_id: int) -> StudyPlan:
    plan = session.exec(
        select(StudyPlan).where(
            StudyPlan.id == plan_id,
            StudyPlan.user_id == user_id,
        )
    ).first()
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Study plan not found.",
        )
    return plan


def _response(session: Session, plan: StudyPlan, today: date, user_id: int) -> PlanResponse:
    subjects = session.exec(
        select(PlanSubject).where(PlanSubject.plan_id == plan.id)
    ).all()
    return svc.build_plan_response(
        session, plan, today, user_id=user_id, db_subjects=subjects
    )


def _validate_deadline(deadline: Optional[date], today: date) -> None:
    if deadline is None:
        return
    if deadline < today:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="That date has already passed.",
        )
    if (deadline - today).days > ps.MAX_HORIZON_DAYS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pick a date within 5 years.",
        )


def _times_or_400(raw: Optional[List[str]]) -> Optional[List[str]]:
    if raw is None:
        return None
    times = ps.parse_times(raw)
    if not times:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Study times must look like 08:15.",
        )
    return times


def _days_or_400(raw: Optional[List[int]]) -> Optional[List[int]]:
    if raw is None:
        return None
    days = ps.parse_days(raw)
    if not days:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pick at least one study day.",
        )
    return days


def _reschedule(session: Session, plan: StudyPlan, today: date) -> None:
    """Replace upcoming scheduler sessions using the current settings."""
    svc.clear_future_plan_sessions(session, plan, today)
    svc.fill_sessions(
        session,
        plan,
        today=today,
        from_date=today,
        through=today + timedelta(days=svc.FILL_AHEAD_DAYS - 1),
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

    plans = session.exec(query.order_by(StudyPlan.created_at.desc())).all()

    items: list[PlanListItem] = []
    for plan in plans:
        sessions = session.exec(
            select(StudySession).where(StudySession.plan_id == plan.id)
        ).all()
        days_remaining = (
            max((plan.deadline - today).days, 0) if plan.deadline else None
        )
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
                completed_count=sum(1 for s in sessions if s.completed),
            )
        )

    return PlanListResponse(plans=items, total_count=len(items))


# ============================================================
# UPCOMING SESSIONS (for phone notifications)
# ============================================================
#
# Declared before /users/me/plan/{plan_id} routes on purpose. The
# app calls this on open and after any plan change, then schedules a
# local notification for every session it returns.

@router.get(
    "/users/me/plan/upcoming",
    response_model=UpcomingResponse,
    status_code=status.HTTP_200_OK,
)
def upcoming_sessions(
    days: int = 14,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    x_timezone: str = Header("UTC"),
):
    days = max(1, min(days, 21))
    today = get_local_now(x_timezone).date()
    svc.ensure_user_plans_filled(session, current_user.id, today)

    plans = {
        p.id: p
        for p in session.exec(
            select(StudyPlan).where(
                StudyPlan.user_id == current_user.id,
                StudyPlan.is_archived == False,  # noqa: E712
            )
        ).all()
    }
    if not plans:
        return UpcomingResponse(sessions=[])

    rows = session.exec(
        select(StudySession)
        .where(
            StudySession.user_id == current_user.id,
            StudySession.date >= today.isoformat(),
            StudySession.date <= (today + timedelta(days=days - 1)).isoformat(),
            StudySession.completed == False,  # noqa: E712
            StudySession.skipped == False,  # noqa: E712
        )
        .order_by(StudySession.date, StudySession.time)
    ).all()

    out = []
    for s in rows:
        plan = plans.get(s.plan_id)
        if plan is None:
            continue
        out.append(
            UpcomingSession(
                id=str(s.id),
                plan_id=plan.id,
                plan_title=plan.title or plan.subject,
                emoji=plan.emoji,
                date=s.date,
                time=s.time or "18:00",
                duration_mins=s.duration_mins,
                topic=s.topic,
                task=s.task,
                mode=normalize_mode(s.mode),
            )
        )
    return UpcomingResponse(sessions=out[:200])


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
    today = get_local_now(x_timezone).date()
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

    if svc.ensure_filled(session, db_plan, today):
        session.commit()
        session.refresh(db_plan)

    return _response(session, db_plan, today, current_user.id)


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
    today = get_local_now(x_timezone).date()
    db_plan = _get_owned_plan(session, plan_id, current_user.id)

    if svc.ensure_filled(session, db_plan, today):
        session.commit()
        session.refresh(db_plan)

    return _response(session, db_plan, today, current_user.id)


# ============================================================
# GENERATE STUDY PLAN
# ============================================================
#
# One AI call (or zero, when a cached roadmap matches) builds the
# whole curriculum. Sessions for the next 14 days are created by
# plan_scheduler at the user's own times. Plans are active straight
# away — no separate approve step needed.

@router.post(
    "/users/me/plan/generate",
    response_model=PlanResponse,
    status_code=status.HTTP_200_OK,
)
@limiter.limit("5/minute")
async def generate_study_plan(
    request: Request,
    payload: PlanGenerateRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    x_timezone: str = Header("UTC"),
):
    today = get_local_now(x_timezone).date()

    # ── 1. What is the goal? ────────────────────────────────
    goal = (payload.goal or "").strip()
    notes = (payload.notes or "").strip() or None
    deadline = payload.deadline

    if payload.subjects:
        # Legacy multi-subject sheet: one plan covering all of them.
        names = [s.name.strip() for s in payload.subjects if s.name.strip()][:10]
        if not goal or len(names) > 1:
            goal = ", ".join(names)
        topic_bits = []
        for s in payload.subjects:
            if s.topics:
                topic_bits.append(f"{s.name}: {', '.join(s.topics[:8])}")
            if s.weak_topics:
                topic_bits.append(f"{s.name} weak: {', '.join(s.weak_topics[:6])}")
        if topic_bits:
            notes = ((notes + ". ") if notes else "") + "; ".join(topic_bits)
        if deadline is None:
            exam_dates = [s.exam_date for s in payload.subjects if s.exam_date]
            deadline = max(exam_dates) if exam_dates else None

    goal = goal[:200]
    if not goal:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tell us what you want to learn or prepare for.",
        )
    _validate_deadline(deadline, today)

    active = session.exec(
        select(StudyPlan.id).where(
            StudyPlan.user_id == current_user.id,
            StudyPlan.is_archived == False,  # noqa: E712
        )
    ).all()
    if len(active) >= MAX_ACTIVE_PLANS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"You can have {MAX_ACTIVE_PLANS} active plans. Archive one first.",
        )

    # ── 2. Schedule settings ───────────────────────────────
    goal_type = resolve_goal_type(payload.goal_type, current_user.goal_type)
    defaults = persona_defaults(goal_type)
    avail = payload.availability

    level = payload.level or "beginner"
    if not payload.level and payload.subjects:
        level = payload.subjects[0].current_level or "beginner"

    minutes = (
        payload.minutes_per_day
        or (avail.minutes_per_day if avail else None)
        or current_user.daily_goal_mins
        or DEFAULT_MINUTES_PER_DAY
    )
    minutes = max(10, min(int(minutes), 600))

    days = _days_or_400(payload.days_of_week)
    if days is None:
        per_week = avail.study_days_per_week if avail and avail.study_days_per_week else 6
        days = list(range(1, per_week + 1))

    times = _times_or_400(payload.preferred_times)
    if times is None:
        first = avail.earliest if avail and avail.earliest else None
        times = ps.parse_times([first]) or [ps.DEFAULT_TIMES.get(goal_type, "18:00")]

    kind_hint = ps.guess_goal_kind(goal)
    total_weeks = ps.horizon_weeks(today, deadline, kind_hint, level)

    # ── 3. Roadmap: cache → AI → fallback ──────────────────
    roadmap, source = await svc.obtain_roadmap(
        session,
        current_user,
        goal=goal,
        level=level,
        total_weeks=total_weeks,
        has_deadline=deadline is not None,
        minutes_per_day=minutes,
        days_per_week=len(days),
        times_count=len(times),
        goal_type=goal_type,
        today=today,
        notes=notes,
    )

    # ── 4. Save the plan + first 14 days ───────────────────
    try:
        title = (payload.title or roadmap.get("title") or goal)[:80]
        db_plan = StudyPlan(
            user_id=current_user.id,
            title=title,
            subject=goal,
            goal_type=goal_type,
            deadline=deadline,
            emoji=payload.emoji or roadmap.get("emoji") or defaults["emoji"],
            color_hex=payload.color_hex or defaults["color_hex"],
            is_approved=True,
            start_date=today,
            window_start=today,
            window_days=svc.FILL_AHEAD_DAYS,
            plan_type=payload.plan_type or roadmap["goal_kind"],
            notes=notes,
            goal_kind=roadmap["goal_kind"],
            level=level,
            minutes_per_day=minutes,
            days_of_week=json.dumps(days),
            preferred_times=json.dumps(times),
            roadmap=json.dumps(roadmap),
            roadmap_source=source,
            roadmap_start=today,
        )
        session.add(db_plan)
        session.commit()
        session.refresh(db_plan)

        svc.fill_sessions(
            session,
            db_plan,
            today=today,
            from_date=today,
            through=today + timedelta(days=svc.FILL_AHEAD_DAYS - 1),
        )

        # Keep the profile in sync so the dashboard countdown and the
        # goal card reflect the newest plan.
        if not current_user.study_goal:
            current_user.study_goal = goal[:200]
        if deadline and not current_user.target_date:
            current_user.target_date = deadline
        session.add(current_user)
        session.commit()
        session.refresh(db_plan)
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        print(f"Plan save failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save your study plan. Please try again.",
        )

    # Multi-year plans: detail the first long phase right away (one
    # AI call, after the response is sent) instead of waiting for the
    # hourly loop.
    if source == "ai" and total_weeks > 26:
        background_tasks.add_task(refine_if_needed, db_plan.id, today)

    return _response(session, db_plan, today, current_user.id)


# ============================================================
# CHANGE SCHEDULE (free — no AI)
# ============================================================

@router.patch(
    "/users/me/plan/{plan_id}/schedule",
    response_model=PlanResponse,
    status_code=status.HTTP_200_OK,
)
def update_plan_schedule(
    plan_id: int,
    payload: PlanScheduleUpdate,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    x_timezone: str = Header("UTC"),
):
    today = get_local_now(x_timezone).date()
    plan = _get_owned_plan(session, plan_id, current_user.id)
    if plan.is_archived:
        raise HTTPException(status_code=400, detail="Restore this plan first.")

    svc.upgrade_legacy_plan(session, plan, today)

    days = _days_or_400(payload.days_of_week)
    times = _times_or_400(payload.preferred_times)
    if payload.minutes_per_day is not None:
        plan.minutes_per_day = payload.minutes_per_day
    if days is not None:
        plan.days_of_week = json.dumps(days)
    if times is not None:
        plan.preferred_times = json.dumps(times)
    if payload.clear_deadline:
        plan.deadline = None
    elif payload.deadline is not None:
        _validate_deadline(payload.deadline, today)
        plan.deadline = payload.deadline

    _reschedule(session, plan, today)
    session.commit()
    session.refresh(plan)
    return _response(session, plan, today, current_user.id)


# ============================================================
# RE-PLAN WITH AI (uses one AI credit)
# ============================================================

@router.post(
    "/users/me/plan/{plan_id}/replan",
    response_model=PlanResponse,
    status_code=status.HTTP_200_OK,
)
@limiter.limit("3/minute")
async def replan_study_plan(
    request: Request,
    plan_id: int,
    payload: PlanReplanRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    x_timezone: str = Header("UTC"),
):
    today = get_local_now(x_timezone).date()
    plan = _get_owned_plan(session, plan_id, current_user.id)
    if plan.is_archived:
        raise HTTPException(status_code=400, detail="Restore this plan first.")

    if svc.ai_plans_left(session, current_user.id, today) <= 0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="You've used today's AI plan builds. You can still change times "
                   "and days for free, or re-plan tomorrow.",
        )
    if plan.deadline and plan.deadline < today:
        raise HTTPException(status_code=400, detail="This plan's date has passed. Set a new date first.")

    svc.upgrade_legacy_plan(session, plan, today)
    if payload.level:
        plan.level = payload.level

    summary = svc.progress_summary(session, plan, today)
    days = svc.plan_days(plan)
    kind = plan.goal_kind or ps.guess_goal_kind(plan.subject)
    remaining_weeks = ps.horizon_weeks(today, plan.deadline, kind, plan.level or "beginner")
    if plan.deadline is None:
        old = svc.load_roadmap(plan)
        if old:
            end = ps.roadmap_end(svc.plan_start(plan), None, old)
            remaining_weeks = max(2, -(-((end - today).days + 1) // 7))

    roadmap, source = await svc.obtain_roadmap(
        session,
        current_user,
        goal=plan.subject,
        level=plan.level or "beginner",
        total_weeks=remaining_weeks,
        has_deadline=plan.deadline is not None,
        minutes_per_day=plan.minutes_per_day or DEFAULT_MINUTES_PER_DAY,
        days_per_week=len(days),
        times_count=len(svc.plan_times(plan)),
        goal_type=plan.goal_type,
        today=today,
        notes=plan.notes,
        progress_summary=summary,
        feedback=(payload.feedback or "").strip() or None,
    )
    if source != "ai":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The AI is busy right now — your plan is unchanged. Try again in a minute.",
        )

    plan.roadmap = json.dumps(roadmap)
    plan.roadmap_source = source
    plan.roadmap_start = today
    plan.goal_kind = roadmap["goal_kind"]
    plan.arc_version = (plan.arc_version or 1) + 1
    _reschedule(session, plan, today)
    session.commit()
    session.refresh(plan)
    if remaining_weeks > 26:
        background_tasks.add_task(refine_if_needed, plan.id, today)
    return _response(session, plan, today, current_user.id)


# ============================================================
# EXTEND (top up the next 14 days — no AI)
# ============================================================

@router.post(
    "/users/me/plan/{plan_id}/extend",
    response_model=PlanResponse,
    status_code=status.HTTP_200_OK,
)
def extend_study_plan(
    plan_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    x_timezone: str = Header("UTC"),
):
    today = get_local_now(x_timezone).date()
    plan = _get_owned_plan(session, plan_id, current_user.id)
    if plan.is_archived:
        raise HTTPException(status_code=400, detail="Cannot extend an archived plan.")
    if plan.deadline and plan.deadline < today:
        raise HTTPException(status_code=400, detail="Plan deadline has passed.")

    start = max(today, (plan.scheduled_through or today - timedelta(days=1)) + timedelta(days=1))
    svc.fill_sessions(
        session, plan, today=today, from_date=start,
        through=start + timedelta(days=svc.FILL_AHEAD_DAYS - 1),
    )
    session.commit()
    session.refresh(plan)
    return _response(session, plan, today, current_user.id)


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
    x_timezone: str = Header("UTC"),
):
    today = get_local_now(x_timezone).date()
    db_plan = _get_owned_plan(session, plan_id, current_user.id)

    if payload.title is not None:
        db_plan.title = payload.title.strip()[:80] or db_plan.title
    if payload.emoji is not None:
        db_plan.emoji = payload.emoji
    if payload.color_hex is not None:
        db_plan.color_hex = payload.color_hex
    if payload.goal_type is not None:
        db_plan.goal_type = resolve_goal_type(payload.goal_type, None)
    if payload.deadline is not None and payload.deadline != db_plan.deadline:
        _validate_deadline(payload.deadline, today)
        db_plan.deadline = payload.deadline
        svc.upgrade_legacy_plan(session, db_plan, today)
        _reschedule(session, db_plan, today)

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
    x_timezone: str = Header("UTC"),
):
    today = get_local_now(x_timezone).date()
    db_plan = _get_owned_plan(session, plan_id, current_user.id)

    db_plan.is_archived = payload.archived
    if payload.archived:
        # Archived plans stop producing sessions and reminders.
        svc.clear_future_plan_sessions(session, db_plan, today)
    session.add(db_plan)
    session.commit()

    return {
        "message": "Plan archived." if payload.archived else "Plan restored.",
        "plan_id": db_plan.id,
        "is_archived": db_plan.is_archived,
    }


# ============================================================
# APPROVE — kept for older app versions (plans are now active
# as soon as they are created)
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
        raise HTTPException(status_code=404, detail="No plan to approve.")

    db_plan.is_approved = request.approved
    session.add(db_plan)
    session.commit()
    return {
        "message": f"Plan {db_plan.id} approved status updated",
        "approved": db_plan.is_approved,
        "plan_id": db_plan.id,
    }


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
    db_plan = _get_owned_plan(session, plan_id, current_user.id)
    db_plan.is_approved = request.approved
    session.add(db_plan)
    session.commit()
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
    db_plan = _get_owned_plan(session, plan_id, current_user.id)

    if not ps.parse_times([payload.time]):
        raise HTTPException(status_code=400, detail="Time must look like 08:15.")
    try:
        date.fromisoformat(payload.date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Date must look like 2026-01-31.")

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
        time=ps.parse_times([payload.time])[0],
        subject=db_plan.title or db_plan.subject,
        topic=payload.subject,
        task=payload.subject,
        subject_id=subject_row.id if subject_row else None,
        duration_mins=payload.duration_mins,
        mode=normalize_mode(payload.mode),
        priority=normalize_priority(payload.priority),
        completed=False,
        source="manual",
    )

    session.add(new_session)
    session.commit()
    session.refresh(new_session)

    return {
        "message": "Session added successfully",
        "session": {
            "id": str(new_session.id),
            "date": new_session.date,
            "time": new_session.time,
            "subject": new_session.subject,
            "topic": new_session.topic,
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
        raise HTTPException(status_code=404, detail="Session not found.")

    if request.scheduled_time is not None:
        cleaned = ps.parse_times([request.scheduled_time])
        if not cleaned:
            raise HTTPException(status_code=400, detail="Time must look like 08:15.")
        db_session.time = cleaned[0]

    if request.duration_mins is not None:
        db_session.duration_mins = request.duration_mins

    if request.skipped is not None:
        if db_session.completed and request.skipped:
            raise HTTPException(status_code=400, detail="This session is already done.")
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
        "task": db_session.task,
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
        raise HTTPException(status_code=404, detail="Session not found.")

    if db_session.source == "plan" and not db_session.completed:
        # Deleting a scheduler session would just be regenerated on the
        # next top-up, so mark it skipped instead.
        db_session.skipped = True
        session.add(db_session)
    else:
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
    local_now = get_local_now(x_timezone)
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

    # Sessions can be done early on the same day, or caught up later,
    # but not ticked off days in advance (XP farming).
    try:
        if date.fromisoformat(db_session.date) > local_now.date():
            raise HTTPException(
                status_code=400,
                detail="You can complete this session on its day.",
            )
    except ValueError:
        pass

    db_session.completed = True
    db_session.skipped = False
    session.add(db_session)

    # One minute studied = one base XP (capped). award_xp adds the
    # pet's trait bonus, levels the pet up, updates streak and quests.
    minutes = min(db_session.duration_mins or 0, 180)
    reward = award_xp(
        session,
        current_user,
        minutes,
        local_now=local_now,
        source="plan_session",
        duration_mins=minutes,
        minutes_studied=minutes,
    )

    session.commit()

    return {
        "message": "Session completed!",
        "xp_gained": reward["xp_earned"],
        "reward": reward,
    }