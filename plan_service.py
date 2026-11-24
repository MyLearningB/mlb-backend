"""
plan_service.py — database helpers shared by routers/plan.py,
plan_rollover.py and the dashboard in main.py.

Key ideas
---------
* A plan stores ONE roadmap (JSON). Sessions are generated from it by
  plan_scheduler.build_sessions — no AI.
* Sessions are kept filled ~14 days ahead. `ensure_filled` runs lazily
  whenever the app reads the plan or dashboard, so it works even when
  Render's free instance was asleep and the hourly loop never ran.
* AI is used only to create a roadmap (new plan / re-plan) or to
  detail one long phase of a multi-year plan. Each user gets
  PLAN_AI_DAILY_LIMIT builds per day; identical goals reuse a cached
  roadmap for free.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Tuple

from sqlmodel import Session, select

import plan_scheduler as ps
from ai_service import generate_plan_roadmap
from models import AiUsage, PlanRoadmapCache, StudyPlan, StudySession, User
from schemas import (
    PlanGoal,
    PlanResponse,
    PlanSchedule,
    PlanStats,
    RoadmapPhase,
    SessionDetail,
    WeekDay,
    normalize_mode,
    normalize_priority,
)

PLAN_AI_DAILY_LIMIT = int(os.getenv("PLAN_AI_DAILY_LIMIT", "5"))
FILL_AHEAD_DAYS = 14          # keep this many days of sessions ready
REFILL_WHEN_LEFT = 10         # top up when fewer days than this remain
CACHE_MAX_AGE_DAYS = 120
DEFAULT_DAYS = [1, 2, 3, 4, 5, 6]


# ============================================================
# SMALL ACCESSORS
# ============================================================

def _loads(raw, default):
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return default


def plan_days(plan: StudyPlan) -> List[int]:
    return ps.parse_days(_loads(plan.days_of_week, None)) or list(DEFAULT_DAYS)


def plan_times(plan: StudyPlan) -> List[str]:
    return ps.parse_times(_loads(plan.preferred_times, None)) or ["18:00"]


def plan_start(plan: StudyPlan) -> date:
    if plan.roadmap_start:
        return plan.roadmap_start
    if plan.start_date:
        return plan.start_date
    created = plan.created_at or datetime.now(timezone.utc)
    return created.date()


def load_roadmap(plan: StudyPlan) -> Optional[dict]:
    rm = _loads(plan.roadmap, None)
    return rm if isinstance(rm, dict) and rm.get("phases") else None


def is_active(plan: StudyPlan) -> bool:
    return not plan.is_archived


# ============================================================
# LEGACY UPGRADE (plans made before v2 have no roadmap)
# ============================================================

def upgrade_legacy_plan(db: Session, plan: StudyPlan, today: date) -> Optional[dict]:
    """Give an old plan a roadmap without spending AI credits.

    Uses its old phase skeleton when present, otherwise a generic
    roadmap for its goal. Existing sessions are kept; new ones start
    after the last existing session.
    """
    rm = load_roadmap(plan)
    if rm:
        return rm

    goal = plan.subject or plan.title or "My goal"
    kind = plan.goal_kind or ps.guess_goal_kind(goal)
    start = plan.start_date or today
    weeks = ps.horizon_weeks(start, plan.deadline, kind, plan.level or "beginner")

    rm = ps.normalize_roadmap(
        _loads(plan.phase_plan, None), total_weeks=weeks, goal=goal, kind_hint=kind
    ) or ps.fallback_roadmap(goal, kind, plan.level or "beginner", weeks)

    last = db.exec(
        select(StudySession.date)
        .where(StudySession.plan_id == plan.id)
        .order_by(StudySession.date.desc())
    ).first()

    plan.roadmap = json.dumps(rm)
    plan.roadmap_source = "legacy"
    plan.goal_kind = rm["goal_kind"]
    plan.roadmap_start = start
    plan.is_approved = True
    if last:
        try:
            plan.scheduled_through = date.fromisoformat(last)
        except ValueError:
            pass
    db.add(plan)
    return rm


# ============================================================
# AI QUOTA + CACHE
# ============================================================

def ai_plans_left(db: Session, user_id: int, today: date) -> int:
    row = db.exec(
        select(AiUsage).where(
            AiUsage.user_id == user_id,
            AiUsage.day == today,
            AiUsage.feature == "plan",
        )
    ).first()
    used = row.count if row else 0
    return max(0, PLAN_AI_DAILY_LIMIT - used)


def consume_ai(db: Session, user_id: int, today: date, feature: str = "plan") -> None:
    row = db.exec(
        select(AiUsage).where(
            AiUsage.user_id == user_id,
            AiUsage.day == today,
            AiUsage.feature == feature,
        )
    ).first()
    if row is None:
        row = AiUsage(user_id=user_id, day=today, feature=feature, count=0)
    row.count += 1
    db.add(row)
    db.commit()


def _norm_goal(goal: str) -> str:
    g = re.sub(r"[^a-z0-9 ]+", " ", goal.lower())
    return re.sub(r"\s+", " ", g).strip()[:120]


def _bucket(weeks: int) -> str:
    for limit in (2, 4, 8, 12, 20, 30, 45, 60, 90, 130, 180, 260):
        if weeks <= limit:
            return str(limit)
    return "max"


def cache_key(goal: str, level: str, total_weeks: int, minutes: int, has_deadline: bool) -> str:
    mins = "s" if minutes < 30 else ("m" if minutes < 75 else "l")
    raw = f"{_norm_goal(goal)}|{level}|{_bucket(total_weeks)}|{mins}|{int(has_deadline)}"
    return hashlib.sha1(raw.encode()).hexdigest()


def cache_get(db: Session, key: str, total_weeks: int, goal: str) -> Optional[dict]:
    row = db.exec(select(PlanRoadmapCache).where(PlanRoadmapCache.cache_key == key)).first()
    if not row:
        return None
    created = row.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) - created > timedelta(days=CACHE_MAX_AGE_DAYS):
        return None
    data = _loads(row.roadmap, None)
    rm = ps.normalize_roadmap(
        data, total_weeks=total_weeks, goal=goal,
        kind_hint=(data or {}).get("goal_kind", "skill"),
    )
    if rm:
        row.hits += 1
        db.add(row)
    return rm


def cache_put(db: Session, key: str, roadmap: dict) -> None:
    row = db.exec(select(PlanRoadmapCache).where(PlanRoadmapCache.cache_key == key)).first()
    if row is None:
        row = PlanRoadmapCache(cache_key=key, roadmap=json.dumps(roadmap))
    else:
        row.roadmap = json.dumps(roadmap)
        row.created_at = datetime.now(timezone.utc)
    db.add(row)


async def obtain_roadmap(
    db: Session,
    user: User,
    *,
    goal: str,
    level: str,
    total_weeks: int,
    has_deadline: bool,
    minutes_per_day: int,
    days_per_week: int,
    times_count: int,
    goal_type: Optional[str],
    today: date,
    notes: Optional[str] = None,
    progress_summary: Optional[str] = None,
    feedback: Optional[str] = None,
) -> Tuple[dict, str]:
    """Cache -> AI (if quota) -> generic fallback. Returns (roadmap, source)."""
    kind_hint = ps.guess_goal_kind(goal)
    personal = bool(notes or progress_summary or feedback)
    key = cache_key(goal, level, total_weeks, minutes_per_day, has_deadline)

    if not personal:
        cached = cache_get(db, key, total_weeks, goal)
        if cached:
            return cached, "cache"

    if ai_plans_left(db, user.id, today) > 0:
        slots = ps.slots_for(minutes_per_day, ["x"] * max(1, times_count))
        raw = await generate_plan_roadmap(
            goal,
            kind_hint=kind_hint,
            level=level,
            total_weeks=total_weeks,
            has_deadline=has_deadline,
            minutes_per_day=minutes_per_day,
            days_per_week=days_per_week,
            session_mins=ps.split_minutes(minutes_per_day, slots)[0],
            goal_type=goal_type,
            today=today,
            notes=notes,
            progress_summary=progress_summary,
            feedback=feedback,
        )
        rm = ps.normalize_roadmap(raw, total_weeks=total_weeks, goal=goal, kind_hint=kind_hint)
        if rm:
            consume_ai(db, user.id, today)
            if not personal:
                cache_put(db, key, rm)
            return rm, "ai"

    return ps.fallback_roadmap(goal, kind_hint, level, total_weeks), "fallback"


# ============================================================
# FILLING SESSIONS (no AI)
# ============================================================

def weak_topics(db: Session, plan: StudyPlan, today: date) -> List[str]:
    """Topics the user skipped recently and has not done since."""
    since = (today - timedelta(days=10)).isoformat()
    rows = db.exec(
        select(StudySession).where(
            StudySession.plan_id == plan.id,
            StudySession.date >= since,
            StudySession.date < today.isoformat(),
        )
    ).all()
    done = {r.topic for r in rows if r.completed and r.topic}
    out: List[str] = []
    for r in sorted(rows, key=lambda r: r.date, reverse=True):
        t = r.topic
        if r.skipped and t and t not in done and t not in out and not t.startswith(("Weekly", "Milestone", "Exam eve")):
            out.append(t)
    return out[:3]


def fill_sessions(
    db: Session,
    plan: StudyPlan,
    *,
    today: date,
    through: date,
    from_date: Optional[date] = None,
) -> int:
    """Create scheduler sessions for dates in [from_date, through] that
    have none yet. Caller commits. Returns number of sessions added."""
    rm = load_roadmap(plan) or upgrade_legacy_plan(db, plan, today)
    if not rm:
        return 0

    if from_date is None:
        from_date = (
            plan.scheduled_through + timedelta(days=1)
            if plan.scheduled_through
            else today
        )
    from_date = max(from_date, today)
    if plan.deadline:
        through = min(through, plan.deadline)
    if from_date > through:
        return 0

    taken = {
        d for d in db.exec(
            select(StudySession.date).where(
                StudySession.plan_id == plan.id,
                StudySession.source == "plan",
                StudySession.date >= from_date.isoformat(),
                StudySession.date <= through.isoformat(),
            )
        ).all()
    }

    specs = ps.build_sessions(
        roadmap=rm,
        start=plan_start(plan),
        deadline=plan.deadline,
        days=plan_days(plan),
        times=plan_times(plan),
        minutes_per_day=plan.minutes_per_day or 45,
        level=plan.level or "beginner",
        subject=plan.title or plan.subject,
        from_date=from_date,
        to_date=through,
        weak_topics=weak_topics(db, plan, today),
    )

    added = 0
    for s in specs:
        if s["date"] in taken:
            continue
        db.add(
            StudySession(
                plan_id=plan.id,
                user_id=plan.user_id,
                date=s["date"],
                time=s["time"],
                subject=s["subject"] or plan.subject,
                topic=s["topic"] or None,
                task=s["task"] or None,
                duration_mins=s["duration_mins"],
                mode=s["mode"],
                priority=s["priority"],
                completed=False,
                source="plan",
            )
        )
        added += 1

    if plan.scheduled_through is None or through > plan.scheduled_through:
        plan.scheduled_through = through
    plan.window_start = from_date
    plan.window_days = FILL_AHEAD_DAYS
    db.add(plan)
    return added


def ensure_filled(db: Session, plan: StudyPlan, today: date) -> bool:
    """Top up a plan to FILL_AHEAD_DAYS when it is running low."""
    if not is_active(plan):
        return False
    if plan.deadline and plan.deadline < today:
        return False
    target = today + timedelta(days=FILL_AHEAD_DAYS - 1)
    if plan.scheduled_through and plan.scheduled_through >= today + timedelta(days=REFILL_WHEN_LEFT):
        return False
    if plan.deadline and plan.scheduled_through and plan.scheduled_through >= plan.deadline:
        return False  # already filled to the end
    fill_sessions(db, plan, today=today, through=target)
    return True


def ensure_user_plans_filled(db: Session, user_id: int, today: date) -> None:
    """Called from GET endpoints. Never raises."""
    try:
        plans = db.exec(
            select(StudyPlan).where(
                StudyPlan.user_id == user_id,
                StudyPlan.is_archived == False,  # noqa: E712
            )
        ).all()
        changed = False
        for p in plans:
            changed = ensure_filled(db, p, today) or changed
        if changed:
            db.commit()
    except Exception as e:  # pragma: no cover - defensive
        print(f"ensure_user_plans_filled failed: {e}")
        db.rollback()


def clear_future_plan_sessions(db: Session, plan: StudyPlan, from_date: date) -> None:
    """Delete not-yet-done scheduler sessions from `from_date` on.
    Completed, skipped-in-the-past and manual sessions are kept."""
    rows = db.exec(
        select(StudySession).where(
            StudySession.plan_id == plan.id,
            StudySession.source == "plan",
            StudySession.completed == False,  # noqa: E712
            StudySession.date >= from_date.isoformat(),
        )
    ).all()
    for r in rows:
        db.delete(r)
    plan.scheduled_through = from_date - timedelta(days=1)
    db.add(plan)


def progress_summary(db: Session, plan: StudyPlan, today: date) -> str:
    rows = db.exec(select(StudySession).where(StudySession.plan_id == plan.id)).all()
    past = [r for r in rows if r.date < today.isoformat()]
    done = [r for r in past if r.completed]
    skipped = [r for r in past if r.skipped]
    done_topics: List[str] = []
    for r in done:
        if r.topic and r.topic not in done_topics:
            done_topics.append(r.topic)
    rm = load_roadmap(plan)
    current = None
    if rm:
        ov = ps.overview(roadmap=rm, start=plan_start(plan), deadline=plan.deadline,
                         days=plan_days(plan), today=today)
        current = ov.get("current_unit")
    parts = [
        f"{len(done)} of {len(past)} past sessions completed, {len(skipped)} skipped",
        f"topics practised: {', '.join(done_topics[-12:]) or 'none yet'}",
    ]
    if current:
        parts.append(f"the schedule had reached '{current}'")
    return "; ".join(parts)


# ============================================================
# RESPONSE BUILDER
# ============================================================

def _session_detail(s: StudySession) -> SessionDetail:
    return SessionDetail(
        id=str(s.id),
        date=s.date,
        time=s.time or "18:00",
        subject=s.subject,
        topic=s.topic,
        task=getattr(s, "task", None),
        subject_id=s.subject_id,
        duration_mins=s.duration_mins,
        mode=normalize_mode(s.mode),
        priority=normalize_priority(s.priority),
        completed=s.completed,
        skipped=bool(s.skipped),
        is_manual=(getattr(s, "source", "plan") == "manual"),
    )


def build_plan_response(
    db: Session,
    plan: StudyPlan,
    today: date,
    *,
    user_id: int,
    db_subjects: Optional[list] = None,
) -> PlanResponse:
    db_sessions = db.exec(select(StudySession).where(StudySession.plan_id == plan.id)).all()
    today_str = today.isoformat()
    upcoming = sorted(
        (s for s in db_sessions if s.date >= today_str),
        key=lambda s: (s.date, s.time or "00:00"),
    )

    rm = load_roadmap(plan)
    days = plan_days(plan)
    ov = None
    if rm:
        ov = ps.overview(roadmap=rm, start=plan_start(plan), deadline=plan.deadline,
                         days=days, today=today)

    days_remaining = max((plan.deadline - today).days, 0) if plan.deadline else -1
    topics_count = (
        sum(len(p["units"]) for p in rm["phases"]) if rm
        else len({s.topic or s.subject for s in upcoming})
    )
    stats = PlanStats(
        days_remaining=days_remaining,
        daily_target_mins=plan.minutes_per_day or 45,
        topics_count=topics_count,
    )

    week = []
    for i in range(7):
        d = today + timedelta(days=i)
        ds = d.isoformat()
        day_sessions = [s for s in upcoming if s.date == ds and not s.skipped]
        if not day_sessions:
            st = "rest"
        elif any(normalize_priority(s.priority) == "weak_area" for s in day_sessions):
            st = "weak_area"
        elif any(normalize_mode(s.mode) in ("review", "mock") for s in day_sessions):
            st = "review"
        else:
            st = "study"
        week.append(WeekDay(date=ds, day_label=d.strftime("%a").upper(),
                            has_session=bool(day_sessions), session_type=st))

    roadmap_phases = []
    phase_name = phase_milestone = None
    if ov:
        for p in ov["phases"]:
            roadmap_phases.append(RoadmapPhase(
                name=p["name"], milestone=p["milestone"], weeks=p["weeks"],
                start_date=date.fromisoformat(p["start_date"]) if p["start_date"] else None,
                end_date=date.fromisoformat(p["end_date"]) if p["end_date"] else None,
                status=p["status"], units=p["units"],
            ))
            if p["status"] == "current":
                phase_name, phase_milestone = p["name"], p["milestone"]

    subjects_resp = []
    for sub in (db_subjects or []):
        subj_sessions = [s for s in db_sessions if s.subject_id == sub.id]
        subjects_resp.append({
            "id": sub.id,
            "name": sub.name,
            "exam_date": sub.exam_date,
            "weight": sub.weight,
            "current_level": sub.current_level,
            "topics": _loads(sub.topics, []),
            "weak_topics": _loads(sub.weak_topics, []),
            "session_count": len(subj_sessions),
            "completed_count": sum(1 for s in subj_sessions if s.completed),
        })

    return PlanResponse(
        plan_id=plan.id,
        title=plan.title or plan.subject,
        goal_type=plan.goal_type or "self_improvement",
        emoji=plan.emoji,
        color_hex=plan.color_hex,
        is_approved=plan.is_approved,
        goal=PlanGoal(subject=plan.subject, deadline=plan.deadline),
        stats=stats,
        week=week,
        sessions=[_session_detail(s) for s in upcoming],
        nudge=None,
        subjects=subjects_resp,
        plan_type=plan.plan_type or "single_goal",
        notes=plan.notes,
        total_weeks=ov["total_weeks"] if ov else None,
        current_week=ov["current_week"] if ov else None,
        phase_name=phase_name,
        phase_milestone=phase_milestone,
        updated_at=plan.updated_at,
        goal_kind=(rm or {}).get("goal_kind") or plan.goal_kind,
        level=plan.level,
        summary=(rm or {}).get("summary") or None,
        schedule=PlanSchedule(
            minutes_per_day=plan.minutes_per_day or 45,
            days_of_week=days,
            preferred_times=plan_times(plan),
        ),
        roadmap=roadmap_phases,
        current_unit=ov["current_unit"] if ov else None,
        end_date=date.fromisoformat(ov["end_date"]) if ov else plan.deadline,
        roadmap_source=plan.roadmap_source,
        ai_plans_left=ai_plans_left(db, user_id, today),
    )