import asyncio
import json
from datetime import date, timedelta
from typing import Optional

from sqlmodel import Session, select

from database import engine
from models import StudyPlan, StudySession
from ai_service import generate_gemini_study_plan


ROLLOVER_LOOKAHEAD_DAYS = 3
RECENT_TOPICS_LOOKBACK_DAYS = 14
ROLLOVER_INTERVAL_SECONDS = 3600


def _phase_for_week(
    plan_phase_plan_raw: Optional[str],
    week: int,
) -> Optional[dict]:
    """Return the phase dict that contains `week`, or None."""
    if not plan_phase_plan_raw:
        return None
    try:
        data = json.loads(plan_phase_plan_raw)
    except (ValueError, TypeError):
        return None
    for p in data.get("phases", []):
        if p.get("start_week", 0) <= week <= p.get("end_week", 0):
            return p
    return None


async def extend_plan_window(plan_id: int) -> bool:
    """
    Generate the next window for a plan if it's due.

    Idempotent — if sessions already exist for the target window,
    returns False without calling the AI.

    Opens the DB session briefly, drops it during the AI call, then
    opens a new one to persist. This keeps the connection pool free
    during the 10–20s the model is thinking.
    """
    # ------------------------------------------------------------
    # STEP 1 — Read state
    # ------------------------------------------------------------
    with Session(engine) as db:
        plan = db.get(StudyPlan, plan_id)
        if not plan or plan.is_archived or not plan.is_approved:
            return False

        if not plan.window_start:
            plan.window_start = plan.start_date or date.today()
            db.add(plan)
            db.commit()

        # ------------------------------------------------------------
        # CATCH-UP: if the next window would start in the past,
        # skip forward in whole-window steps until it lands on or
        # after today. This handles the case where the app was
        # asleep or down for days and the natural rollover fire
        # time has passed.
        # ------------------------------------------------------------
        next_start = plan.window_start + timedelta(days=plan.window_days)
        today_utc = date.today()

        while next_start < today_utc:
            next_start = next_start + timedelta(days=plan.window_days)

        next_end = next_start + timedelta(days=plan.window_days - 1)

        if plan.deadline and next_start > plan.deadline:
            return False

        existing = db.exec(
            select(StudySession).where(
                StudySession.plan_id == plan.id,
                StudySession.date >= next_start.isoformat(),
                StudySession.date <= next_end.isoformat(),
            )
        ).first()

        if existing:
            # Window already generated — advance the pointer and bail.
            plan.window_start = next_start
            db.add(plan)
            db.commit()
            return False

        # Snapshot values we need after the session closes.
        snapshot = {
            "subject": plan.subject,
            "goal_type": plan.goal_type,
            "deadline": plan.deadline,
            "start_date": plan.start_date,
            "window_days": plan.window_days,
            "phase_plan_raw": plan.phase_plan,
        }

        # Recent context — topics and skips from the last 14 days.
        cutoff = (
            date.today() - timedelta(days=RECENT_TOPICS_LOOKBACK_DAYS)
        ).isoformat()

        recent_rows = db.exec(
            select(StudySession).where(
                StudySession.plan_id == plan.id,
                StudySession.date >= cutoff,
            )
        ).all()

        recent_topics: list[str] = []
        recent_skips: list[str] = []
        seen: set[str] = set()
        for s in recent_rows:
            if s.subject and s.subject not in seen:
                seen.add(s.subject)
                recent_topics.append(s.subject)
            if s.skipped and s.subject not in recent_skips:
                recent_skips.append(s.subject)

    # ------------------------------------------------------------
    # STEP 2 — Compute phase context (no DB)
    # ------------------------------------------------------------
    phase = None
    if snapshot["start_date"]:
        week_for_next = (
            (next_start - snapshot["start_date"]).days // 7
        ) + 1
        phase = _phase_for_week(snapshot["phase_plan_raw"], week_for_next)

    days_remaining = None
    if snapshot["deadline"]:
        days_remaining = max((snapshot["deadline"] - next_start).days, 0)

 
    # ------------------------------------------------------------
    # STEP 3 — AI call (no DB lock held)
    # ------------------------------------------------------------
    # Change the function name here:
    window_data = await generate_gemini_study_plan(
        goal=snapshot["subject"],
        target_date=snapshot["deadline"],
        days_remaining=days_remaining,
        goal_type=snapshot["goal_type"],
        window_start=next_start,
        window_days=snapshot["window_days"],
        phase_context=phase,
        recent_topics=recent_topics,
        recent_skips=recent_skips,
    )

    if not window_data or not window_data.get("sessions"):
        return False

    # ------------------------------------------------------------
    # STEP 4 — Persist (short session, with a race re-check)
    # ------------------------------------------------------------
    with Session(engine) as db:
        plan = db.get(StudyPlan, plan_id)
        if not plan:
            return False

        # Re-check: a parallel run might have beaten us to it.
        existing = db.exec(
            select(StudySession).where(
                StudySession.plan_id == plan.id,
                StudySession.date >= next_start.isoformat(),
                StudySession.date <= next_end.isoformat(),
            )
        ).first()

        if existing:
            plan.window_start = next_start
            db.add(plan)
            db.commit()
            return False

        for s in window_data["sessions"]:
            db.add(
                StudySession(
                    plan_id=plan.id,
                    user_id=plan.user_id,
                    date=s["date"],
                    time=s["time"],
                    subject=s["subject"],
                    duration_mins=s["duration_mins"],
                    mode=s["mode"],
                    priority=s["priority"],
                    completed=False,
                )
            )

        plan.window_start = next_start
        db.add(plan)
        db.commit()

    return True


async def _rollover_tick() -> None:
    """
    One pass over all eligible plans. Extracted so the loop can
    call it both immediately on startup and then on every sleep.
    """
    with Session(engine) as db:
        candidates = db.exec(
            select(StudyPlan).where(
                StudyPlan.is_archived == False,  # noqa: E712
                StudyPlan.is_approved == True,   # noqa: E712
                StudyPlan.window_start != None,  # noqa: E711
            )
        ).all()

        today = date.today()
        due_ids: list[int] = []
        for plan in candidates:
            days_left = (
                plan.window_start
                + timedelta(days=plan.window_days)
                - today
            ).days
            # Fire whenever the current window is about to end OR
            # has already ended. Negative values mean the rollover
            # was missed (app was asleep / down) — we want to catch
            # up in that case, not skip.
            if days_left <= ROLLOVER_LOOKAHEAD_DAYS:
                due_ids.append(plan.id)

    for pid in due_ids:
        try:
            await extend_plan_window(pid)
        except Exception as e:
            print(f"Rollover failed for plan {pid}: {e}")


async def plan_rollover_loop():
    """
    Runs once at startup, then fires hourly. Extends any plan
    whose current window is about to expire (or already has).
    """
    # Immediate pass so a fresh deploy doesn't wait an hour.
    try:
        await _rollover_tick()
    except Exception as e:
        print(f"Startup rollover failed: {e}")

    while True:
        try:
            await asyncio.sleep(ROLLOVER_INTERVAL_SECONDS)
            await _rollover_tick()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"Rollover loop error: {e}")