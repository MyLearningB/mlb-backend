"""
plan_rollover.py — keeps every active plan filled ~14 days ahead.

v2 changes
----------
* Topping up is pure code (plan_scheduler) — no AI call per window.
  The old version called Gemini for every plan every week.
* The same top-up also runs lazily when the app opens the dashboard,
  plan or upcoming list (plan_service.ensure_filled), so plans keep
  going even when Render's free instance was asleep.
* Multi-year plans: when a learner reaches a long phase that is still
  coarse, ONE AI call breaks that phase into detailed units
  (refine_plan_phase). At most once per phase.
"""

import asyncio
import json
from datetime import date

from sqlmodel import Session, select

import plan_scheduler as ps
import plan_service as svc
from ai_service import refine_plan_phase
from database import engine
from models import StudyPlan


ROLLOVER_INTERVAL_SECONDS = 3600
MAX_REFINES_PER_TICK = 5


async def refine_if_needed(plan_id: int, today: date) -> bool:
    """Detail the current phase of a long plan. Returns True if changed."""
    with Session(engine) as db:
        plan = db.get(StudyPlan, plan_id)
        if not plan or plan.is_archived:
            return False
        rm = svc.load_roadmap(plan)
        if not rm:
            return False
        days = svc.plan_days(plan)
        start = svc.plan_start(plan)
        idx = ps.phase_needing_detail(
            roadmap=rm, start=start, deadline=plan.deadline, days=days, today=today,
        )
        if idx is None:
            return False
        snapshot = {
            "goal": plan.subject,
            "kind": rm.get("goal_kind", "skill"),
            "level": plan.level or "beginner",
            "minutes": plan.minutes_per_day or 45,
            "phase": rm["phases"][idx],
            "done": [u["title"] for p in rm["phases"][:idx] for u in p["units"]],
            "version": plan.arc_version,
        }

    # AI call with no DB connection held.
    units = await refine_plan_phase(
        snapshot["goal"],
        kind=snapshot["kind"],
        level=snapshot["level"],
        phase=snapshot["phase"],
        phase_weeks=snapshot["phase"]["weeks"],
        minutes_per_day=snapshot["minutes"],
        done_units=snapshot["done"],
    )

    with Session(engine) as db:
        plan = db.get(StudyPlan, plan_id)
        if not plan or plan.arc_version != snapshot["version"]:
            return False  # re-planned meanwhile
        rm = svc.load_roadmap(plan)
        if not rm or idx >= len(rm["phases"]):
            return False
        phase = rm["phases"][idx]
        if units:
            candidate = dict(rm)
            candidate["phases"] = list(rm["phases"])
            candidate["phases"][idx] = {**phase, "units": units, "detailed": True}
            fixed = ps.normalize_roadmap(
                candidate,
                total_weeks=rm.get("total_weeks") or sum(p["weeks"] for p in rm["phases"]),
                goal=plan.subject,
                kind_hint=rm.get("goal_kind", "skill"),
            )
            if fixed:
                fixed["phases"][idx]["detailed"] = True
                rm = fixed
        # Mark as detailed even on failure so we never retry in a loop.
        rm["phases"][idx]["detailed"] = True
        plan.roadmap = json.dumps(rm)
        db.add(plan)

        # Sessions not yet started should use the new detail.
        if units:
            svc.clear_future_plan_sessions(db, plan, today)
            svc.ensure_filled(db, plan, today)
        db.commit()
    return bool(units)


async def _rollover_tick() -> None:
    today = date.today()

    with Session(engine) as db:
        plans = db.exec(
            select(StudyPlan).where(StudyPlan.is_archived == False)  # noqa: E712
        ).all()
        plan_ids = [p.id for p in plans]
        changed = False
        for p in plans:
            try:
                changed = svc.ensure_filled(db, p, today) or changed
            except Exception as e:
                print(f"Rollover fill failed for plan {p.id}: {e}")
                db.rollback()
        if changed:
            db.commit()

    refined = 0
    for pid in plan_ids:
        if refined >= MAX_REFINES_PER_TICK:
            break
        try:
            if await refine_if_needed(pid, today):
                refined += 1
        except Exception as e:
            print(f"Phase refine failed for plan {pid}: {e}")


async def plan_rollover_loop():
    """Runs once at startup, then hourly."""
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