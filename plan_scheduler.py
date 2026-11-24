"""
plan_scheduler.py — turns ONE AI roadmap into daily study sessions.

No database, no network. Everything here is a pure function so it is
cheap, testable and identical every time it runs. This is what lets a
3-year plan cost one (or a handful of) AI calls instead of one call per
week: the AI writes the curriculum once, and this file spreads it over
the calendar at the times the user picked.

How a day is built
------------------
1. All study dates between the roadmap start and the end (deadline, or
   start + total_weeks when there is no deadline) are listed, using the
   user's chosen weekdays.
2. Each phase gets a share of those dates proportional to its `weeks`,
   and each unit inside a phase gets an even share of the phase.
3. For one date we know: phase, unit, how far into the unit we are.
   - Slot 1 (the user's first time) = new work on the current unit,
     cycling learn -> practice -> flashcards/feynman (exams) or
     learn -> practice -> project (skills).
   - Extra slots = spaced review of earlier units (1, 2, 4, 8, 16 units
     back), or a skipped topic marked weak_area.
   - Last study day of a phase = milestone session (mock test / project).
   - Last study day of a week = weekly review (when studying 4+ days).
   - Exam eve = light review, no new material.
"""

from __future__ import annotations

import math
import re
from datetime import date, timedelta
from typing import Iterable, List, Optional

# ============================================================
# CANONICAL VALUES
# ============================================================

GOAL_KINDS = ("exam", "course", "skill", "habit")
LEVELS = ("beginner", "intermediate", "advanced")
MODES = ("learn", "practice", "review", "flashcard", "feynman", "mock", "project")

MAX_PHASES = 6
MAX_UNITS_TOTAL = 150
MAX_TASKS_PER_UNIT = 5
MAX_HORIZON_DAYS = 365 * 5 + 2  # 5 years

DEFAULT_TIMES = {
    "high_school": "16:00",
    "university": "18:00",
    "professional": "20:00",
    "self_improvement": "19:00",
}

_EXAM_WORDS = (
    "exam", "test", "sat", "act", "gre", "gmat", "lsat", "mcat", "ielts",
    "toefl", "waec", "jamb", "neco", "utme", "post-utme", "post utme",
    "gce", "igcse", "a-level", "a level", "o-level", "ap ", "bar ",
    "boards", "board exam", "usmle", "plab", "nclex", "acca", "ican",
    "cfa", "cpa", "cima", "pmp", "certification", "certificate",
    "final", "finals", "midterm", "mid-term", "quiz", "paper",
    "license", "licence", "entrance", "olympiad", "cbt", "jupeb",
)
_COURSE_WORDS = (
    "course", "class", "module", "semester", "lecture", "syllabus",
    "unit ", "chapter", "textbook", "curriculum", "degree",
)
_HABIT_WORDS = (
    "habit", "daily", "every day", "routine", "read more", "journal",
    "meditat", "consisten", "books a", "pages a",
)


# ============================================================
# SMALL PARSERS
# ============================================================

_TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


def parse_times(raw: Optional[Iterable]) -> List[str]:
    """Clean a list of 'HH:MM' strings: valid, unique, sorted, max 3."""
    out = set()
    for t in raw or []:
        m = _TIME_RE.match(str(t).strip())
        if m:
            out.add(f"{int(m.group(1)):02d}:{m.group(2)}")
    return sorted(out)[:3]


def parse_days(raw: Optional[Iterable]) -> List[int]:
    """Clean a list of ISO weekdays (Mon=1 .. Sun=7)."""
    out = set()
    for d in raw or []:
        try:
            v = int(d)
        except (TypeError, ValueError):
            continue
        if 1 <= v <= 7:
            out.add(v)
    return sorted(out)


def guess_goal_kind(goal: str) -> str:
    """Cheap keyword guess. The AI makes the final call."""
    g = f" {goal.lower()} "
    if any(w in g for w in _EXAM_WORDS):
        return "exam"
    if any(w in g for w in _COURSE_WORDS) or re.search(r"[a-z]{2,4}\s?\d{3}", g):
        return "course"
    if any(w in g for w in _HABIT_WORDS):
        return "habit"
    return "skill"


def default_horizon_weeks(kind: str, level: str) -> int:
    """How long a plan with no deadline should run."""
    if kind == "habit":
        return 8
    if kind == "skill":
        return {"beginner": 12, "intermediate": 10, "advanced": 8}.get(level, 12)
    return 12


def horizon_weeks(start: date, deadline: Optional[date], kind: str, level: str) -> int:
    if deadline:
        days = (deadline - start).days + 1
        return max(1, math.ceil(days / 7))
    return default_horizon_weeks(kind, level)


def roadmap_end(start: date, deadline: Optional[date], roadmap: dict) -> date:
    if deadline:
        return deadline
    weeks = int(roadmap.get("total_weeks") or 0) or sum(
        int(p.get("weeks") or 1) for p in roadmap.get("phases", [])
    )
    return start + timedelta(days=max(1, weeks) * 7 - 1)


def split_minutes(total: int, slots: int) -> List[int]:
    """Split a daily budget into `slots` sessions, rounded to 5 mins."""
    slots = max(1, slots)
    base = max(10, int(round(total / slots / 5.0)) * 5)
    return [base] * slots


def slots_for(minutes_per_day: int, times: List[str]) -> int:
    """One session per chosen time, but never shorter than ~15 mins."""
    return max(1, min(len(times) or 1, minutes_per_day // 15 or 1))


def _clip(value, limit: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


# ============================================================
# ROADMAP NORMALISATION
# ============================================================

def normalize_roadmap(
    data: Optional[dict],
    *,
    total_weeks: int,
    goal: str,
    kind_hint: str,
) -> Optional[dict]:
    """
    Validate whatever the AI returned and coerce it into the shape the
    scheduler expects. Returns None when nothing usable came back.
    """
    if not isinstance(data, dict):
        return None
    raw_phases = data.get("phases")
    if not isinstance(raw_phases, list):
        return None

    phases = []
    units_left = MAX_UNITS_TOTAL
    for raw in raw_phases[:MAX_PHASES]:
        if not isinstance(raw, dict):
            continue
        units = []
        for u in raw.get("units") or []:
            if units_left <= 0:
                break
            if isinstance(u, str):
                u = {"title": u, "tasks": []}
            if not isinstance(u, dict):
                continue
            title = _clip(u.get("title"), 80)
            if not title:
                continue
            tasks = [
                _clip(t, 120)
                for t in (u.get("tasks") or [])
                if isinstance(t, (str, int, float)) and _clip(t, 120)
            ][:MAX_TASKS_PER_UNIT]
            if not tasks:
                tasks = [f"Study {title}", f"Practise {title}"]
            units.append({"title": title, "tasks": tasks})
            units_left -= 1

        name = _clip(raw.get("name"), 60) or f"Phase {len(phases) + 1}"
        if not units:
            # Older shape: {"focus": ["..."]}
            focus = [f for f in (raw.get("focus") or []) if _clip(f, 80)]
            for f in focus[:6]:
                if units_left <= 0:
                    break
                units.append({"title": _clip(f, 80), "tasks": [f"Study {_clip(f, 80)}"]})
                units_left -= 1
        if not units:
            continue
        try:
            weeks = int(raw.get("weeks") or 0)
        except (TypeError, ValueError):
            weeks = 0
        if weeks <= 0:
            try:
                weeks = int(raw.get("end_week") or 0) - int(raw.get("start_week") or 0) + 1
            except (TypeError, ValueError):
                weeks = 0
        phases.append(
            {
                "name": name,
                "weeks": max(1, weeks),
                "milestone": _clip(raw.get("milestone"), 160),
                "units": units,
                "detailed": bool(raw.get("detailed", False)),
            }
        )

    if not phases:
        return None

    _rescale_weeks(phases, total_weeks)

    kind = str(data.get("goal_kind") or "").strip().lower()
    if kind not in GOAL_KINDS:
        kind = kind_hint if kind_hint in GOAL_KINDS else "skill"

    return {
        "version": 2,
        "goal_kind": kind,
        "title": _clip(data.get("title"), 60) or _clip(goal, 60),
        "emoji": _clip(data.get("emoji"), 8) or None,
        "summary": _clip(data.get("summary"), 240),
        "total_weeks": total_weeks,
        "phases": phases,
    }


def _rescale_weeks(phases: List[dict], total_weeks: int) -> None:
    """Make phase weeks add up to total_weeks (each phase >= 1 when possible)."""
    raw_total = sum(p["weeks"] for p in phases) or 1
    if total_weeks < len(phases):
        for p in phases:
            p["weeks"] = 1
        return
    scaled = [max(1, round(p["weeks"] * total_weeks / raw_total)) for p in phases]
    diff = total_weeks - sum(scaled)
    i = len(scaled) - 1
    while diff != 0 and i >= -len(scaled) * 4:
        j = i % len(scaled)
        if diff > 0:
            scaled[j] += 1
            diff -= 1
        elif scaled[j] > 1:
            scaled[j] -= 1
            diff += 1
        i -= 1
    for p, w in zip(phases, scaled):
        p["weeks"] = w


# ============================================================
# FALLBACK ROADMAP (no AI available)
# ============================================================

def fallback_roadmap(goal: str, kind: str, level: str, total_weeks: int) -> dict:
    """A sensible generic curriculum, used when the AI is down or the
    user is out of AI plans for the day. Users can re-plan later."""
    g = _clip(goal, 60) or "your goal"
    if kind in ("exam", "course"):
        phases = [
            {
                "name": "Map the syllabus",
                "weeks": 1,
                "milestone": f"Know every topic {g} covers and your weak spots",
                "units": [
                    {"title": "Syllabus & baseline", "tasks": [
                        f"Get the official syllabus or outline for {g}",
                        "List every topic and rate yourself 1–5 on each",
                        "Do a short past paper or quiz untimed to find gaps",
                    ]},
                ],
            },
            {
                "name": "Build the foundations",
                "weeks": 4,
                "milestone": "Cover every core topic once",
                "units": [
                    {"title": f"Core topics: part {i}", "tasks": [
                        "Read and summarise the next weakest topic on your list",
                        "Work through 10 worked examples or questions",
                        "Write a one-page cheat sheet for the topic",
                    ]} for i in (1, 2, 3, 4)
                ],
            },
            {
                "name": "Practice & weak areas",
                "weeks": 4,
                "milestone": "Score 70%+ on a timed section",
                "units": [
                    {"title": "Past questions", "tasks": [
                        "Do 20 past questions on one topic, then mark them",
                        "Redo every question you got wrong",
                        "Timed section: answer under exam conditions",
                    ]},
                    {"title": "Weak-area repair", "tasks": [
                        "Pick your lowest-rated topic and relearn it",
                        "Explain the hardest idea out loud from memory",
                        "Make 15 flashcards for formulas and definitions",
                    ]},
                ],
            },
            {
                "name": "Final review",
                "weeks": 2,
                "milestone": "Full mock exam under timed conditions",
                "units": [
                    {"title": "Mock exams & review", "tasks": [
                        "Sit a full timed mock exam",
                        "Review every mistake from the last mock",
                        "Skim all cheat sheets and flashcards",
                    ]},
                ],
            },
        ]
    elif kind == "habit":
        phases = [
            {"name": "Start small", "weeks": 2, "milestone": "7 days in a row",
             "units": [{"title": f"Tiny daily {g}", "tasks": [
                 f"Do 10 focused minutes of {g}",
                 "Note one thing you noticed or learned today",
             ]}]},
            {"name": "Build the streak", "weeks": 4, "milestone": "21 days in a row",
             "units": [{"title": f"Steady {g}", "tasks": [
                 f"Do a full session of {g}",
                 "Raise the bar slightly from last week",
             ]}]},
            {"name": "Make it yours", "weeks": 2, "milestone": "Habit feels automatic",
             "units": [{"title": f"Level up {g}", "tasks": [
                 f"Try a harder variation of {g}",
                 "Review your progress since day one",
             ]}]},
        ]
    else:
        start_tip = "from zero" if level == "beginner" else "at your level"
        phases = [
            {"name": "Basics", "weeks": 3, "milestone": f"Finish a tiny first {g} project",
             "units": [
                 {"title": "Tools & first steps", "tasks": [
                     f"Find one good beginner guide for {g} {start_tip}",
                     "Set up the tools or materials you need",
                     "Follow the guide's first lesson start to finish",
                 ]},
                 {"title": "Core techniques", "tasks": [
                     "Learn one core technique from your guide",
                     "Practise that technique for the whole session",
                     "Record or photograph your attempt to compare later",
                 ]},
             ]},
            {"name": "Build skill", "weeks": 5, "milestone": "Complete a medium project on your own",
             "units": [
                 {"title": f"Intermediate {g}", "tasks": [
                     "Learn the next technique in your guide",
                     "Practise it until it feels comfortable",
                     "Combine two techniques in one small piece",
                 ]},
                 {"title": "Mini project", "tasks": [
                     "Plan a small project using what you know",
                     "Work on the project",
                     "Fix the weakest part of your project",
                 ]},
             ]},
            {"name": "Projects", "weeks": 4, "milestone": f"Show off a finished {g} project",
             "units": [
                 {"title": "Signature project", "tasks": [
                     "Pick a project slightly above your level",
                     "Work on your project",
                     "Get feedback and improve one thing",
                 ]},
             ]},
        ]
    rm = normalize_roadmap(
        {"goal_kind": kind, "title": g, "phases": phases},
        total_weeks=total_weeks,
        goal=goal,
        kind_hint=kind,
    )
    return rm  # type: ignore[return-value]


# ============================================================
# CALENDAR ALLOCATION
# ============================================================

def study_dates(start: date, end: date, days: List[int]) -> List[date]:
    days = days or [1, 2, 3, 4, 5, 6]
    out = []
    d = start
    while d <= end:
        if d.isoweekday() in days:
            out.append(d)
        d += timedelta(days=1)
    if not out:
        # e.g. a 2-day plan that falls on rest days: use every day.
        d = start
        while d <= end:
            out.append(d)
            d += timedelta(days=1)
    return out


def _largest_remainder(weights: List[float], total: int) -> List[int]:
    s = sum(weights) or 1.0
    exact = [w * total / s for w in weights]
    base = [int(math.floor(x)) for x in exact]
    rem = total - sum(base)
    order = sorted(range(len(weights)), key=lambda i: exact[i] - base[i], reverse=True)
    for i in order[:rem]:
        base[i] += 1
    return base


def allocate(roadmap: dict, n_dates: int) -> List[dict]:
    """
    Map each study-date index -> where it sits in the roadmap.
    Returns a list of n_dates dicts:
      {phase, unit (global index), unit_in_phase, day_in_unit, unit_len,
       phase_last (bool)}
    """
    phases = roadmap.get("phases") or []
    if not phases or n_dates <= 0:
        return []

    per_phase = _largest_remainder([max(1, p["weeks"]) for p in phases], n_dates)
    out: List[dict] = []
    unit_offset = 0
    for pi, (phase, count) in enumerate(zip(phases, per_phase)):
        units = phase["units"]
        U = len(units)
        if count <= 0:
            unit_offset += U
            continue
        if count >= U:
            per_unit = _largest_remainder([1.0] * U, count)
            for ui, n in enumerate(per_unit):
                for j in range(n):
                    out.append({
                        "phase": pi,
                        "unit": unit_offset + ui,
                        "unit_in_phase": ui,
                        "day_in_unit": j,
                        "unit_len": n,
                        "phase_last": False,
                        "phase_len": count,
                    })
        else:
            # Fewer days than units: step through units, skipping some.
            for j in range(count):
                ui = min(U - 1, (j * U) // count)
                out.append({
                    "phase": pi,
                    "unit": unit_offset + ui,
                    "unit_in_phase": ui,
                    "day_in_unit": 0,
                    "unit_len": 1,
                    "phase_last": False,
                    "phase_len": count,
                })
        out[-1]["phase_last"] = True
        unit_offset += U
    return out[:n_dates]


def flat_units(roadmap: dict) -> List[dict]:
    units = []
    for pi, p in enumerate(roadmap.get("phases") or []):
        for u in p["units"]:
            units.append({**u, "phase": pi})
    return units


# ============================================================
# SESSION BUILDER
# ============================================================

_EXAM_CYCLE = ("practice", "learn", "flashcard", "practice", "feynman")
_SKILL_CYCLE = ("practice", "practice", "learn", "project")


def _primary_mode(kind: str, day_in_unit: int, level: str) -> str:
    if kind == "habit":
        return "practice"
    if day_in_unit == 0:
        return "learn"
    cycle = _EXAM_CYCLE if kind in ("exam", "course") else _SKILL_CYCLE
    if level == "advanced" and kind in ("exam", "course"):
        cycle = ("practice", "feynman", "practice", "flashcard", "learn")
    return cycle[(day_in_unit - 1) % len(cycle)]


def _task_index(kind: str, day_in_unit: int, level: str) -> int:
    """How many task-consuming days (learn/practice/project) came before."""
    n = 0
    for j in range(day_in_unit):
        if _primary_mode(kind, j, level) in ("learn", "practice", "project"):
            n += 1
    return n


def _unit_task_text(mode: str, unit: dict, idx: int) -> str:
    title = unit["title"]
    tasks = unit["tasks"]
    if mode == "flashcard":
        return f"Flashcards: drill the key terms and formulas from {title}"
    if mode == "feynman":
        return f"Explain {title} out loud in simple words, then fill the gaps"
    return tasks[idx % len(tasks)]


def build_sessions(
    *,
    roadmap: dict,
    start: date,
    deadline: Optional[date],
    days: List[int],
    times: List[str],
    minutes_per_day: int,
    level: str,
    subject: str,
    from_date: date,
    to_date: date,
    weak_topics: Optional[List[str]] = None,
) -> List[dict]:
    """Sessions for every study date in [from_date, to_date]."""
    kind = roadmap.get("goal_kind", "skill")
    end = roadmap_end(start, deadline, roadmap)
    times = times or ["18:00"]
    days = days or [1, 2, 3, 4, 5, 6]
    units = flat_units(roadmap)
    phases = roadmap.get("phases") or []
    if not units:
        return []

    all_dates = study_dates(start, end, days)
    alloc = allocate(roadmap, len(all_dates))
    index_of = {d: i for i, d in enumerate(all_dates)}

    n_slots = slots_for(minutes_per_day, times)
    durations = split_minutes(minutes_per_day, n_slots)
    slot_times = times[:n_slots]
    weak = [w for w in (weak_topics or []) if w][:3]

    sessions: List[dict] = []
    d = max(from_date, start)
    while d <= to_date:
        if deadline and d > deadline:
            break
        if d.isoweekday() not in days and d not in index_of:
            d += timedelta(days=1)
            continue

        if d in index_of:
            k = index_of[d]
            a = alloc[k] if k < len(alloc) else alloc[-1]
            day = _build_day(
                d=d, k=k, a=a, all_dates=all_dates, alloc=alloc,
                units=units, phases=phases, kind=kind, level=level,
                deadline=deadline, weak=weak,
            )
        elif d > end:
            # No deadline and the roadmap is finished: keep a light
            # maintenance routine (review + practice) going.
            k = (d - end).days
            day = _maintenance_day(k, units, kind)
        else:
            d += timedelta(days=1)
            continue

        for slot, (t, dur) in enumerate(zip(slot_times, durations)):
            if slot >= len(day):
                break
            s = day[slot]
            sessions.append({
                "date": d.isoformat(),
                "time": t,
                "subject": _clip(subject, 120),
                "topic": _clip(s["topic"], 200),
                "task": _clip(s["task"], 300),
                "duration_mins": dur,
                "mode": s["mode"],
                "priority": s.get("priority") or _priority(kind, deadline, d),
            })
        d += timedelta(days=1)
    return sessions


def _priority(kind: str, deadline: Optional[date], d: date) -> str:
    if deadline and kind in ("exam", "course") and (deadline - d).days <= 14:
        return "high"
    return "normal"


def _build_day(*, d, k, a, all_dates, alloc, units, phases, kind, level,
               deadline, weak) -> List[dict]:
    """Up to 3 session specs for one study date, most important first."""
    unit = units[a["unit"]]
    phase = phases[a["phase"]]
    day: List[dict] = []
    is_last_date = k == len(all_dates) - 1
    exam_like = kind in ("exam", "course")

    # ---- primary slot ------------------------------------------
    if exam_like and deadline and is_last_date and len(all_dates) >= 3:
        recent = _recent_unit_titles(alloc, units, k, 3)
        day.append({
            "topic": "Exam eve",
            "task": "Light review only: skim your notes on " + ", ".join(recent)
                    + ". Sleep early.",
            "mode": "review",
            "priority": "high",
        })
    elif a["phase_last"] and a.get("phase_len", 0) >= 5:
        milestone = phase.get("milestone") or f"Finish {phase['name']}"
        day.append({
            "topic": f"Milestone · {phase['name']}",
            "task": milestone,
            "mode": "mock" if exam_like else "project",
            "priority": "high",
        })
    elif _is_weekly_review_day(d, k, all_dates) and a["day_in_unit"] > 0:
        recent = _recent_unit_titles(alloc, units, k, 3)
        day.append({
            "topic": "Weekly review",
            "task": "Review this week: " + ", ".join(recent),
            "mode": "review",
        })
    else:
        mode = _primary_mode(kind, a["day_in_unit"], level)
        idx = _task_index(kind, a["day_in_unit"], level)
        day.append({
            "topic": unit["title"],
            "task": _unit_task_text(mode, unit, idx),
            "mode": mode,
        })

    if day[0]["topic"] == "Exam eve":
        return day  # no extra load the night before

    # ---- extra slots: spaced review / weak topics ----------------
    prev = [a["unit"] - g for g in (1, 2, 4, 8, 16) if a["unit"] - g >= 0]
    for slot in (1, 2):
        if slot == 1 and weak:
            day.append({
                "topic": weak[k % len(weak)],
                "task": f"Catch up on {weak[k % len(weak)]} (you skipped it)",
                "mode": "review" if exam_like else "practice",
                "priority": "weak_area",
            })
            continue
        if prev:
            u = units[prev[(k + slot) % len(prev)]]
            if exam_like:
                mode = ("flashcard", "feynman", "review")[(k + slot) % 3]
                task = _unit_task_text(mode, u, k)
                if mode == "review":
                    task = f"Quick quiz yourself on {u['title']}"
            else:
                mode = "practice"
                task = f"Warm-up: revisit {u['title']} — " + u["tasks"][k % len(u["tasks"])]
            day.append({"topic": u["title"], "task": task, "mode": mode})
        else:
            idx = _task_index(kind, a["day_in_unit"], level)
            if exam_like:
                mode = ("flashcard", "feynman")[(k + slot) % 2]
                task = _unit_task_text(mode, unit, idx)
            else:
                mode = "practice"
                tasks = unit["tasks"]
                if idx == 0:
                    task = "Preview: " + tasks[1 % len(tasks)]
                else:
                    task = "Practise again: " + tasks[(idx - 1) % len(tasks)]
            day.append({"topic": unit["title"], "task": task, "mode": mode})
    return day


def _maintenance_day(k: int, units: List[dict], kind: str) -> List[dict]:
    u = units[k % len(units)]
    v = units[(k * 3 + 1) % len(units)]
    exam_like = kind in ("exam", "course")
    return [
        {"topic": u["title"],
         "task": ("Review: " if exam_like else "Keep it sharp: ") + u["tasks"][k % len(u["tasks"])],
         "mode": "review" if exam_like else "practice"},
        {"topic": v["title"],
         "task": _unit_task_text("flashcard" if exam_like else "practice", v, k),
         "mode": "flashcard" if exam_like else "practice"},
        {"topic": "Stretch goal", "task": "Try something a little harder than last week",
         "mode": "project" if not exam_like else "practice"},
    ]


def _is_weekly_review_day(d: date, k: int, all_dates: List[date]) -> bool:
    """Last study date of its ISO week, when the user studies 4+ days."""
    if k + 1 >= len(all_dates):
        return False
    week = d.isocalendar()[:2]
    if all_dates[k + 1].isocalendar()[:2] == week:
        return False
    same_week = sum(1 for x in all_dates[max(0, k - 6): k + 1] if x.isocalendar()[:2] == week)
    return same_week >= 4


def _recent_unit_titles(alloc, units, k, n) -> List[str]:
    seen: List[str] = []
    for i in range(k, max(-1, k - 14), -1):
        t = units[alloc[i]["unit"]]["title"] if i < len(alloc) else None
        if t and t not in seen:
            seen.append(t)
        if len(seen) >= n:
            break
    return seen or [units[0]["title"]]


# ============================================================
# OVERVIEW (for the plan screen)
# ============================================================

def overview(
    *,
    roadmap: dict,
    start: date,
    deadline: Optional[date],
    days: List[int],
    today: date,
) -> dict:
    """Phase date ranges + where `today` falls. Cheap enough per request."""
    end = roadmap_end(start, deadline, roadmap)
    all_dates = study_dates(start, end, days)
    alloc = allocate(roadmap, len(all_dates))
    phases = roadmap.get("phases") or []
    units = flat_units(roadmap)

    ranges: dict = {}
    for d, a in zip(all_dates, alloc):
        r = ranges.setdefault(a["phase"], [d, d])
        r[1] = d

    cur_idx = None
    for i, d in enumerate(all_dates):
        if d >= today:
            cur_idx = i
            break
    current_phase = alloc[cur_idx]["phase"] if cur_idx is not None and alloc else None
    current_unit = units[alloc[cur_idx]["unit"]]["title"] if cur_idx is not None and alloc else None

    out_phases = []
    for pi, p in enumerate(phases):
        r = ranges.get(pi)
        if current_phase is None:
            status = "done" if r and r[1] < today else "upcoming"
        elif pi < current_phase:
            status = "done"
        elif pi == current_phase:
            status = "current"
        else:
            status = "upcoming"
        out_phases.append({
            "name": p["name"],
            "milestone": p.get("milestone") or "",
            "weeks": p["weeks"],
            "start_date": r[0].isoformat() if r else None,
            "end_date": r[1].isoformat() if r else None,
            "status": status,
            "units": [u["title"] for u in p["units"]],
        })

    total_weeks = max(1, math.ceil(((end - start).days + 1) / 7))
    current_week = min(max(((today - start).days // 7) + 1, 1), total_weeks)
    return {
        "phases": out_phases,
        "current_phase": current_phase,
        "current_unit": current_unit,
        "total_weeks": total_weeks,
        "current_week": current_week,
        "end_date": end.isoformat(),
        "finished": today > end,
    }


def phase_needing_detail(
    *, roadmap: dict, start: date, deadline: Optional[date], days: List[int],
    today: date, min_study_days: int = 60,
) -> Optional[int]:
    """Index of the current phase if it is long and still coarse."""
    end = roadmap_end(start, deadline, roadmap)
    all_dates = study_dates(start, end, days)
    alloc = allocate(roadmap, len(all_dates))
    counts: dict = {}
    for a in alloc:
        counts[a["phase"]] = counts.get(a["phase"], 0) + 1
    for d, a in zip(all_dates, alloc):
        if d >= today:
            p = roadmap["phases"][a["phase"]]
            n_days = counts.get(a["phase"], 0)
            # Coarse = long phase with fewer than ~1 unit per 15 study days.
            if (not p.get("detailed") and n_days >= min_study_days
                    and len(p["units"]) * 15 < n_days):
                return a["phase"]
            return None
    return None