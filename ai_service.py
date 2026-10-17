import json
import uuid
import os
import traceback
from dotenv import load_dotenv
from datetime import date, datetime, timedelta
from typing import List, Optional
from openai import AsyncOpenAI

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-api-key-here")

# Point the OpenAI client to the Gemini API endpoint
client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# ==========================================
# CANONICAL VALUES
# ==========================================

VALID_MODES = {"review", "flashcard", "feynman"}
VALID_PRIORITIES = {"normal", "high", "weak_area"}
VALID_SESSION_TYPES = {"study", "review", "rest"}

_MODE_ALIASES = {
    "review": "review",
    "reviews": "review",
    "revision": "review",
    "revise": "review",
    "flashcard": "flashcard",
    "flashcards": "flashcard",
    "flash card": "flashcard",
    "flash cards": "flashcard",
    "feynman": "feynman",
    "feynman technique": "feynman",
}

_PRIORITY_ALIASES = {
    "normal": "normal",
    "medium": "normal",
    "regular": "normal",
    "standard": "normal",
    "high": "high",
    "urgent": "high",
    "important": "high",
    "weak_area": "weak_area",
    "weak area": "weak_area",
    "weak": "weak_area",
    "weakness": "weak_area",
}


def _normalize_mode(value) -> str:
    if not isinstance(value, str):
        return "review"
    return _MODE_ALIASES.get(value.strip().lower(), "review")


def _normalize_priority(value) -> str:
    if not isinstance(value, str):
        return "normal"
    key = value.strip().lower().replace("_", " ")
    return _PRIORITY_ALIASES.get(key, "normal")


def _normalize_session_type(value) -> str:
    if not isinstance(value, str):
        return "study"
    v = value.strip().lower()
    return v if v in VALID_SESSION_TYPES else "study"


# ==========================================
# PERSONA HINTS
# ==========================================

PERSONA_HINTS = {
    "university": (
        "The student is at university. Use exam-focused language and "
        "prioritise deep understanding. Recommend 45–90 minute sessions, "
        "past papers, and spaced repetition. Assume they can handle dense "
        "topics in a single sitting."
    ),
    "professional": (
        "The user is a working professional preparing for a certification "
        "(ACCA, CFA, ICAN, PMP, etc.). Recommend 30–60 minute sessions, "
        "early-morning or evening slots, and tie every topic back to the "
        "exam syllabus objectives. Assume limited weekday time."
    ),
    "high_school": (
        "The student is preparing for WAEC, JAMB, or NECO. Use exam-board "
        "language. Prioritise past-question practice and shorter, more "
        "frequent sessions (25–40 minutes). Reinforce fundamentals and "
        "common exam traps."
    ),
    "self_improvement": (
        "The user is a casual, self-directed learner — chess, a new "
        "language, guitar, coding for fun. Keep sessions short and varied "
        "(20–40 minutes), project- or practice-based, and low-pressure. "
        "Alternate between theory and hands-on practice."
    ),
}


# ==========================================
# 1. AI SOLVE ENGINE
# ==========================================

async def generate_gemini_solution(question_text: str) -> Optional[dict]:
    """
    Sends the user's question to the Gemini API and forces it to return
    a highly structured JSON response matching our Flutter app's expectations.
    """

    system_prompt = """
    You are myLB AI, an expert academic tutor.
    Analyze the user's question and provide a clear, step-by-step solution.

    You MUST respond in valid JSON format exactly matching this structure:
    {
      "steps": [
        {
          "step_number": 1,
          "text": "Your detailed explanation for this step...",
          "highlight_terms": [
            {"term": "exact word to highlight", "color": "mint"}
          ]
        }
      ],
      "confidence_score": 0.95
    }

    Rules for highlight_terms:
    - Use 'mint' for positive concepts, formulas, or correct answers.
    - Use 'peach' for warnings, common mistakes, or critical exceptions.
    - Do not highlight whole sentences, only key terms.
    - If there are no terms to highlight, leave the array empty [].

    Rules for confidence_score:
    - A float between 0.0 and 1.0.
    - Be honest: if the question is ambiguous or you're unsure, use a lower value.
    """

    try:
        response = await client.chat.completions.create(
            model="gemini-3.8-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Solve this: {question_text}"},
            ],
            response_format={"type": "json_object"},
        )

        ai_data = json.loads(response.choices[0].message.content)

        if not isinstance(ai_data.get("steps"), list):
            ai_data["steps"] = []

        cleaned_steps = []
        for i, step in enumerate(ai_data["steps"], start=1):
            if not isinstance(step, dict):
                continue
            highlights = step.get("highlight_terms") or []
            if not isinstance(highlights, list):
                highlights = []
            clean_highlights = [
                {
                    "term": str(h.get("term", "")),
                    "color": h.get("color") if h.get("color") in ("mint", "peach") else "mint",
                }
                for h in highlights
                if isinstance(h, dict) and h.get("term")
            ]
            cleaned_steps.append(
                {
                    "step_number": int(step.get("step_number") or i),
                    "text": str(step.get("text") or "").strip(),
                    "highlight_terms": clean_highlights,
                }
            )

        ai_data["steps"] = cleaned_steps

        confidence = ai_data.get("confidence_score", 0.90)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.90
        ai_data["confidence_score"] = max(0.0, min(confidence, 1.0))

        ai_data["solution_id"] = f"sol_{uuid.uuid4().hex[:8]}"
        ai_data["canvas_links"] = []

        return ai_data

    except Exception as e:
        print(f"Gemini AI Solve Error: {e}")
        traceback.print_exc()
        return None


# ==========================================
# 2. AI PHASE SKELETON GENERATOR
# ==========================================

MIN_PHASE_WEEKS = 2
MIN_PHASES = 2
MAX_PHASES = 6

async def generate_gemini_phase_plan(
    goal: str,
    goal_type: Optional[str],
    total_weeks: int,
    minutes_per_day: Optional[int] = None,
    study_days_per_week: Optional[int] = None,
    progress_summary: Optional[str] = None,
) -> Optional[dict]:
    """
    Generate a 2–6 phase skeleton spanning `total_weeks`.
    """

    if total_weeks < MIN_PHASE_WEEKS:
        return None

    today_str = date.today().isoformat()

    persona_block = PERSONA_HINTS.get(
        goal_type or "",
        "The user's background is unspecified — keep sessions moderate "
        "length and balanced between theory and practice.",
    )

    availability_line = ""
    if minutes_per_day and study_days_per_week:
        availability_line = (
            f"They can study {minutes_per_day} minutes/day, "
            f"{study_days_per_week} days/week."
        )

    rebase_block = ""
    if progress_summary:
        rebase_block = f"""
        IMPORTANT — this is a REBASE, not the initial plan.
        The user has already completed part of the arc. Replan the
        REMAINING {total_weeks} weeks based on:
        {progress_summary}
        Compress anything they've already mastered. Expand anything
        they struggled with.
        """

    system_prompt = f"""
    You are myLB AI, an expert curriculum architect.
    The user is preparing for "{goal}" over the next {total_weeks} weeks.
    Today is {today_str}.

    LEARNER PROFILE:
    {persona_block}

    {availability_line}
    {rebase_block}

    Produce between {MIN_PHASES} and {MAX_PHASES} contiguous phases
    covering weeks 1 through {total_weeks}. Phase boundaries are
    inclusive: phase 1 might be weeks 1–4, phase 2 weeks 5–11, etc.
    They must not overlap and must have no gaps.

    A good arc front-loads foundations, spends the middle on depth
    and structured practice, and ends with full-length practice and
    a short taper. Longer spans naturally split into more phases —
    do not force the same shape for 13 weeks as for 34 weeks.

    You MUST respond in valid JSON:
    {{
      "phases": [
        {{
          "name": "Diagnostic & Foundations",
          "start_week": 1,
          "end_week": 4,
          "focus": ["baseline test", "arithmetic", "algebra basics"],
          "milestone": "Complete diagnostic and log weak areas"
        }}
      ],
      "total_weeks": {total_weeks}
    }}

    Rules:
    - 'start_week' of the first phase must be 1.
    - 'end_week' of the last phase must be {total_weeks}.
    - Each phase's start_week must equal previous phase's end_week + 1.
    - 'focus' should be 3–6 short topic strings, not sentences.
    - 'milestone' is one concrete measurable checkpoint.
    """

    try:
        response = await client.chat.completions.create(
            model="gemini-3.8-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Build my {total_weeks}-week phase plan for {goal}.",
                },
            ],
            response_format={"type": "json_object"},
        )

        data = json.loads(response.choices[0].message.content)

        raw_phases = data.get("phases")
        if not isinstance(raw_phases, list):
            return None

        clean_phases = []
        for raw in raw_phases:
            if not isinstance(raw, dict):
                continue
            try:
                start = int(raw.get("start_week") or 0)
                end = int(raw.get("end_week") or 0)
            except (TypeError, ValueError):
                continue
            if start < 1 or end < start or end > total_weeks:
                continue

            focus = raw.get("focus") or []
            if not isinstance(focus, list):
                focus = []
            focus = [str(f).strip() for f in focus if str(f).strip()][:6]

            clean_phases.append(
                {
                    "name": str(raw.get("name") or "Phase").strip()[:60],
                    "start_week": start,
                    "end_week": end,
                    "focus": focus,
                    "milestone": str(raw.get("milestone") or "").strip()[:200],
                }
            )

        if not clean_phases:
            return None

        clean_phases.sort(key=lambda p: p["start_week"])

        if clean_phases[0]["start_week"] != 1:
            clean_phases[0]["start_week"] = 1

        if clean_phases[-1]["end_week"] != total_weeks:
            clean_phases[-1]["end_week"] = total_weeks

        for i in range(1, len(clean_phases)):
            expected_start = clean_phases[i - 1]["end_week"] + 1
            if clean_phases[i]["start_week"] != expected_start:
                clean_phases[i]["start_week"] = expected_start

        clean_phases = [
            p for p in clean_phases
            if p["end_week"] >= p["start_week"]
        ]

        if not clean_phases:
            return None

        if len(clean_phases) > MAX_PHASES:
            clean_phases = clean_phases[:MAX_PHASES]
            clean_phases[-1]["end_week"] = total_weeks

        return {
            "phases": clean_phases,
            "total_weeks": total_weeks,
        }

    except Exception as e:
        print(f"Gemini Phase Plan Error: {e}")
        traceback.print_exc()
        return None


# ==========================================
# 3. AI WINDOW GENERATOR
# ==========================================

async def generate_gemini_study_plan(
    goal: str,
    target_date: Optional[date] = None,
    days_remaining: Optional[int] = None,
    goal_type: Optional[str] = None,
    window_start: Optional[date] = None,
    window_days: int = 7,
    phase_context: Optional[dict] = None,
    recent_topics: Optional[List[str]] = None,
    recent_skips: Optional[List[str]] = None,
    pace_feedback: Optional[str] = None,
    minutes_per_day: Optional[int] = None,
    study_days_per_week: Optional[int] = None,
) -> Optional[dict]:
    """
    Generate sessions for a single window (default 7 days).
    """
    if window_start is None:
        window_start = date.today()

    today_str = window_start.isoformat()
    window_end = window_start + timedelta(days=window_days - 1)

    if target_date and days_remaining is not None:
        context_prompt = (
            f"Their deadline is in {days_remaining} days "
            f"({target_date.isoformat()}). Escalate intensity as the "
            f"deadline approaches."
        )
    else:
        context_prompt = (
            "They are learning continuously without a strict deadline. "
            "Build a sustainable, consistent routine."
        )

    persona_block = PERSONA_HINTS.get(
        goal_type or "",
        "Keep sessions moderate length and balanced between theory and practice.",
    )

    phase_block = ""
    if phase_context:
        focus_str = ", ".join(phase_context.get("focus", []))
        phase_block = f"""
        CURRENT PHASE:
          Name: {phase_context.get("name", "Current phase")}
          Focus: {focus_str}
          Milestone: {phase_context.get("milestone", "")}

        Only schedule topics that serve this focus. Do not introduce
        topics from future phases.
        """

    history_block = ""
    if recent_topics:
        history_block += (
            "RECENT TOPICS (past 14 days):\n"
            + "\n".join(f"  - {t}" for t in recent_topics[:20])
            + "\nDo NOT repeat these unless it's an explicit review.\n"
        )
    if recent_skips:
        history_block += (
            "SKIPPED BY USER:\n"
            + "\n".join(f"  - {t}" for t in recent_skips[:10])
            + "\nReschedule skipped topics EARLY in this window, "
            "and mark them as weak_area priority.\n"
        )
    if pace_feedback:
        history_block += f"USER PACE FEEDBACK: {pace_feedback}\n"

    availability_block = ""
    if minutes_per_day and study_days_per_week:
        availability_block = (
            f"BUDGET: {minutes_per_day} min/day, "
            f"{study_days_per_week} days/week. Do not exceed.\n"
        )

    system_prompt = f"""
    You are myLB AI, an expert academic planner.
    The user is studying for "{goal}". {context_prompt}

    WINDOW: {today_str} through {window_end.isoformat()}
    ({window_days} days).

    LEARNER PROFILE:
    {persona_block}

    {phase_block}
    {availability_block}
    {history_block}

    Generate sessions ONLY within the window above.
    Include rest days where appropriate.

    You MUST respond in valid JSON:
    {{
      "sessions": [
        {{
          "date": "YYYY-MM-DD",
          "time": "16:00",
          "subject": "Specific topic name",
          "duration_mins": 30,
          "mode": "flashcard",
          "priority": "normal"
        }}
      ],
      "nudge": null
    }}

    Rules:
    - 'date' must be between {today_str} and {window_end.isoformat()}.
    - 'mode' must be exactly: "review", "flashcard", or "feynman".
    - 'priority' must be exactly: "normal", "high", or "weak_area".
    - 'subject' must name a specific topic, not the overall goal.
    - Vary modes; do not use the same mode every day.
    """

    try:
        response = await client.chat.completions.create(
            model="gemini-3.8-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Generate my next {window_days} days.",
                },
            ],
            response_format={"type": "json_object"},
        )

        plan_data = json.loads(response.choices[0].message.content)

        if not isinstance(plan_data.get("sessions"), list):
            plan_data["sessions"] = []

        clean_sessions = []
        for raw in plan_data["sessions"]:
            if not isinstance(raw, dict):
                continue

            date_str = str(raw.get("date") or "").strip()

            try:
                s_date = date.fromisoformat(date_str)
            except ValueError:
                continue
            if not (window_start <= s_date <= window_end):
                continue

            subject = str(raw.get("subject") or goal).strip() or goal
            time_str = str(raw.get("time") or "16:00").strip()

            try:
                duration = int(raw.get("duration_mins") or 30)
            except (TypeError, ValueError):
                duration = 30
            if duration <= 0:
                duration = 30
            if duration > 180:
                duration = 180

            clean_sessions.append(
                {
                    "date": date_str,
                    "time": time_str,
                    "subject": subject,
                    "duration_mins": duration,
                    "mode": _normalize_mode(raw.get("mode")),
                    "priority": _normalize_priority(raw.get("priority")),
                }
            )

        plan_data["sessions"] = clean_sessions
        plan_data.setdefault("nudge", None)
        return plan_data

    except Exception as e:
        print(f"Gemini Window Plan Error: {e}")
        traceback.print_exc()
        return None