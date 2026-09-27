import json
import uuid
import os
import traceback
from dotenv import load_dotenv
from datetime import date, datetime, timedelta
from typing import Optional
from openai import AsyncOpenAI

# This tells Python to look for the .env file and load it
load_dotenv()

# Initialize the client pointing to DeepSeek's API
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-deepseek-api-key-here")

client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"  # Overriding the base URL to DeepSeek
)


# ==========================================
# CANONICAL VALUES (must match schemas.py Literals)
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
#
# Injected into the system prompt so the plan shape and language
# change with the learner. This is the single biggest lever for
# "the onboarding choice actually does something."

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
# 1. AI SOLVE ENGINE (Screen 10)
# ==========================================

async def generate_deepseek_solution(question_text: str) -> Optional[dict]:
    """
    Sends the user's question to DeepSeek and forces it to return
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
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Solve this: {question_text}"},
            ],
            response_format={"type": "json_object"},
        )

        ai_data = json.loads(response.choices[0].message.content)

        # ---- Defensive cleanup so SolveResponse(**data) can't 500 ----

        if not isinstance(ai_data.get("steps"), list):
            ai_data["steps"] = []

        cleaned_steps = []
        for i, step in enumerate(ai_data["steps"], start=1):
            if not isinstance(step, dict):
                continue
            highlights = step.get("highlight_terms") or []
            if not isinstance(highlights, list):
                highlights = []
            # Only keep highlights with the expected shape
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
        print(f"DeepSeek AI Solve Error: {e}")
        traceback.print_exc()
        return None


# ==========================================
# 2. AI STUDY PLAN GENERATOR (Screen 9)
# ==========================================

async def generate_deepseek_study_plan(
    goal: str,
    target_date: Optional[date] = None,
    days_remaining: Optional[int] = None,
    goal_type: Optional[str] = None,
) -> Optional[dict]:
    """
    Prompts DeepSeek to generate a structured 7-day study schedule.

    Persona-aware: `goal_type` (university / professional / high_school /
    self_improvement) is injected into the system prompt so that a WAEC
    Biology plan looks very different from an ACCA Financial Reporting plan.

    Adapts dynamically for strict deadlines vs continuous learning.
    """
    today_str = date.today().isoformat()

    # ------------------------------------------------
    # Deadline context
    # ------------------------------------------------
    if target_date and days_remaining is not None:
        context_prompt = (
            f"Their deadline is in {days_remaining} days "
            f"({target_date.isoformat()}). Build the plan to prepare them "
            f"for this specific target, escalating intensity and difficulty "
            f"as the deadline approaches. Front-load fundamentals and "
            f"back-load review + past-paper practice."
        )
        stats_days = days_remaining
    else:
        context_prompt = (
            "They are learning continuously without a strict deadline. "
            "Build a sustainable, consistent foundational routine that they "
            "can keep going for months without burning out."
        )
        stats_days = -1  # Indicates "no deadline" to the frontend

    # ------------------------------------------------
    # Persona context
    # ------------------------------------------------
    persona_block = PERSONA_HINTS.get(
        goal_type or "",
        "The user's background is unspecified — keep sessions moderate "
        "length and balanced between theory and practice.",
    )

    system_prompt = f"""
    You are myLB AI, an expert academic planner.
    The user wants to study for "{goal}". {context_prompt}
    Today is {today_str}.

    LEARNER PROFILE:
    {persona_block}

    Generate a highly realistic, balanced 7-day study plan starting from today.
    Include rest days to prevent burnout.

    You MUST respond in valid JSON format exactly matching this structure:
    {{
      "stats": {{
        "days_remaining": {stats_days},
        "daily_target_mins": 60,
        "topics_count": 5
      }},
      "week": [
        {{
          "date": "YYYY-MM-DD",
          "day_label": "MON",
          "has_session": true,
          "session_type": "study"
        }}
      ],
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

    Rules for 'week':
    - Generate EXACTLY 7 items, one for each of the next 7 days starting today.
    - 'day_label' must be a 3-letter uppercase code (MON, TUE, WED, ...).
    - 'session_type' must be exactly one of: "study", "review", "rest" (lowercase).

    Rules for 'sessions':
    - Generate ONLY sessions for days where has_session is true.
    - 'date' must be YYYY-MM-DD and must fall within the 7-day window.
    - 'time' must be 24-hour HH:MM.
    - 'mode' must be exactly one of: "review", "flashcard", "feynman" (lowercase).
    - 'priority' must be exactly one of: "normal", "high", "weak_area" (lowercase).
    - 'duration_mins' must be a positive integer, appropriate for the learner profile above.
    - 'subject' should name a specific topic, not the overall goal.

    Vary the modes across the week — do not use the same mode every day.
    If there is a deadline, place 'weak_area' priorities later in the week
    on topics that were studied early on.
    """

    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Generate my 7-day study plan for {goal}.",
                },
            ],
            response_format={"type": "json_object"},
        )

        plan_data = json.loads(response.choices[0].message.content)

        # ------------------------------------------------
        # Defensive coercion pass
        # ------------------------------------------------
        # The AI is *told* to return canonical lowercase values, but
        # models drift. This makes the output safe for PlanResponse /
        # SessionDetail no matter what.

        if not isinstance(plan_data.get("sessions"), list):
            plan_data["sessions"] = []

        cleaned_sessions = []
        for raw in plan_data["sessions"]:
            if not isinstance(raw, dict):
                continue

            subject = str(raw.get("subject") or goal).strip() or goal
            date_str = str(raw.get("date") or today_str).strip()
            time_str = str(raw.get("time") or "16:00").strip()

            try:
                duration = int(raw.get("duration_mins") or 30)
            except (TypeError, ValueError):
                duration = 30
            if duration <= 0:
                duration = 30

            cleaned_sessions.append(
                {
                    "date": date_str,
                    "time": time_str,
                    "subject": subject,
                    "duration_mins": duration,
                    "mode": _normalize_mode(raw.get("mode")),
                    "priority": _normalize_priority(raw.get("priority")),
                }
            )

        plan_data["sessions"] = cleaned_sessions

        # Week items — normalise session_type, keep only 7
        if isinstance(plan_data.get("week"), list):
            clean_week = []
            for day in plan_data["week"][:7]:
                if not isinstance(day, dict):
                    continue
                clean_week.append(
                    {
                        "date": str(day.get("date") or "").strip(),
                        "day_label": str(day.get("day_label") or "").strip().upper()[:3],
                        "has_session": bool(day.get("has_session", False)),
                        "session_type": _normalize_session_type(day.get("session_type")),
                    }
                )
            plan_data["week"] = clean_week

        # Ensure nudge key exists (main.py currently overrides to None,
        # but keeping the shape stable helps if that ever changes)
        plan_data.setdefault("nudge", None)

        return plan_data

    except Exception as e:
        print(f"DeepSeek AI Plan Error: {e}")
        traceback.print_exc()
        return None