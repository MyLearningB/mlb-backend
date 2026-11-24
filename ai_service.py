import json
import os
import traceback
import uuid
from datetime import date
from typing import List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI, Timeout

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-api-key-here")

SOLVE_MODEL = os.getenv("SOLVE_AI_MODEL", "gemini-3.8-flash")
PLAN_MODEL = os.getenv("PLAN_AI_MODEL", "gemini-3.8-flash")

# Point the OpenAI client to the Gemini API endpoint.
# The timeout stops a hung request from holding a worker forever.
client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=Timeout(75, connect=10),
    max_retries=1,
)


# ==========================================
# PERSONA HINTS (a guide only — never a limit)
# ==========================================
#
# The onboarding answer (goal_type) only sets tone and session
# style. The GOAL text decides what the plan is about, so a
# university student can still plan chess or knitting.

PERSONA_HINTS = {
    "university": "The learner is a university student: comfortable with dense material and longer sessions.",
    "professional": "The learner is a working professional: limited weekday time, values efficiency and real-world relevance.",
    "high_school": "The learner is a secondary-school student: keep language simple, tasks bite-sized, lots of past-question practice for exams.",
    "self_improvement": "The learner is a self-directed learner: keep it encouraging, hands-on and low-pressure.",
}

LEVEL_HINTS = {
    "beginner": "They are starting from zero — begin with fundamentals and tools.",
    "intermediate": "They know the basics — skip absolute fundamentals, focus on depth and practice.",
    "advanced": "They are advanced — focus on weak spots, hard material and timed/real-world practice.",
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
            model=SOLVE_MODEL,
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
# 2. STUDY PLAN ROADMAP (one call per plan)
# ==========================================
#
# This is the ONLY AI call a new plan makes. It returns a curriculum
# (phases -> units -> concrete tasks). plan_scheduler.py then spreads
# it over the calendar with no further AI cost. The output is checked
# and repaired by plan_scheduler.normalize_roadmap().

def _units_target(total_weeks: int) -> tuple[int, int]:
    """(min_units, max_units) — about one unit per week, capped for long plans."""
    if total_weeks <= 2:
        return 2, 5
    if total_weeks <= 12:
        return max(3, total_weeks - 2), total_weeks + 2
    if total_weeks <= 52:
        return 12, 30
    return 20, 36


def _json_from(response) -> Optional[dict]:
    text = response.choices[0].message.content or ""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start:end + 1])
        except (ValueError, TypeError):
            return None
    return data if isinstance(data, dict) else None


async def generate_plan_roadmap(
    goal: str,
    *,
    kind_hint: str,
    level: str,
    total_weeks: int,
    has_deadline: bool,
    minutes_per_day: int,
    days_per_week: int,
    session_mins: int,
    goal_type: Optional[str],
    today: date,
    notes: Optional[str] = None,
    progress_summary: Optional[str] = None,
    feedback: Optional[str] = None,
) -> Optional[dict]:
    """Return the raw roadmap JSON, or None on any failure."""
    min_u, max_u = _units_target(total_weeks)
    persona = PERSONA_HINTS.get(goal_type or "", "Learner background unknown — keep it balanced.")
    level_hint = LEVEL_HINTS.get(level, LEVEL_HINTS["beginner"])
    span = (
        f"They must be ready in {total_weeks} weeks (fixed deadline)."
        if has_deadline
        else f"No deadline — design a realistic {total_weeks}-week path."
    )
    long_note = ""
    if total_weeks > 52:
        long_note = (
            "This is a multi-year plan. Give the FIRST phase detailed units (about one "
            "per 2 weeks); keep later phases coarse (broad units) — they are detailed "
            "later. Start gently: early phases build habits and fundamentals, "
            "intensity rises towards the end."
        )

    extra = ""
    if notes:
        extra += f"\nLEARNER NOTES (follow them): {notes[:600]}"
    if progress_summary:
        extra += (
            f"\nTHIS IS A RE-PLAN for the REMAINING {total_weeks} weeks. "
            f"Progress so far: {progress_summary[:800]}. Do not repeat what is done "
            "unless it needs review."
        )
    if feedback:
        extra += f"\nLEARNER FEEDBACK ON THE OLD PLAN (act on it): {feedback[:500]}"

    system_prompt = f"""You are myLB, an expert learning designer. Build a study roadmap for ANY goal:
exams (known or obscure), school courses, skills (chess, sewing, coding, music) or habits.

GOAL: "{goal}"
Today: {today.isoformat()}. {span}
Time: about {minutes_per_day} min/day, {days_per_week} days/week, sessions of ~{session_mins} min.
{persona}
{level_hint}
{long_note}{extra}

Decide goal_kind:
- "exam": any test, certification or exam (SAT, GRE, WAEC, ICAN, a uni final, an unknown local exam).
- "course": working through a school/uni course or syllabus without a single exam focus.
- "skill": a practical skill or hobby (chess, knitting, guitar, drawing, coding).
- "habit": a routine to build (read daily, journal, exercise consistency).

Rules:
- 2 to 6 phases in order; each has "weeks" (positive integers summing to {total_weeks}) and one measurable "milestone".
- Between {min_u} and {max_u} units in total. A unit is a topic or skill block.
- Each unit has 3 to 5 tasks. A task is ONE concrete thing to do in a single ~{session_mins}-min session,
  starting with a verb, max 90 characters, measurable where possible.
  Exam example: "Do 20 SAT linear-equation questions and mark them".
  Skill example: "Knit 10 rows of garter stitch with even tension".
- For exams use the real syllabus topics. If you do not know the exam, infer typical content from its name
  and make the first task "Get the official syllabus for <exam>".
- Exams end with timed mock practice and a short final review. Skills end with a real project.
- No dates, no markdown. Short names.

Respond with JSON only:
{{"goal_kind":"exam","title":"GRE Prep","emoji":"📝","summary":"one sentence",
 "phases":[{{"name":"Foundations","weeks":4,"milestone":"Score 150+ on a diagnostic",
   "units":[{{"title":"Arithmetic & number properties","tasks":["...","...","..."]}}]}}]}}"""

    try:
        response = await client.chat.completions.create(
            model=PLAN_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Build my roadmap for: {goal}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
        )
        return _json_from(response)
    except Exception as e:
        print(f"Gemini Roadmap Error: {e}")
        traceback.print_exc()
        return None


# ==========================================
# 3. PHASE REFINEMENT (long plans only)
# ==========================================
#
# Multi-year plans start with a coarse roadmap. When the learner
# reaches a long phase, plan_rollover asks for a detailed unit list
# for THAT phase only. At most one call per phase (max 6 per plan).

async def refine_plan_phase(
    goal: str,
    *,
    kind: str,
    level: str,
    phase: dict,
    phase_weeks: int,
    minutes_per_day: int,
    done_units: List[str],
) -> Optional[List[dict]]:
    """Return a detailed list of units for one phase, or None."""
    min_u = max(4, min(phase_weeks, 10))
    max_u = max(min_u, min(phase_weeks + 2, 16))
    coarse = ", ".join(u.get("title", "") for u in phase.get("units", []))
    done = ", ".join(done_units[-15:]) or "nothing yet"
    system_prompt = f"""You are myLB, an expert learning designer.
Goal: "{goal}" (kind: {kind}, level: {level}).
Current phase: "{phase.get('name')}" lasting {phase_weeks} weeks, about {minutes_per_day} min/day.
Phase milestone: {phase.get('milestone') or 'n/a'}.
Coarse outline for this phase: {coarse}.
Already covered: {done}.

Break this phase into {min_u}-{max_u} detailed units in learning order. Each unit has 3-5 tasks;
each task is one concrete ~session-sized action starting with a verb, max 90 characters.

Respond with JSON only: {{"units":[{{"title":"...","tasks":["...","...","..."]}}]}}"""
    try:
        response = await client.chat.completions.create(
            model=PLAN_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Detail this phase."},
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
        )
        data = _json_from(response)
        units = data.get("units") if data else None
        return units if isinstance(units, list) and units else None
    except Exception as e:
        print(f"Gemini Refine Error: {e}")
        traceback.print_exc()
        return None