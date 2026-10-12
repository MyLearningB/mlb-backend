from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Optional


# ============================================================
# SHARED CONSTANTS
# ============================================================
#
# Used by main.py (onboarding quiz, study-profile) and by
# routers/plan.py (generate / update). Keeping them here means
# there is one source of truth for what counts as a valid persona.

ALLOWED_GOAL_TYPES = {
    "university",
    "professional",
    "high_school",
    "self_improvement",
}

# Per-persona default emoji + accent colour. Used when the client
# doesn't supply their own, so a uni student and a chess learner
# see visually different chips on the dashboard.
GOAL_TYPE_DEFAULTS = {
    "university":       {"emoji": "🎓", "color_hex": "#7C6EE6"},
    "professional":     {"emoji": "💼", "color_hex": "#3CFFC8"},
    "high_school":      {"emoji": "📚", "color_hex": "#FF8A65"},
    "self_improvement": {"emoji": "✨", "color_hex": "#C8F53C"},
}


# ============================================================
# TIMEZONE HELPER
# ============================================================

def get_local_now(timezone_str: str = "UTC") -> datetime:
    """
    Return the current local datetime for a tz string.
    Falls back to UTC when the header is missing or invalid, so
    a malformed `X-Timezone` never crashes a request.
    """
    try:
        tz = ZoneInfo(timezone_str)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo("UTC")
    return datetime.now(tz)


# ============================================================
# GOAL TYPE HELPERS
# ============================================================

def persona_defaults(goal_type: Optional[str]) -> dict:
    """Return the default emoji / color_hex for a goal_type."""
    return GOAL_TYPE_DEFAULTS.get(
        goal_type or "",
        {"emoji": "🎯", "color_hex": "#7C6EE6"},
    )


def resolve_goal_type(
    payload_goal_type: Optional[str],
    user_goal_type: Optional[str],
) -> str:
    """
    Pick a valid goal_type from (payload, user, fallback).

    Order matters: an explicit request-level persona always wins
    over the user's default, so a uni student can plan a chess
    goal without changing their profile.
    """
    resolved = payload_goal_type or user_goal_type or "self_improvement"
    if resolved not in ALLOWED_GOAL_TYPES:
        resolved = "self_improvement"
    return resolved


# ============================================================
# PET EVOLUTION
# ============================================================
#
# A pet goes through four visual stages as its level climbs:
#
#   stage 0  🥚  level 1–3
#   stage 1  🦊  level 4–7
#   stage 2  🔥  level 8–12
#   stage 3  🐉  level 13+ (maxed)
#
# XP spans: level N lives between (N-1)*100 and N*100.
#   * To reach level 2 you need 100 XP.
#   * To reach level 5 you need 400 XP.
#   * So at level 4 with 350 XP, you're 50% through the level
#     and 50 XP away from level 5.
#
# Two fields are exposed so the client never has to do math:
#
#   * `xp_to_next`     — remaining XP until the next level (50)
#   * `xp_into_level`  — XP earned inside the current level (50)
#   * `level_xp_span`  — total XP the current level requires (100)

def get_pet_evolution_data(
    pet_type: str,
    level: int,
    current_xp: int,
) -> dict:
    evolution_lines = {
        'nova': ['🥚', '🦊', '🔥', '🐉'],
        'pip':  ['🥚', '🐧', '🦅', '🦅'],
        'luna': ['🧶', '🐱', '🐈‍⬛', '🐈‍⬛'],
        'zap':  ['🔋', '🐶', '⚡', '⚡'],
    }

    line = evolution_lines.get(
        pet_type.lower(),
        ['🥚', '✨', '🌟', '💫'],
    )

    # ---- Stage from level ----
    if level <= 3:
        stage = 0
    elif level <= 7:
        stage = 1
    elif level <= 12:
        stage = 2
    else:
        stage = 3

    is_maxed = stage == 3

    # ---- Maxed pets have nothing left to unlock ----
    if is_maxed:
        return {
            "current_emoji": line[stage],
            "next_emoji": line[stage],
            "progress_percent": 1.0,
            "is_maxed": True,
            "xp_to_next": 0,
            "xp_into_level": 0,
            "level_xp_span": 0,
        }

    # ---- Progress within the current level ----
    # level N spans [(N-1)*100, N*100]. The span is always 100,
    # but keeping it as a subtraction documents the intent.
    level_floor = (level - 1) * 100
    level_ceil = level * 100
    span = level_ceil - level_floor

    xp_into_level = max(0, current_xp - level_floor)
    progress_percent = min(xp_into_level / span, 1.0)
    xp_to_next = max(0, level_ceil - current_xp)

    return {
        "current_emoji": line[stage],
        "next_emoji": line[stage + 1],
        "progress_percent": progress_percent,
        "is_maxed": False,
        "xp_to_next": xp_to_next,
        "xp_into_level": xp_into_level,
        "level_xp_span": span,
    }