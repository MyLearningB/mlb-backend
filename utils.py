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
# Moved to gamification.py so the catalog (emoji, traits, stages)
# lives in exactly one place. Kept here as a thin wrapper so any old
# imports keep working.

def get_pet_evolution_data(
    pet_type: str,
    level: int,
    current_xp: int,
) -> dict:
    from gamification import evolution_data
    return evolution_data(pet_type, level, current_xp)