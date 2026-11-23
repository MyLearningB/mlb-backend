"""
gamification.py — the single source of truth for pets, XP, levels,
evolution, mood and streaks.

Every place that rewards the user (plan sessions, flashcards, Feynman)
calls `award_xp()`. Every place that SHOWS the pet (dashboard, profile,
onboarding) calls `pet_view()`. That way the numbers can never drift
apart again.

Rules
-----
* 100 XP per level. Level is derived from XP and saved on the pet.
* Evolution stages unlock at levels 1 / 4 / 8 / 13.
* Each pet has a real trait that boosts XP (matches what the adoption
  screen promises):
    Nova  — +50% on your first session of the day
    Pip   — +20% on every completed session
    Luna  — +50% for sessions between 20:00 and 04:00 (your time)
    Zap   — +50% for sessions of 20 minutes or less
* Mood is computed from the last day you studied (no DB column):
    studied today → happy, yesterday → sleepy, longer → sad,
    never studied → new
* Streaks use the user's local date everywhere.
"""

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlmodel import Session, select

from models import DailyActivity, Pet, User

# ============================================================
# CONSTANTS
# ============================================================

XP_PER_LEVEL = 100
STAGE_MIN_LEVELS = [1, 4, 8, 13]
DEFAULT_PET_TYPE = "nova"

# Base emoji (index 0) MUST match what the adoption screen shows.
PET_CATALOG: Dict[str, Dict[str, Any]] = {
    "nova": {
        "name": "Nova",
        "title": "The Mysterious",
        "desc": "Nova grows stronger every day you show up.",
        "emoji_line": ["🥚", "🦎", "🐲", "🐉"],
        "stage_names": ["Egg", "Hatchling", "Drake", "Dragon"],
        "trait": "Daily ritual",
        "trait_desc": "+50% XP on your first session of the day",
    },
    "pip": {
        "name": "Pip",
        "title": "The Eager",
        "desc": "Pip gets excited every time you finish a session.",
        "emoji_line": ["🐥", "🐤", "🐦", "🦅"],
        "stage_names": ["Chick", "Fledgling", "Songbird", "Eagle"],
        "trait": "Cheerleader",
        "trait_desc": "+20% XP on every session you complete",
    },
    "luna": {
        "name": "Luna",
        "title": "The Night Owl",
        "desc": "Luna thrives when you study after dark.",
        "emoji_line": ["🌙", "🐱", "🐈", "🐆"],
        "stage_names": ["Moonbeam", "Kitten", "Night Cat", "Shadow Panther"],
        "trait": "Night shift",
        "trait_desc": "+50% XP for sessions between 8pm and 4am",
    },
    "zap": {
        "name": "Zap",
        "title": "The Sprinter",
        "desc": "Zap loves short, intense bursts of focus.",
        "emoji_line": ["⚡", "🐇", "🦊", "🐺"],
        "stage_names": ["Spark", "Bunny", "Swift Fox", "Storm Wolf"],
        "trait": "Quick bursts",
        "trait_desc": "+50% XP for sessions of 20 minutes or less",
    },
}


def catalog_entry(pet_type: Optional[str]) -> Dict[str, Any]:
    return PET_CATALOG.get((pet_type or "").lower(), PET_CATALOG[DEFAULT_PET_TYPE])


def catalog_list() -> List[Dict[str, Any]]:
    """Shape used by GET /pets/catalog."""
    return [
        {
            "id": pet_id,
            "name": e["name"],
            "title": e["title"],
            "desc": e["desc"],
            "emoji": e["emoji_line"][0],
            "evolution_line": e["emoji_line"],
            "stage_names": e["stage_names"],
            "trait": e["trait"],
            "trait_desc": e["trait_desc"],
        }
        for pet_id, e in PET_CATALOG.items()
    ]


# ============================================================
# LEVELS & EVOLUTION
# ============================================================

def level_for_xp(xp: Optional[int]) -> int:
    return max(0, int(xp or 0)) // XP_PER_LEVEL + 1


def stage_for_level(level: int) -> int:
    stage = 0
    for i, min_level in enumerate(STAGE_MIN_LEVELS):
        if level >= min_level:
            stage = i
    return stage


def evolution_data(pet_type: Optional[str], level: int, xp: int) -> Dict[str, Any]:
    """
    Everything the client needs to draw the pet + progress bar.

    `is_maxed` now means "final evolution reached". Levels keep going
    after that, so the XP bar never freezes at 100%.
    """
    entry = catalog_entry(pet_type)
    line = entry["emoji_line"]
    xp = max(0, int(xp or 0))
    level = max(1, int(level or 1))

    stage = stage_for_level(level)
    is_maxed = stage >= len(line) - 1

    level_floor = (level - 1) * XP_PER_LEVEL
    into_level = max(0, min(XP_PER_LEVEL, xp - level_floor))

    return {
        "current_emoji": line[stage],
        "next_emoji": line[stage] if is_maxed else line[stage + 1],
        "progress_percent": round(into_level / XP_PER_LEVEL, 4),
        "is_maxed": is_maxed,
        "xp_to_next": XP_PER_LEVEL - into_level,
        "xp_into_level": into_level,
        "level_xp_span": XP_PER_LEVEL,
        "stage": stage,
        "stage_name": entry["stage_names"][stage],
        "next_stage_level": None if is_maxed else STAGE_MIN_LEVELS[stage + 1],
        "evolution_line": line,
        "title": entry["title"],
        "trait": entry["trait"],
        "trait_desc": entry["trait_desc"],
    }


# ============================================================
# MOOD
# ============================================================

def mood_for(last_active: Optional[date], today: date) -> str:
    if last_active is None:
        return "new"
    if last_active >= today:
        return "happy"
    if last_active == today - timedelta(days=1):
        return "sleepy"
    return "sad"


def mood_message(name: str, mood: str) -> str:
    return {
        "happy": f"{name} is buzzing — you studied today!",
        "sleepy": f"{name} is waiting for you. Study today to keep your streak alive.",
        "sad": f"{name} misses you. One short session will cheer them up.",
        "new": f"{name} can't wait for your first study session!",
    }.get(mood, f"{name} is ready when you are.")


# ============================================================
# STREAKS (one implementation, local date)
# ============================================================

def streak_view(user: User, today: date) -> Dict[str, Any]:
    last = user.last_active_date
    alive = last is not None and last >= today - timedelta(days=1)
    return {
        "days": (user.current_streak or 0) if alive else 0,
        "active_today": last is not None and last >= today,
        "longest": user.longest_streak or 0,
    }


def _bump_streak(user: User, today: date) -> bool:
    """Returns True if the streak grew (first activity today)."""
    if user.last_active_date == today:
        return False
    if user.last_active_date == today - timedelta(days=1):
        user.current_streak = (user.current_streak or 0) + 1
    else:
        user.current_streak = 1
    if user.current_streak > (user.longest_streak or 0):
        user.longest_streak = user.current_streak
    user.last_active_date = today
    return True


# ============================================================
# TRAITS
# ============================================================

def trait_bonus(
    pet_type: Optional[str],
    *,
    local_now: datetime,
    duration_mins: Optional[int],
    first_today: bool,
) -> Tuple[float, Optional[str]]:
    t = (pet_type or "").lower()
    name = catalog_entry(t)["name"]
    if t == "nova" and first_today:
        return 0.5, f"{name}'s daily ritual"
    if t == "pip":
        return 0.2, f"{name}'s cheer"
    if t == "luna" and (local_now.hour >= 20 or local_now.hour < 4):
        return 0.5, f"{name}'s night shift"
    if t == "zap" and duration_mins is not None and 0 < duration_mins <= 20:
        return 0.5, f"{name}'s quick burst"
    return 0.0, None


# ============================================================
# AWARD XP — the only function that hands out XP
# ============================================================

def get_pet(db: Session, user_id: int) -> Optional[Pet]:
    return db.exec(select(Pet).where(Pet.user_id == user_id)).first()


def award_xp(
    db: Session,
    user: User,
    base_xp: int,
    *,
    local_now: datetime,
    source: str,
    duration_mins: Optional[int] = None,
    minutes_studied: int = 0,
) -> Dict[str, Any]:
    """
    Adds XP (plus the pet's trait bonus) to: today's DailyActivity,
    the pet (with level-up / evolution), the streak and co-op quests.

    Does NOT commit — the caller commits once, so everything succeeds
    or nothing does.
    """
    base_xp = max(0, int(base_xp or 0))
    today = local_now.date()
    today_str = today.isoformat()

    activity = db.exec(
        select(DailyActivity).where(
            DailyActivity.user_id == user.id,
            DailyActivity.date == today_str,
        )
    ).first()
    first_today = activity is None or (activity.xp_earned or 0) == 0

    pet = get_pet(db, user.id)

    bonus_rate, bonus_reason = (0.0, None)
    if pet is not None and base_xp > 0:
        bonus_rate, bonus_reason = trait_bonus(
            pet.pet_type,
            local_now=local_now,
            duration_mins=duration_mins,
            first_today=first_today,
        )
    bonus_xp = int(round(base_xp * bonus_rate))
    total_xp = base_xp + bonus_xp

    # ---- Daily activity ----
    if activity is None:
        activity = DailyActivity(
            user_id=user.id, date=today_str, xp_earned=0, study_time_mins=0,
        )
    activity.xp_earned = (activity.xp_earned or 0) + total_xp
    activity.study_time_mins = (activity.study_time_mins or 0) + max(0, int(minutes_studied or 0))
    db.add(activity)

    # ---- Pet ----
    pet_result = None
    if pet is not None:
        old_level = level_for_xp(pet.xp)
        old_stage = stage_for_level(old_level)
        pet.xp = (pet.xp or 0) + total_xp
        pet.level = level_for_xp(pet.xp)
        new_stage = stage_for_level(pet.level)
        db.add(pet)

        evo = evolution_data(pet.pet_type, pet.level, pet.xp)
        pet_result = {
            "name": pet.pet_name,
            "type": pet.pet_type,
            "xp": pet.xp,
            "old_level": old_level,
            "level": pet.level,
            "leveled_up": pet.level > old_level,
            "evolved": new_stage > old_stage,
            "emoji": evo["current_emoji"],
            "stage_name": evo["stage_name"],
            "progress_percent": evo["progress_percent"],
        }

    # ---- Streak ----
    streak_grew = False
    if total_xp > 0:
        streak_grew = _bump_streak(user, today)
        db.add(user)

    # ---- Co-op quests (no commit inside) ----
    if total_xp > 0:
        from routers.community import contribute_to_group_quests
        contribute_to_group_quests(user_id=user.id, xp_amount=total_xp, db=db, commit=False)

    return {
        "source": source,
        "xp_earned": total_xp,
        "base_xp": base_xp,
        "bonus_xp": bonus_xp,
        "bonus_reason": bonus_reason,
        "pet": pet_result,
        "streak": {
            "days": user.current_streak or 0,
            "extended": streak_grew,
        },
    }


# ============================================================
# PET VIEW — what dashboard / profile / onboarding show
# ============================================================

def pet_view(
    pet: Optional[Pet],
    user: User,
    today: date,
    xp_history: Optional[List[int]] = None,
) -> Dict[str, Any]:
    pet_type = pet.pet_type if pet else DEFAULT_PET_TYPE
    entry = catalog_entry(pet_type)
    xp = pet.xp if pet else 0
    level = level_for_xp(xp)
    name = pet.pet_name if pet else entry["name"]
    mood = mood_for(user.last_active_date, today)

    return {
        "name": name,
        "type": pet_type,
        "level": level,
        "xp": xp,
        "mood": mood,
        "mood_message": mood_message(name, mood),
        "xp_history": xp_history if xp_history is not None else [0] * 7,
        **evolution_data(pet_type, level, xp),
    }


def xp_history_last_7(db: Session, user_id: int, today: date) -> List[int]:
    days = [(today - timedelta(days=i)).isoformat() for i in range(6, -1, -1)]
    rows = db.exec(
        select(DailyActivity).where(
            DailyActivity.user_id == user_id,
            DailyActivity.date.in_(days),
        )
    ).all()
    by_day = {r.date: r.xp_earned for r in rows}
    return [by_day.get(d, 0) for d in days]