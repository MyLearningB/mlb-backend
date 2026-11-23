from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Header, status
from sqlmodel import Session, select, func
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date, datetime, timedelta

# Import your existing project dependencies
from database import get_session
from security import get_current_user
from models import (
    User, Pet, DailyActivity, StudySession, Quest, Reminder, Flashcard,
    StudySet, Feedback, FeynmanSession,
)

from gamification import get_pet, pet_view, streak_view, xp_history_last_7
from utils import get_local_now

router = APIRouter()

# ─────────────────────────────────────────────────────────────
# HELPER: CALL THIS WHENEVER A USER EARNS XP
# ─────────────────────────────────────────────────────────────
def update_user_streak(user: User, session: Session):
    """Updates the user's streak in O(1) time. Call this when an activity is completed."""
    today = datetime.now().date()

    if user.last_active_date == today:
        return  # Already studied today

    yesterday = today - timedelta(days=1)
    if user.last_active_date == yesterday:
        user.current_streak += 1
    else:
        user.current_streak = 1  # Streak reset/started

    if user.current_streak > user.longest_streak:
        user.longest_streak = user.current_streak

    user.last_active_date = today
    session.add(user)
    session.commit()

# ─────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────
class ProfileUpdate(BaseModel):
    display_name: Optional[str] = None
    username: Optional[str] = Field(None, max_length=20)
    bio: Optional[str] = Field(None, max_length=120)
    study_goal: Optional[str] = None  # Supports the Quiz onboarding screen

class ReminderCreate(BaseModel):
    type: str
    title: str
    schedule: str
    time: Optional[str] = None
    days_of_week: Optional[List[int]] = None
    enabled: bool = True
    color: Optional[str] = "#5040D0"

class ReminderUpdate(BaseModel):
    enabled: Optional[bool] = None
    time: Optional[str] = None
    title: Optional[str] = None
    days_of_week: Optional[List[int]] = None
    color: Optional[str] = None

class AccountDeleteRequest(BaseModel):
    confirmation: str

class SettingsUpdate(BaseModel):
    srs_intensity: Optional[str] = None
    daily_goal_mins: Optional[int] = None
    public_profile: Optional[bool] = None
    push_notifications: Optional[bool] = None
    quest_updates: Optional[bool] = None
    access_requests_alerts: Optional[bool] = None

class FeedbackRequest(BaseModel):
    message: str

# ─────────────────────────────────────────────────────────────
# 11.1 PROFILE ROOT ENDPOINTS
# ─────────────────────────────────────────────────────────────

@router.get("/profile", status_code=status.HTTP_200_OK)
def get_profile(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    x_timezone: str = Header("UTC"),
):
    """Returns full user profile + pet + streak + stats.

    Pet and streak come from gamification.py — the exact same numbers
    the dashboard shows.
    """
    today = get_local_now(x_timezone).date()

    pet = get_pet(session, current_user.id)
    pet_data = pet_view(
        pet,
        current_user,
        today,
        xp_history_last_7(session, current_user.id, today),
    )
    streak = streak_view(current_user, today)

    quests_won = session.exec(
        select(func.count(Quest.id)).where(
            Quest.user_id == current_user.id,
            Quest.progress >= Quest.target,
        )
    ).one()

    total_study_minutes = session.exec(
        select(func.coalesce(func.sum(DailyActivity.study_time_mins), 0)).where(
            DailyActivity.user_id == current_user.id,
        )
    ).one()

    mastered_cards = session.exec(
        select(func.count(Flashcard.id))
        .join(StudySet, StudySet.id == Flashcard.study_set_id)
        .where(
            StudySet.user_id == current_user.id,
            Flashcard.is_weak == False,  # noqa: E712
        )
    ).one()

    recent_activity_query = session.exec(
        select(StudySession)
        .where(
            StudySession.user_id == current_user.id,
            StudySession.completed == True,  # noqa: E712
        )
        .order_by(StudySession.date.desc())
        .limit(5)
    ).all()

    return {
        "user": {
            "username": current_user.username,
            "display_name": current_user.name,
            "bio": current_user.bio,
            "avatar_url": None,
            "goal_type": current_user.goal_type,
            "study_goal": current_user.study_goal,
            "target_date": current_user.target_date,
            "daily_goal_mins": current_user.daily_goal_mins,
        },
        "has_pet": pet is not None,
        "pet": pet_data,
        "streak": {
            "days": streak["days"],
            "active_today": streak["active_today"],
            "longest_ever": streak["longest"],
        },
        "stats": {
            "quests_won": quests_won,
            "cards_mastered": mastered_cards,
            "total_study_hours": round((total_study_minutes or 0) / 60.0, 1),
        },
        "recent_activity": [
            {
                "subject": s.subject,
                "mode": s.mode,
                "duration": s.duration_mins,
                "date": s.date,
            }
            for s in recent_activity_query
        ],
    }

@router.patch("/profile", status_code=status.HTTP_200_OK)
def update_profile(
    data: ProfileUpdate,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Update display name, username, bio, and study goal"""

    if data.username:
        existing = session.exec(
            select(User).where(User.username == data.username)
        ).first()
        if existing and existing.id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username taken",
            )
        current_user.username = data.username

    if data.display_name:
        current_user.name = data.display_name
    if data.bio is not None:
        current_user.bio = data.bio

    # Update the study_goal if provided by the Quiz frontend.
    # (Previously there was a fallback to `bio` if the column was
    # missing — that silently polluted the bio field. If the column
    # ever goes away, we WANT this to blow up loudly instead.)
    if data.study_goal is not None:
        current_user.study_goal = data.study_goal

    session.add(current_user)
    session.commit()
    session.refresh(current_user)

    return {"message": "Profile updated successfully"}

@router.post("/avatar")
def upload_avatar(
    image: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Upload new profile photo"""
    # TODO: Connect AWS S3 / Cloudinary / Firebase Storage
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Cloud storage integration required",
    )

# ─────────────────────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────────────────────

@router.get("/analytics", status_code=status.HTTP_200_OK)
def get_analytics(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    x_timezone: str = Header("UTC"),
):
    """Weekly chart + streak calendar (user's local days)."""
    today = get_local_now(x_timezone).date()

    # 1. Weekly Chart (Last 7 Days)
    last_7_dates = [
        (today - timedelta(days=i)).isoformat() for i in range(6, -1, -1)
    ]
    # Minutes come from DailyActivity, which now records plan sessions,
    # flashcards AND Feynman time (award_xp), not just plan sessions.
    week_activities = session.exec(
        select(DailyActivity).where(
            DailyActivity.user_id == current_user.id,
            DailyActivity.date.in_(last_7_dates),
        )
    ).all()
    week_minutes = {a.date: a.study_time_mins or 0 for a in week_activities}

    weekly_chart = []
    for i in range(6, -1, -1):
        target_date = today - timedelta(days=i)
        target_date_str = target_date.isoformat()
        day_mins = week_minutes.get(target_date_str, 0)
        weekly_chart.append({
            "day": target_date.strftime("%a"),
            "date": target_date_str,
            "minutes_studied": day_mins,
        })

    # 2. Streak Calendar (Last 30 Days)
    last_30_dates = [
        (today - timedelta(days=i)).isoformat() for i in range(29, -1, -1)
    ]
    month_activities = session.exec(
        select(DailyActivity).where(
            DailyActivity.user_id == current_user.id,
            DailyActivity.date.in_(last_30_dates),
        )
    ).all()

    activity_map = {act.date: act for act in month_activities}
    streak_calendar = []

    for i in range(29, -1, -1):
        target_date = today - timedelta(days=i)
        target_date_str = target_date.isoformat()
        act = activity_map.get(target_date_str)

        streak_calendar.append({
            "date": target_date_str,
            "studied": bool(act and act.xp_earned > 0),
            "session_count": 1 if (act and act.xp_earned > 0) else 0,
            "total_minutes": (act.study_time_mins or 0) if act else 0,
        })

    return {
        "heatmap": streak_calendar,
        "weekly_chart": weekly_chart,
        "streak_calendar": streak_calendar,
    }

# ─────────────────────────────────────────────────────────────
# REMINDERS
# ─────────────────────────────────────────────────────────────

@router.get("/reminders", status_code=status.HTTP_200_OK)
def get_reminders(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    statement = select(Reminder).where(Reminder.user_id == current_user.id)
    reminders = session.exec(statement).all()
    return {"reminders": reminders}

@router.post("/reminders", status_code=status.HTTP_200_OK)
def create_reminder(
    data: ReminderCreate,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    import json
    new_reminder = Reminder(
        user_id=current_user.id,
        type=data.type,
        title=data.title,
        schedule=data.schedule,
        time=data.time,
        days_of_week=json.dumps(data.days_of_week) if data.days_of_week else None,
        enabled=data.enabled,
        color=data.color,
    )
    session.add(new_reminder)
    session.commit()
    session.refresh(new_reminder)
    return {"reminder_id": new_reminder.id}

@router.patch("/reminders/{reminder_id}", status_code=status.HTTP_200_OK)
def update_reminder(
    reminder_id: int,
    data: ReminderUpdate,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    reminder = session.exec(
        select(Reminder).where(
            Reminder.id == reminder_id,
            Reminder.user_id == current_user.id,
        )
    ).first()
    if not reminder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Reminder not found",
        )

    import json
    update_data = data.dict(exclude_unset=True)
    if "days_of_week" in update_data and update_data["days_of_week"] is not None:
        update_data["days_of_week"] = json.dumps(update_data["days_of_week"])

    for key, value in update_data.items():
        setattr(reminder, key, value)

    session.add(reminder)
    session.commit()
    return {"message": "Reminder updated"}

@router.delete("/reminders/{reminder_id}", status_code=status.HTTP_200_OK)
def delete_reminder(
    reminder_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    reminder = session.exec(
        select(Reminder).where(
            Reminder.id == reminder_id,
            Reminder.user_id == current_user.id,
        )
    ).first()
    if not reminder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Reminder not found",
        )

    session.delete(reminder)
    session.commit()
    return {"deleted": True}

# ─────────────────────────────────────────────────────────────
# SETTINGS & SUPPORT
# ─────────────────────────────────────────────────────────────

@router.patch("/settings", status_code=status.HTTP_200_OK)
def update_settings(
    data: SettingsUpdate,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Update notification prefs, SRS intensity, daily goal, privacy"""
    update_data = data.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(current_user, key, value)

    session.add(current_user)
    session.commit()
    return {"message": "Settings updated successfully"}

@router.post("/feedback", status_code=status.HTTP_200_OK)
def submit_feedback(
    data: FeedbackRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Submit user feedback and save it to the database."""

    new_feedback = Feedback(
        user_id=current_user.id,
        message=data.message,
    )

    session.add(new_feedback)
    session.commit()

    return {"message": "Thank you! Your feedback has been safely recorded."}

# ─────────────────────────────────────────────────────────────
# DELETE ACCOUNT
# ─────────────────────────────────────────────────────────────
#
# Uses POST (not DELETE) because some edge proxies (including
# Render's, in some configurations) strip the request body from
# DELETE calls before they reach the app. That made the
# confirmation string impossible to deliver reliably in
# production, resulting in 422 "field required: confirmation"
# even when the user typed the right thing.
#
# POST bodies are preserved universally, so the confirmation
# always reaches us.

@router.post("/delete", status_code=status.HTTP_200_OK)
def delete_account(
    data: AccountDeleteRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """
    Delete account and all associated user data safely.
    """
    if data.confirmation.strip() != "DELETE MY ACCOUNT":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid confirmation string",
        )

    try:
        # 1. EXPLICITLY DELETE RELATED DATA FIRST
        # This prevents Foreign Key IntegrityErrors if your database 
        # migrations don't have perfect ON DELETE CASCADE configurations.
        user_id = current_user.id

        # Delete Pet
        pets = session.exec(select(Pet).where(Pet.user_id == user_id)).all()
        for pet in pets: session.delete(pet)

        # Delete Daily Activities
        activities = session.exec(select(DailyActivity).where(DailyActivity.user_id == user_id)).all()
        for act in activities: session.delete(act)

        # Delete Study Sessions
        study_sessions = session.exec(select(StudySession).where(StudySession.user_id == user_id)).all()
        for s in study_sessions: session.delete(s)

        # Delete Quests (even if unused in V1, good to clear if they exist)
        quests = session.exec(select(Quest).where(Quest.user_id == user_id)).all()
        for q in quests: session.delete(q)

        # Delete Reminders
        reminders = session.exec(select(Reminder).where(Reminder.user_id == user_id)).all()
        for r in reminders: session.delete(r)

        # Feynman sessions FIRST — their card_id is NOT NULL, so deleting
        # cards before them makes the ORM try to null it and crash.
        feynman = session.exec(
            select(FeynmanSession).where(FeynmanSession.user_id == user_id)
        ).all()
        for fs in feynman: session.delete(fs)
        session.flush()

        # Delete StudySets and their Flashcards.
        # FIX: was `Flashcard.set_id`, which doesn't exist — every account
        # deletion crashed with a 500.
        study_sets = session.exec(select(StudySet).where(StudySet.user_id == user_id)).all()
        for s_set in study_sets:
            for fs in session.exec(
                select(FeynmanSession).where(FeynmanSession.study_set_id == s_set.id)
            ).all():
                session.delete(fs)
            session.flush()
            cards = session.exec(
                select(Flashcard).where(Flashcard.study_set_id == s_set.id)
            ).all()
            for card in cards: session.delete(card)
            session.flush()
            session.delete(s_set)
        session.flush()

        # Delete Feedback
        feedbacks = session.exec(select(Feedback).where(Feedback.user_id == user_id)).all()
        for f in feedbacks: session.delete(f)

        # 2. FINALLY, DELETE THE USER
        session.delete(current_user)
        session.commit()

        return {
            "message": "Account and all associated data successfully deleted",
        }

    except Exception as e:
        session.rollback()
        print(f"Failed to delete account: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting account data. Please contact support.",
        )