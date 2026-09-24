from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlmodel import Session, select
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date, datetime, timedelta

# Import your existing project dependencies
from database import get_session
from security import get_current_user
from models import User, Pet, DailyActivity, StudySession, Quest, Reminder, Flashcard, StudySet, Feedback

# NEW: Import the evolution math
from utils import get_pet_evolution_data

router = APIRouter()

# ─────────────────────────────────────────────────────────────
# HELPER: CALL THIS WHENEVER A USER EARNS XP
# ─────────────────────────────────────────────────────────────
def update_user_streak(user: User, session: Session):
    """Updates the user's streak in O(1) time. Call this when an activity is completed."""
    today = datetime.now().date()
    
    if user.last_active_date == today:
        return # Already studied today
        
    yesterday = today - timedelta(days=1)
    if user.last_active_date == yesterday:
        user.current_streak += 1
    else:
        user.current_streak = 1 # Streak reset/started

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
    study_goal: Optional[str] = None # FIXED: Added to support the Quiz onboarding screen

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
    session: Session = Depends(get_session)
):
    """Returns full user profile + pet + streak + stats"""
    
    # 1. Pet Data
    pet = session.exec(select(Pet).where(Pet.user_id == current_user.id)).first()
    
    pet_type = pet.pet_type if pet else "nova"
    pet_level = pet.level if pet else 1
    pet_xp = pet.xp if pet else 0
    pet_name = pet.pet_name if pet else "Nova"
    pet_mood = getattr(pet, 'mood', "happy") if pet else "happy"
    
    pet_data = {
        "name": pet_name,
        "type": pet_type,
        "level": pet_level,
        "xp": pet_xp,
        "mood": pet_mood,
        "xp_history": [0,0,0,0,0,0,0] 
    }
    
    # NEW: Merge the dynamic UI fields
    pet_data.update(get_pet_evolution_data(pet_type, pet_level, pet_xp))

    # 2. Optimized Streak Data (O(1) lookup)
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    
    is_streak_alive = current_user.last_active_date in (today, yesterday)
    display_current_streak = current_user.current_streak if is_streak_alive else 0

    # 3. Stats Data 
    quests_won = session.exec(
        select(Quest).where(Quest.user_id == current_user.id, Quest.progress >= Quest.target)
    ).all()
    
    completed_sessions = session.exec(
        select(StudySession).where(StudySession.user_id == current_user.id, StudySession.completed == True)
    ).all()
    total_study_minutes = sum(s.duration_mins for s in completed_sessions)

    # Calculate real mastered cards (cards that are NOT marked as weak)
    mastered_cards = session.exec(
        select(Flashcard)
        .join(StudySet)
        .where(StudySet.user_id == current_user.id, Flashcard.is_weak == False)
    ).all()

    # Fetch actual recent activity (Last 5 completed sessions)
    recent_activity_query = session.exec(
        select(StudySession)
        .where(StudySession.user_id == current_user.id, StudySession.completed == True)
        .order_by(StudySession.date.desc())
        .limit(5)
    ).all()
    
    recent_activity_list = [
        {"subject": s.subject, "mode": s.mode, "duration": s.duration_mins, "date": s.date} 
        for s in recent_activity_query
    ]

    return {
        "user": {
            "username": current_user.username,
            "display_name": current_user.name,
            "bio": current_user.bio,
            "avatar_url": None 
        },
        "pet": pet_data,
        "streak": {
            "days": display_current_streak,
            "longest_ever": current_user.longest_streak
        },
        "stats": {
            "quests_won": len(quests_won),
            "cards_mastered": len(mastered_cards), 
            "total_study_hours": round(total_study_minutes / 60.0, 1)
        },
        "recent_activity": recent_activity_list
    }

@router.patch("/profile", status_code=status.HTTP_200_OK)
def update_profile(
    data: ProfileUpdate, 
    current_user: User = Depends(get_current_user), 
    session: Session = Depends(get_session)
):
    """Update display name, username, bio, and study goal"""
    
    if data.username:
        existing = session.exec(select(User).where(User.username == data.username)).first()
        if existing and existing.id != current_user.id:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username taken")
        current_user.username = data.username

    if data.display_name:
        current_user.name = data.display_name
    if data.bio is not None:
        current_user.bio = data.bio
        
    # FIXED: Check and update the study_goal if provided by the Quiz frontend
    if data.study_goal is not None:
        if hasattr(current_user, 'study_goal'):
            current_user.study_goal = data.study_goal
        else:
            current_user.bio = data.study_goal

    session.add(current_user)
    session.commit()
    session.refresh(current_user)
    
    return {"message": "Profile updated successfully"}

@router.post("/avatar")
def upload_avatar(
    image: UploadFile = File(...), 
    current_user: User = Depends(get_current_user), 
    session: Session = Depends(get_session)
):
    """Upload new profile photo"""
    # TODO: Connect AWS S3 / Cloudinary / Firebase Storage
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Cloud storage integration required")

# ─────────────────────────────────────────────────────────────
# ANALYTICS 
# ─────────────────────────────────────────────────────────────

@router.get("/analytics", status_code=status.HTTP_200_OK)
def get_analytics(
    current_user: User = Depends(get_current_user), 
    session: Session = Depends(get_session)
):
    """Weekly chart + streak calendar"""
    today = datetime.now().date()
    
    # 1. Weekly Chart (Last 7 Days)
    last_7_dates = [(today - timedelta(days=i)).isoformat() for i in range(6, -1, -1)]
    recent_sessions = session.exec(
        select(StudySession)
        .where(StudySession.user_id == current_user.id, StudySession.date.in_(last_7_dates), StudySession.completed == True)
    ).all()
    
    weekly_chart = []
    for i in range(6, -1, -1):
        target_date = today - timedelta(days=i)
        target_date_str = target_date.isoformat()
        day_mins = sum(s.duration_mins for s in recent_sessions if s.date == target_date_str)
        weekly_chart.append({
            "day": target_date.strftime("%a"),
            "date": target_date_str,
            "minutes_studied": day_mins
        })

    # 2. Streak Calendar (Last 30 Days)
    last_30_dates = [(today - timedelta(days=i)).isoformat() for i in range(29, -1, -1)]
    month_activities = session.exec(
        select(DailyActivity).where(DailyActivity.user_id == current_user.id, DailyActivity.date.in_(last_30_dates))
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
            "total_minutes": 0 
        })

    return {
        "heatmap": streak_calendar,
        "weekly_chart": weekly_chart,
        "streak_calendar": streak_calendar
    }

# ─────────────────────────────────────────────────────────────
# REMINDERS 
# ─────────────────────────────────────────────────────────────

@router.get("/reminders", status_code=status.HTTP_200_OK)
def get_reminders(
    current_user: User = Depends(get_current_user), 
    session: Session = Depends(get_session)
):
    statement = select(Reminder).where(Reminder.user_id == current_user.id)
    reminders = session.exec(statement).all()
    return {"reminders": reminders}

@router.post("/reminders", status_code=status.HTTP_200_OK)
def create_reminder(
    data: ReminderCreate, 
    current_user: User = Depends(get_current_user), 
    session: Session = Depends(get_session)
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
        color=data.color 
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
    session: Session = Depends(get_session)
):
    reminder = session.exec(select(Reminder).where(Reminder.id == reminder_id, Reminder.user_id == current_user.id)).first()
    if not reminder:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Reminder not found")

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
    session: Session = Depends(get_session)
):
    reminder = session.exec(select(Reminder).where(Reminder.id == reminder_id, Reminder.user_id == current_user.id)).first()
    if not reminder:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Reminder not found")
        
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
    session: Session = Depends(get_session)
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
    session: Session = Depends(get_session) 
):
    """Submit user feedback and save it to the database."""
    
    new_feedback = Feedback(
        user_id=current_user.id,
        message=data.message
    )
    
    session.add(new_feedback)
    session.commit()
    
    return {"message": "Thank you! Your feedback has been safely recorded."}

@router.delete("/", status_code=status.HTTP_200_OK)
def delete_account(
    data: AccountDeleteRequest, 
    current_user: User = Depends(get_current_user), 
    session: Session = Depends(get_session)
):
    """
    Delete account and all associated user data.
    Relies on ON DELETE CASCADE set on the database tables!
    """
    if data.confirmation != "DELETE MY ACCOUNT":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid confirmation string")
    
    try:
        session.delete(current_user)
        session.commit()
        return {"message": "Account and all associated data successfully deleted"}
        
    except Exception as e:
        session.rollback()
        print(f"Failed to delete account: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Error deleting account. Did you remember to apply the ON DELETE CASCADE migration to your database?"
        )