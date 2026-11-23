from sqlmodel import Session, select
from datetime import datetime, timezone
from models import User, UserTrophy, Collection, FeynmanSession, TrophyDefinition, StudySession

def utc_now():
    """Helper to generate the exact timestamp when the badge is awarded."""
    return datetime.now(timezone.utc)

def check_and_award_trophies(user: User, db: Session):
    """
    Dynamically evaluates the user's current stats against the TrophyDefinition table.
    Awards any new trophies they have earned and records the exact date.
    """
    newly_earned = []
    
    # 1. Fetch trophies the user already has so we don't award them twice
    existing_trophies = db.exec(
        select(UserTrophy.trophy_id).where(UserTrophy.user_id == user.id)
    ).all()
    earned_set = set(existing_trophies)

    # 2. Fetch all active trophy rules from the database
    all_trophies = db.exec(select(TrophyDefinition)).all()

    # 3. Evaluate rules dynamically
    for trophy in all_trophies:
        if trophy.id in earned_set:
            continue # Already earned it
            
        earned = False
        
        # 🏆 RULE: FIRST SESSION (The Journey Begins)
        if trophy.condition_type == "first_session":
            session_count = len(db.exec(
                select(StudySession.id).where(
                    StudySession.user_id == user.id, 
                    StudySession.completed == True
                )
            ).all())
            if session_count >= trophy.condition_value:
                earned = True

        # 🏆 RULE: STREAKS
        elif trophy.condition_type == "streak":
            if user.current_streak >= trophy.condition_value:
                earned = True
                
        # 🏆 RULE: COLLECTIONS
        elif trophy.condition_type == "collection":
            collection_count = len(db.exec(
                select(Collection.id).where(Collection.user_id == user.id)
            ).all())
            if collection_count >= trophy.condition_value:
                earned = True
                
        # 🏆 RULE: FEYNMAN
        elif trophy.condition_type == "feynman":
            feynman_count = len(db.exec(
                select(FeynmanSession.id).where(
                    FeynmanSession.user_id == user.id, 
                    FeynmanSession.is_complete == True
                )
            ).all())
            if feynman_count >= trophy.condition_value:
                earned = True

        # Award it if they passed the condition!
        if earned:
            now = utc_now()
            new_record = UserTrophy(user_id=user.id, trophy_id=trophy.id, earned_at=now)
            db.add(new_record)
            
            # Format exactly how the frontend expects it, including the date!
            newly_earned.append({
                "id": trophy.id,
                "title": trophy.title,
                "description": trophy.description,
                "icon": trophy.icon,
                "is_coop": trophy.is_coop,
                "earned_at": now.isoformat() # Returns e.g. "2026-09-24T12:11:59Z"
            })

    if newly_earned:
        db.commit()
        
    return newly_earned