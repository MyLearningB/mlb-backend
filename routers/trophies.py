from fastapi import APIRouter, Depends, status
from sqlmodel import Session, select
from database import get_session
from security import get_current_user
from models import User, UserTrophy, TrophyDefinition
from services.gamification import check_and_award_trophies

router = APIRouter(tags=["Trophies"])

@router.get("/users/me/trophies", status_code=status.HTTP_200_OK)
def get_my_trophies(
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_session)
):
    # 1. Run a quick background check to see if they earned anything new just now
    check_and_award_trophies(current_user, db)

    # 2. Fetch all their earned trophy records
    earned_records = db.exec(
        select(UserTrophy)
        .where(UserTrophy.user_id == current_user.id)
        .order_by(UserTrophy.earned_at.desc())
    ).all()

    # Map for easy lookup: { "streak_3": "2024-03-12T10:00:00" }
    earned_map = {record.trophy_id: record.earned_at for record in earned_records}

    # 3. Fetch all possible trophies from the database
    all_trophies = db.exec(select(TrophyDefinition)).all()

    solo_earned = []
    solo_locked = []
    coop_earned = []
    coop_locked = []
    
    # 4. Group them for the Flutter UI
    for t in all_trophies:
        t_dict = {
            "id": t.id,
            "title": t.title,
            "description": t.description,
            "icon": t.icon,
        }
        
        is_earned = t.id in earned_map
        
        if is_earned:
            t_dict["earned_at"] = earned_map[t.id].isoformat()
            if t.is_coop:
                coop_earned.append(t_dict)
            else:
                solo_earned.append(t_dict)
        else:
            t_dict["icon"] = "🔒" # Hidden icon for unearned trophies
            if t.is_coop:
                coop_locked.append(t_dict)
            else:
                solo_locked.append(t_dict)

    return {
        "earned": solo_earned,
        "locked": solo_locked,
        "coop_earned": coop_earned,
        "coop_locked": coop_locked,
        "total_earned": len(solo_earned) + len(coop_earned),
        "total_available": len(all_trophies)
    }