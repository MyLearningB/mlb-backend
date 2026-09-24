import string
import random
from fastapi import APIRouter, Depends, status, Query, HTTPException
from sqlmodel import Session, select, func, or_
from typing import Optional
from pydantic import BaseModel, Field

from database import get_session
from security import get_current_user
from models import User, Collection, StudyGroup, GroupMember, CoopQuest, Relic, UserRelic

router = APIRouter(prefix="/community", tags=["Community Tab"])

# ==========================================
# PYDANTIC SCHEMAS
# ==========================================
class GroupCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=50)

class GroupJoin(BaseModel):
    invite_code: str = Field(..., min_length=6, max_length=6)

# ==========================================
# 1. COLLECTIONS DISCOVERY
# ==========================================
@router.get("/collections", status_code=status.HTTP_200_OK)
def get_discover_collections(
    search: Optional[str] = None,
    filter: str = Query("All"), 
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=50),
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_session)
):
    query = select(Collection).where(
        or_(Collection.visibility == "public", Collection.visibility == "private")
    )

    if search:
        query = query.where(
            or_(Collection.title.icontains(search), Collection.subject.icontains(search))
        )

    if filter and filter.lower() != "all":
        if filter.lower() == "private":
            query = query.where(Collection.visibility == "private")
        else:
            query = query.where(Collection.subject.ilike(filter))

    query = query.order_by(Collection.save_count.desc(), Collection.created_at.desc())

    offset = (page - 1) * limit
    collections = db.exec(query.offset(offset).limit(limit)).all()
    
    total_count = db.exec(
        select(func.count(Collection.id)).where(
            or_(Collection.visibility == "public", Collection.visibility == "private")
        )
    ).one()

    return {
        "collections": collections,
        "total_count": total_count,
        "has_more": total_count > (offset + limit)
    }

# ==========================================
# 2. STUDY GROUPS & CO-OP
# ==========================================
def generate_invite_code():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

@router.post("/groups", status_code=status.HTTP_201_CREATED)
def create_group(
    payload: GroupCreate, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_session)
):
    """Creates a new study group and auto-assigns the creator as a member atomically."""
    new_group = StudyGroup(name=payload.name, invite_code=generate_invite_code())
    db.add(new_group)
    
    # FIXED: Flush secures the new_group.id without committing the transaction prematurely
    db.flush() 
    
    member = GroupMember(group_id=new_group.id, user_id=current_user.id)
    db.add(member)
    
    # FIXED: A single atomic commit guarantees both records succeed, or neither do
    db.commit()
    db.refresh(new_group)
    
    return new_group

@router.post("/groups/join", status_code=status.HTTP_200_OK)
def join_group(
    payload: GroupJoin, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_session)
):
    """Allows a user to join an existing group using the 6-character code."""
    group = db.exec(select(StudyGroup).where(StudyGroup.invite_code == payload.invite_code.upper())).first()
    
    if not group:
        raise HTTPException(status_code=404, detail="Invalid invite code")
        
    existing = db.exec(select(GroupMember).where(GroupMember.group_id == group.id, GroupMember.user_id == current_user.id)).first()
    if existing:
        return {"message": "You are already a member of this group"}
        
    db.add(GroupMember(group_id=group.id, user_id=current_user.id))
    db.commit()
    
    return {"message": f"Successfully joined {group.name}!"}

@router.get("/quests/active", status_code=status.HTTP_200_OK)
def get_active_quests(current_user: User = Depends(get_current_user), db: Session = Depends(get_session)):
    """Returns the formatted JSON exactly as expected by the Flutter Dashboard using a single fast query."""
    
    # FIXED: Flattens the N+1 loop into a single optimized JOIN and GROUP BY
    statement = (
        select(CoopQuest, func.count(GroupMember.user_id).label("member_count"))
        .join(GroupMember, GroupMember.group_id == CoopQuest.group_id)
        .where(GroupMember.user_id == current_user.id)
        .where(CoopQuest.is_completed == False)
        .group_by(CoopQuest.id)
    )
    
    results = db.exec(statement).all()
    
    return {"quests": [
        {
            "id": str(quest.id),
            "title": quest.title,
            "progress": quest.current_xp,
            "target": quest.target_xp,
            "type": "coop",
            "membersCount": count
        } for quest, count in results
    ]}

# ==========================================
# 3. XP HOOK (Import this into Flashcard/Feynman routers)
# ==========================================
def contribute_to_group_quests(user_id: int, xp_amount: int, db: Session):
    """
    Hook to pool XP. 
    Optimized to fetch all relevant quests in one query and commit once.
    """
    memberships = db.exec(select(GroupMember).where(GroupMember.user_id == user_id)).all()
    group_ids = [m.group_id for m in memberships]
    
    if not group_ids:
        return
        
    # FIXED: Fetches all active quests for all of the user's groups in a single query
    active_quests = db.exec(
        select(CoopQuest)
        .where(CoopQuest.group_id.in_(group_ids), CoopQuest.is_completed == False)
    ).all()
    
    for quest in active_quests:
        quest.current_xp += xp_amount
        
        if quest.current_xp >= quest.target_xp:
            quest.is_completed = True
            # --- Relic Logic Injection Point ---
            
        db.add(quest)
            
    # FIXED: Single atomic commit for all quest updates
    db.commit()