import string
import random
import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlmodel import Session, select, func, or_
from pydantic import BaseModel, Field

# Database, Models, and Auth
from database import get_session
from security import get_current_user
from models import (
    User, Collection, CollectionItem, CollectionAccess, CollectionRequest,
    Note, StudySet, Canvas, CanvasConnection, CanvasNode, Flashcard,
    StudyGroup, GroupMember, CoopQuest, Relic, UserRelic
)
from services.notifications import send_collection_notification
from schemas import CollectionUpdate, ItemReorderRequest

router = APIRouter(tags=["Collections & Community Section"])

# ==========================================
# PYDANTIC SCHEMAS
# ==========================================
class CollectionCreate(BaseModel):
    title: str
    subject: str
    visibility: str # "private", "shared", "public"
    description: Optional[str] = None
    cover_emoji: Optional[str] = None
    item_ids: List[str] = [] 
    item_types: List[str] = [] 

class ItemAddRequest(BaseModel):
    item_id: str
    item_type: str 

class InviteRequest(BaseModel):
    emails: List[str]

class AccessRequestSubmit(BaseModel):
    message: Optional[str] = None

class RequestStatusUpdate(BaseModel):
    status: str  # "approved" or "denied"

class SaveItemRequest(BaseModel):
    item_id: str
    item_type: str

class ReportRequest(BaseModel):
    reason: str

class GroupCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=50)

class GroupJoin(BaseModel):
    invite_code: str = Field(..., min_length=6, max_length=6)


# ==========================================
# 1. MY COLLECTIONS LIBRARY
# ==========================================
@router.get("/users/me/collections", status_code=status.HTTP_200_OK)
def get_my_collections(
    search: Optional[str] = None,
    sort: str = Query("recent"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    query = select(Collection).where(Collection.user_id == current_user.id)
    if search:
        query = query.where(Collection.title.icontains(search))
    query = query.order_by(Collection.updated_at.desc())
    
    offset = (page - 1) * limit
    collections = db.exec(query.offset(offset).limit(limit)).all()
    total_count = db.exec(select(func.count(Collection.id)).where(Collection.user_id == current_user.id)).one()

    return {
        "collections": collections,
        "total_count": total_count,
        "has_more": total_count > (offset + limit)
    }


# ==========================================
# 2. CREATE & EDIT COLLECTION
# ==========================================
@router.post("/collections", status_code=status.HTTP_201_CREATED)
def create_collection(
    payload: CollectionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    new_collection = Collection(
        user_id=current_user.id, title=payload.title, subject=payload.subject,
        visibility=payload.visibility, description=payload.description, cover_emoji=payload.cover_emoji
    )
    db.add(new_collection)
    db.flush() # Replaced commit with flush to secure the ID safely

    if payload.item_ids and len(payload.item_ids) == len(payload.item_types):
        for i in range(len(payload.item_ids)):
            new_item = CollectionItem(
                collection_id=new_collection.id, item_type=payload.item_types[i],
                item_id=payload.item_ids[i], position=i
            )
            db.add(new_item)
            
    db.commit() # Single atomic commit
    db.refresh(new_collection)

    return {"collection_id": new_collection.id, "share_token": new_collection.share_token, "message": "Collection created successfully"}

@router.get("/collections/{collection_id}", status_code=status.HTTP_200_OK)
def get_collection_detail(
    collection_id: int, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_session)
):
    statement = (
        select(Collection, User)
        .join(User, Collection.user_id == User.id)
        .where(Collection.id == collection_id)
    )
    result = db.exec(statement).first()

    if not result: 
        raise HTTPException(status_code=404, detail="Collection not found")
        
    collection, owner = result

    if collection.user_id != current_user.id and collection.visibility != "public":
        access = db.exec(select(CollectionAccess).where(
            CollectionAccess.collection_id == collection.id, 
            CollectionAccess.user_id == current_user.id
        )).first()
        if not access: 
            raise HTTPException(status_code=403, detail="You do not have access to this collection")

    item_mappings = db.exec(
        select(CollectionItem)
        .where(CollectionItem.collection_id == collection.id)
        .order_by(CollectionItem.position)
    ).all()
    
    resolved_items = []
    for mapping in item_mappings:
        if mapping.item_type == "note":
            note = db.get(Note, int(mapping.item_id))
            if note: resolved_items.append({"id": str(note.id), "type": "note", "title": note.title, "subject": note.subject})
        elif mapping.item_type == "set":
            study_set = db.get(StudySet, int(mapping.item_id))
            if study_set: resolved_items.append({"id": str(study_set.id), "type": "set", "title": study_set.title, "card_count": study_set.card_count})
        elif mapping.item_type == "canvas":
            canvas = db.get(Canvas, uuid.UUID(mapping.item_id))
            if canvas: resolved_items.append({"id": str(canvas.id), "type": "canvas", "title": canvas.name, "node_count": canvas.node_count})

    response_data = collection.model_dump()
    response_data["owner_username"] = owner.name

    return {"collection": response_data, "items": resolved_items}

@router.patch("/collections/{collection_id}", status_code=status.HTTP_200_OK)
def update_collection_settings(
    collection_id: int, payload: CollectionUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_session)
):
    collection = db.get(Collection, collection_id)
    if not collection or collection.user_id != current_user.id: raise HTTPException(status_code=404)

    update_data = payload.model_dump(exclude_unset=True)
    for key, value in update_data.items(): setattr(collection, key, value)
        
    db.add(collection)
    db.commit()
    db.refresh(collection)
    return {"message": "Collection updated successfully"}

@router.delete("/collections/{collection_id}", status_code=status.HTTP_200_OK)
def delete_collection(
    collection_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_session)
):
    collection = db.get(Collection, collection_id)
    if not collection or collection.user_id != current_user.id: raise HTTPException(status_code=404)

    items = db.exec(select(CollectionItem).where(CollectionItem.collection_id == collection.id)).all()
    for item in items: db.delete(item)

    access_rows = db.exec(select(CollectionAccess).where(CollectionAccess.collection_id == collection.id)).all()
    for row in access_rows: db.delete(row)
        
    req_rows = db.exec(select(CollectionRequest).where(CollectionRequest.collection_id == collection.id)).all()
    for row in req_rows: db.delete(row)

    db.delete(collection)
    db.commit()
    return {"message": "Collection deleted successfully. Your items are safe."}


# ==========================================
# 3. MANAGE ITEMS IN A COLLECTION
# ==========================================
@router.post("/collections/{collection_id}/items", status_code=status.HTTP_200_OK)
def add_item_to_collection(
    collection_id: int, payload: ItemAddRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_session)
):
    collection = db.get(Collection, collection_id)
    if not collection or collection.user_id != current_user.id: raise HTTPException(status_code=404)

    existing_item = db.exec(select(CollectionItem).where(
        CollectionItem.collection_id == collection.id, CollectionItem.item_id == payload.item_id, CollectionItem.item_type == payload.item_type
    )).first()

    if existing_item: return {"message": "Item already in collection"}

    max_pos = db.exec(select(func.max(CollectionItem.position)).where(CollectionItem.collection_id == collection.id)).one()
    next_pos = (max_pos or 0) + 1

    new_item = CollectionItem(collection_id=collection.id, item_type=payload.item_type, item_id=payload.item_id, position=next_pos)
    db.add(new_item)
    db.commit()
    return {"message": "Item added successfully"}

@router.patch("/collections/{collection_id}/items", status_code=status.HTTP_200_OK)
def reorder_collection_items(
    collection_id: int, payload: ItemReorderRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_session)
):
    collection = db.get(Collection, collection_id)
    if not collection or collection.user_id != current_user.id: raise HTTPException(status_code=404)

    for reorder_item in payload.positions:
        mapping = db.exec(select(CollectionItem).where(
            CollectionItem.collection_id == collection.id, CollectionItem.item_id == reorder_item.item_id
        )).first()
        if mapping:
            mapping.position = reorder_item.position
            db.add(mapping)
            
    db.commit() # Atomic save at the end
    return {"message": "Items reordered successfully"}


# ==========================================
# 4. SHARE SETTINGS: LINKS & INVITES
# ==========================================
@router.post("/collections/{collection_id}/share-token/regenerate", status_code=status.HTTP_200_OK)
def regenerate_share_token(
    collection_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_session)
):
    collection = db.get(Collection, collection_id)
    if not collection or collection.user_id != current_user.id: raise HTTPException(status_code=404)

    collection.share_token = uuid.uuid4().hex[:12]
    db.add(collection)
    db.commit()
    db.refresh(collection)
    return {"new_share_token": collection.share_token}

@router.post("/collections/{collection_id}/invites", status_code=status.HTTP_200_OK)
def invite_users_by_email(
    collection_id: int, payload: InviteRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_session)
):
    collection = db.get(Collection, collection_id)
    if not collection or collection.user_id != current_user.id: raise HTTPException(status_code=404)

    invited, already_had_access, not_found = [], [], []

    for email in payload.emails:
        user = db.exec(select(User).where(User.email == email.lower())).first()
        if not user:
            not_found.append(email)
            continue
            
        existing_access = db.exec(select(CollectionAccess).where(CollectionAccess.collection_id == collection.id, CollectionAccess.user_id == user.id)).first()

        if existing_access:
            already_had_access.append(email)
        else:
            new_access = CollectionAccess(collection_id=collection.id, user_id=user.id)
            db.add(new_access)
            invited.append(email)
            
            send_collection_notification(
                db=db, user_id=user.id, title="Collection Invitation",
                body=f"@{current_user.name} invited you to '{collection.title}'.", deep_link=f"/collections/{collection.id}/view"
            )

    db.commit()
    return {"invited": invited, "already_had_access": already_had_access, "not_found": not_found}

@router.delete("/collections/{collection_id}/access/{user_id}", status_code=status.HTTP_200_OK)
def revoke_access(
    collection_id: int, user_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_session)
):
    collection = db.get(Collection, collection_id)
    if not collection or collection.user_id != current_user.id: raise HTTPException(status_code=404)

    access_record = db.exec(select(CollectionAccess).where(CollectionAccess.collection_id == collection.id, CollectionAccess.user_id == user_id)).first()

    if access_record:
        db.delete(access_record)
        db.commit()
        send_collection_notification(
            db=db, user_id=user_id, title="Access Removed",
            body=f"Your access to '{collection.title}' has been removed by @{current_user.name}.", deep_link=None
        )
    return {"message": "Access revoked successfully"}

# ==========================================
# 5. ACCESS REQUESTS 
# ==========================================
@router.post("/collections/{collection_id}/requests", status_code=status.HTTP_200_OK)
def submit_access_request(
    collection_id: int, payload: AccessRequestSubmit, current_user: User = Depends(get_current_user), db: Session = Depends(get_session)
):
    collection = db.get(Collection, collection_id)
    if not collection: raise HTTPException(status_code=404)

    existing_req = db.exec(select(CollectionRequest).where(
        CollectionRequest.collection_id == collection.id, CollectionRequest.user_id == current_user.id, CollectionRequest.status == "pending"
    )).first()

    if existing_req: 
        raise HTTPException(status_code=409, detail="You already have a pending request for this collection.")

    new_request = CollectionRequest(collection_id=collection.id, user_id=current_user.id, message=payload.message)
    db.add(new_request)
    db.commit()
    db.refresh(new_request) 

    send_collection_notification(
        db=db, user_id=collection.user_id, title="New Access Request",
        body=f"{current_user.name} wants access to '{collection.title}'", 
        deep_link=f"/collections/{collection.id}/share?request_id={new_request.id}"
    )
    return {"message": "Request sent successfully"}

@router.get("/collections/{collection_id}/requests", status_code=status.HTTP_200_OK)
def get_pending_requests(
    collection_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_session)
):
    collection = db.get(Collection, collection_id)
    if not collection or collection.user_id != current_user.id: raise HTTPException(status_code=404)

    requests = db.exec(select(CollectionRequest, User).join(User, CollectionRequest.user_id == User.id).where(
        CollectionRequest.collection_id == collection.id, CollectionRequest.status == "pending"
    )).all()

    formatted_requests = [{"request_id": req.id, "user_id": user.id, "username": user.name, "email": user.email, "message": req.message, "requested_at": req.requested_at} for req, user in requests]
    return {"requests": formatted_requests}

@router.patch("/collections/requests/{request_id}", status_code=status.HTTP_200_OK)
def update_request_status(
    request_id: int, payload: RequestStatusUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_session)
):
    coll_request = db.get(CollectionRequest, request_id)
    if not coll_request: raise HTTPException(status_code=404, detail="Request not found")

    collection = db.get(Collection, coll_request.collection_id)
    if not collection or collection.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="You do not have permission to manage this collection")

    if payload.status not in ["approved", "denied"]:
        raise HTTPException(status_code=400, detail="Status must be 'approved' or 'denied'")

    coll_request.status = payload.status
    db.add(coll_request)

    if payload.status == "approved":
        existing_access = db.exec(select(CollectionAccess).where(
            CollectionAccess.collection_id == collection.id, CollectionAccess.user_id == coll_request.user_id
        )).first()
        
        if not existing_access:
            access = CollectionAccess(collection_id=collection.id, user_id=coll_request.user_id)
            db.add(access)
            
            send_collection_notification(
                db=db, user_id=coll_request.user_id, title="Access Granted!",
                body=f"@{current_user.name} approved your access to '{collection.title}' — open it now!", 
                deep_link=f"/collections/{collection.id}/view"
            )
            
    elif payload.status == "denied":
        send_collection_notification(
            db=db, user_id=coll_request.user_id, title="Access Denied",
            body=f"Your request to '{collection.title}' was not approved.", deep_link=None
        )

    db.commit()
    return {"message": f"Request {payload.status} successfully", "request_id": coll_request.id, "status": coll_request.status}

# ==========================================
# 6. ITEM PICKER & VIEWER ACTIONS (DEEP CLONING)
# ==========================================
@router.post("/collections/{collection_id}/save", status_code=status.HTTP_200_OK)
def save_public_collection(
    collection_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_session)
):
    original_collection = db.get(Collection, collection_id)
    if not original_collection: raise HTTPException(status_code=404, detail="Collection not found")
    if original_collection.visibility != "public": raise HTTPException(status_code=403, detail="Only public collections can be saved")

    original_collection.save_count += 1
    db.add(original_collection)

    new_collection = Collection(
        user_id=current_user.id, title=f"{original_collection.title} (Copy)",
        description=original_collection.description, subject=original_collection.subject,
        cover_emoji=original_collection.cover_emoji, visibility="private",
        share_token=uuid.uuid4().hex[:12]
    )
    db.add(new_collection)
    db.flush() # Replaced commit with flush

    original_items = db.exec(select(CollectionItem).where(CollectionItem.collection_id == original_collection.id)).all()

    for item in original_items:
        new_item_id = None

        if item.item_type == "note":
            original_note = db.get(Note, int(item.item_id))
            if original_note:
                new_note = Note(
                    user_id=current_user.id, title=original_note.title, subject=original_note.subject, 
                    content_text=original_note.content_text, content_html=original_note.content_html, 
                    word_count=original_note.word_count, snippet=original_note.snippet
                )
                db.add(new_note)
                db.flush()
                new_item_id = str(new_note.id)

        elif item.item_type == "set":
            original_set = db.get(StudySet, int(item.item_id))
            if original_set:
                new_set = StudySet(
                    user_id=current_user.id, title=original_set.title, 
                    subject=original_set.subject, card_count=original_set.card_count
                )
                db.add(new_set)
                db.flush()
                
                original_cards = db.exec(select(Flashcard).where(Flashcard.study_set_id == original_set.id)).all()
                for old_card in original_cards:
                    new_card = Flashcard(
                        study_set_id=new_set.id, question=old_card.question, answer=old_card.answer, 
                        subject=old_card.subject, difficulty=old_card.difficulty
                    )
                    db.add(new_card)
                db.flush()
                new_item_id = str(new_set.id)

        elif item.item_type == "canvas":
            original_canvas = db.get(Canvas, uuid.UUID(item.item_id))
            if original_canvas:
                new_canvas = Canvas(
                    user_id=current_user.id, name=original_canvas.name, 
                    subject=original_canvas.subject, source_type="manual"
                )
                db.add(new_canvas)
                db.flush()

                original_nodes = db.exec(select(CanvasNode).where(CanvasNode.canvas_id == original_canvas.id)).all()
                id_map = {}
                for old_node in original_nodes:
                    new_node = CanvasNode(
                        canvas_id=new_canvas.id, label=old_node.label, x=old_node.x, 
                        y=old_node.y, size=old_node.size, is_hero=old_node.is_hero
                    )
                    db.add(new_node)
                    db.flush()
                    id_map[old_node.id] = new_node.id

                original_conns = db.exec(select(CanvasConnection).where(CanvasConnection.canvas_id == original_canvas.id)).all()
                for old_conn in original_conns:
                    if old_conn.from_node_id in id_map and old_conn.to_node_id in id_map:
                        new_conn = CanvasConnection(
                            canvas_id=new_canvas.id, from_node_id=id_map[old_conn.from_node_id], 
                            to_node_id=id_map[old_conn.to_node_id], label=old_conn.label
                        )
                        db.add(new_conn)
                db.flush()
                new_item_id = str(new_canvas.id)

        if new_item_id:
            new_mapping = CollectionItem(
                collection_id=new_collection.id, item_type=item.item_type, 
                item_id=new_item_id, position=item.position
            )
            db.add(new_mapping)
    
    db.commit() # Single final commit guarantees atomic success
    return {"message": "Collection cloned to your library! You can now edit it safely.", "new_collection_id": new_collection.id}

@router.post("/users/me/library/items", status_code=status.HTTP_200_OK)
def save_individual_item(
    payload: SaveItemRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_session)
):
    if payload.item_type == "note":
        original = db.get(Note, int(payload.item_id))
        if not original: raise HTTPException(status_code=404)
        new_note = Note(
            user_id=current_user.id, title=original.title, subject=original.subject, 
            content_text=original.content_text, content_html=original.content_html, word_count=original.word_count, snippet=original.snippet
        )
        db.add(new_note)
        
    elif payload.item_type == "set":
        original_set = db.get(StudySet, int(payload.item_id))
        if not original_set: raise HTTPException(status_code=404)
        
        new_set = StudySet(user_id=current_user.id, title=original_set.title, subject=original_set.subject, card_count=original_set.card_count)
        db.add(new_set)
        db.flush() 
        
        original_cards = db.exec(select(Flashcard).where(Flashcard.study_set_id == original_set.id)).all()
        for old_card in original_cards:
            new_card = Flashcard(
                study_set_id=new_set.id, question=old_card.question, answer=old_card.answer, subject=old_card.subject, difficulty=old_card.difficulty
            )
            db.add(new_card)

    elif payload.item_type == "canvas":
        original_canvas = db.get(Canvas, uuid.UUID(payload.item_id))
        if not original_canvas: raise HTTPException(status_code=404)
        
        new_canvas = Canvas(user_id=current_user.id, name=original_canvas.name, subject=original_canvas.subject, source_type="manual")
        db.add(new_canvas)
        db.flush()
        
        original_nodes = db.exec(select(CanvasNode).where(CanvasNode.canvas_id == original_canvas.id)).all()
        id_map = {}
        for old_node in original_nodes:
            new_node = CanvasNode(
                canvas_id=new_canvas.id, label=old_node.label, x=old_node.x, y=old_node.y, size=old_node.size, is_hero=old_node.is_hero
            )
            db.add(new_node)
            db.flush() 
            id_map[old_node.id] = new_node.id
            
        original_connections = db.exec(select(CanvasConnection).where(CanvasConnection.canvas_id == original_canvas.id)).all()
        for old_conn in original_connections:
            if old_conn.from_node_id in id_map and old_conn.to_node_id in id_map:
                new_conn = CanvasConnection(
                    canvas_id=new_canvas.id, from_node_id=id_map[old_conn.from_node_id], to_node_id=id_map[old_conn.to_node_id], label=old_conn.label
                )
                db.add(new_conn)

    db.commit() # Atomic save
    return {"message": f"{payload.item_type.capitalize()} saved to your library"}

@router.get("/collections/by-token/{share_token}", status_code=status.HTTP_200_OK)
def get_collection_metadata_by_token(share_token: str, db: Session = Depends(get_session)):
    collection = db.exec(select(Collection).where(Collection.share_token == share_token)).first()
    if not collection: raise HTTPException(status_code=404, detail="Link invalid or expired")
    owner = db.get(User, collection.user_id)
    return {"collection_id": collection.id, "title": collection.title, "owner_username": owner.name if owner else "Unknown User", "visibility": collection.visibility}

@router.get("/collections/{collection_id}/requests/my", status_code=status.HTTP_200_OK)
def check_my_request_status(collection_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_session)):
    req = db.exec(select(CollectionRequest).where(
        CollectionRequest.collection_id == collection_id, CollectionRequest.user_id == current_user.id
    ).order_by(CollectionRequest.id.desc())).first()
    return {"status": req.status if req else "none"}

@router.post("/collections/{collection_id}/report", status_code=status.HTTP_200_OK)
def report_collection(collection_id: int, payload: ReportRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_session)):
    print(f"Collection {collection_id} reported by {current_user.id} for: {payload.reason}")
    return {"message": "Collection reported. Our team will review it."}

# ==========================================
# 7. COMMUNITY DISCOVERY
# ==========================================
@router.get("/community/collections", status_code=status.HTTP_200_OK)
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
# 8. STUDY GROUPS & CO-OP
# ==========================================
def generate_invite_code():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

@router.post("/community/groups", status_code=status.HTTP_201_CREATED)
def create_group(
    payload: GroupCreate, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_session)
):
    new_group = StudyGroup(name=payload.name, invite_code=generate_invite_code())
    db.add(new_group)
    db.flush()
    
    member = GroupMember(group_id=new_group.id, user_id=current_user.id)
    db.add(member)
    db.commit()
    
    return new_group

@router.post("/community/groups/join", status_code=status.HTTP_200_OK)
def join_group(
    payload: GroupJoin, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_session)
):
    group = db.exec(select(StudyGroup).where(StudyGroup.invite_code == payload.invite_code.upper())).first()
    if not group:
        raise HTTPException(status_code=404, detail="Invalid invite code")
        
    existing = db.exec(select(GroupMember).where(GroupMember.group_id == group.id, GroupMember.user_id == current_user.id)).first()
    if existing:
        return {"message": "You are already a member of this group"}
        
    db.add(GroupMember(group_id=group.id, user_id=current_user.id))
    db.commit()
    
    return {"message": f"Successfully joined {group.name}!"}

@router.get("/community/quests/active", status_code=status.HTTP_200_OK)
def get_active_quests(current_user: User = Depends(get_current_user), db: Session = Depends(get_session)):
    # Flattens N+1 query into a single fast JOIN query
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

def contribute_to_group_quests(user_id: int, xp_amount: int, db: Session):
    memberships = db.exec(select(GroupMember).where(GroupMember.user_id == user_id)).all()
    
    for member in memberships:
        quest = db.exec(
            select(CoopQuest)
            .where(CoopQuest.group_id == member.group_id, CoopQuest.is_completed == False)
        ).first()
        
        if quest:
            quest.current_xp += xp_amount
            if quest.current_xp >= quest.target_xp:
                quest.is_completed = True
            db.add(quest)
            db.flush()
            
    db.commit()