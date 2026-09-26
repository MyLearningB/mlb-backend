import random
import string
import uuid
from typing import List, Optional, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlmodel import Session, select, func, or_, and_

# Database, Models, and Auth
from database import get_session
from security import get_current_user
from models import (
    User,
    Collection,
    CollectionItem,
    CollectionAccess,
    CollectionRequest,
    Note,
    StudySet,
    Canvas,
    CanvasConnection,
    CanvasNode,
    Flashcard,
    StudyGroup,
    GroupMember,
    CoopQuest,
)
from services.notifications import send_collection_notification
from schemas import CollectionUpdate, ItemReorderRequest


router = APIRouter(tags=["Collections & Community Section"])


# ============================================================
# CONSTANTS
# ============================================================

COLLECTION_VISIBILITIES = {"private", "shared", "public"}
COLLECTION_ITEM_TYPES = {"note", "set", "canvas"}

REQUEST_STATUSES = {"pending", "approved", "denied"}


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class CollectionCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    subject: str = Field(..., min_length=1, max_length=100)
    visibility: Literal["private", "shared", "public"]
    description: Optional[str] = Field(default=None, max_length=5000)
    cover_emoji: Optional[str] = Field(default=None, max_length=20)

    item_ids: List[str] = Field(default_factory=list)
    item_types: List[str] = Field(default_factory=list)


class ItemAddRequest(BaseModel):
    item_id: str = Field(..., min_length=1)
    item_type: Literal["note", "set", "canvas"]


class InviteRequest(BaseModel):
    emails: List[str] = Field(..., min_length=1, max_length=50)


class AccessRequestSubmit(BaseModel):
    message: Optional[str] = Field(default=None, max_length=2000)


class RequestStatusUpdate(BaseModel):
    status: Literal["approved", "denied"]


class SaveItemRequest(BaseModel):
    item_id: str = Field(..., min_length=1)
    item_type: Literal["note", "set", "canvas"]


class ReportRequest(BaseModel):
    reason: str = Field(..., min_length=1, max_length=2000)


class GroupCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=50)


class GroupJoin(BaseModel):
    invite_code: str = Field(..., min_length=6, max_length=6)


# ============================================================
# HELPERS
# ============================================================

def generate_invite_code() -> str:
    """
    Generate a six-character group invite code.
    """
    return "".join(
        random.choices(
            string.ascii_uppercase + string.digits,
            k=6,
        )
    )


def generate_unique_invite_code(db: Session) -> str:
    """
    Generate an invite code that does not already exist.
    """
    for _ in range(20):
        code = generate_invite_code()

        existing = db.exec(
            select(StudyGroup).where(
                StudyGroup.invite_code == code
            )
        ).first()

        if not existing:
            return code

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Unable to generate a unique invite code",
    )


def parse_int_item_id(item_id: str, item_type: str) -> int:
    """
    Safely parse integer-backed item IDs.
    """
    try:
        return int(item_id)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {item_type} ID",
        )


def parse_uuid_item_id(item_id: str) -> uuid.UUID:
    """
    Safely parse UUID-backed Canvas IDs.
    """
    try:
        return uuid.UUID(item_id)
    except (TypeError, ValueError, AttributeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid canvas ID",
        )


def validate_collection_items(
    item_ids: List[str],
    item_types: List[str],
) -> None:
    """
    Validate that item IDs and types are supplied together.
    """
    if len(item_ids) != len(item_types):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="item_ids and item_types must contain the same number of items",
        )

    invalid_types = [
        item_type
        for item_type in item_types
        if item_type not in COLLECTION_ITEM_TYPES
    ]

    if invalid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported item type(s): {invalid_types}",
        )


def get_owned_collection(
    collection_id: int,
    current_user: User,
    db: Session,
) -> Collection:
    """
    Return a collection only if it exists and belongs to the user.
    """
    collection = db.get(Collection, collection_id)

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found",
        )

    if collection.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found",
        )

    return collection


def user_has_collection_access(
    collection: Collection,
    user_id: int,
    db: Session,
) -> bool:
    """
    Determine whether a user can access a collection.
    """
    if collection.user_id == user_id:
        return True

    if collection.visibility == "public":
        return True

    access = db.exec(
        select(CollectionAccess).where(
            CollectionAccess.collection_id == collection.id,
            CollectionAccess.user_id == user_id,
        )
    ).first()

    return access is not None


# ============================================================
# 1. MY COLLECTIONS LIBRARY
# ============================================================

@router.get(
    "/users/me/collections",
    status_code=status.HTTP_200_OK,
)
def get_my_collections(
    search: Optional[str] = None,
    sort: str = Query(
        "recent",
        pattern="^(recent|oldest|title|saves)$",
    ),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    query = select(Collection).where(
        Collection.user_id == current_user.id
    )

    if search:
        search_term = search.strip()

        if search_term:
            query = query.where(
                or_(
                    Collection.title.icontains(search_term),
                    Collection.subject.icontains(search_term),
                )
            )

    if sort == "recent":
        query = query.order_by(Collection.updated_at.desc())

    elif sort == "oldest":
        query = query.order_by(Collection.created_at.asc())

    elif sort == "title":
        query = query.order_by(Collection.title.asc())

    elif sort == "saves":
        query = query.order_by(Collection.save_count.desc())

    offset = (page - 1) * limit

    collections = db.exec(
        query.offset(offset).limit(limit)
    ).all()

    count_query = select(
        func.count(Collection.id)
    ).where(
        Collection.user_id == current_user.id
    )

    if search:
        search_term = search.strip()

        if search_term:
            count_query = count_query.where(
                or_(
                    Collection.title.icontains(search_term),
                    Collection.subject.icontains(search_term),
                )
            )

    total_count = db.exec(count_query).one()

    return {
        "collections": collections,
        "total_count": total_count,
        "has_more": total_count > offset + limit,
        "page": page,
        "limit": limit,
    }


# ============================================================
# 2. CREATE & EDIT COLLECTION
# ============================================================

@router.post(
    "/collections",
    status_code=status.HTTP_201_CREATED,
)
def create_collection(
    payload: CollectionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    validate_collection_items(
        payload.item_ids,
        payload.item_types,
    )

    try:
        new_collection = Collection(
            user_id=current_user.id,
            title=payload.title.strip(),
            subject=payload.subject.strip(),
            visibility=payload.visibility,
            description=payload.description,
            cover_emoji=payload.cover_emoji,
        )

        db.add(new_collection)
        db.flush()

        for position, (item_id, item_type) in enumerate(
            zip(payload.item_ids, payload.item_types)
        ):
            new_item = CollectionItem(
                collection_id=new_collection.id,
                item_type=item_type,
                item_id=item_id,
                position=position,
            )

            db.add(new_item)

        db.commit()
        db.refresh(new_collection)

        return {
            "collection_id": new_collection.id,
            "share_token": new_collection.share_token,
            "message": "Collection created successfully",
        }

    except Exception:
        db.rollback()
        raise


@router.get(
    "/collections/{collection_id}",
    status_code=status.HTTP_200_OK,
)
def get_collection_detail(
    collection_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    statement = (
        select(Collection, User)
        .join(User, Collection.user_id == User.id)
        .where(Collection.id == collection_id)
    )

    result = db.exec(statement).first()

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found",
        )

    collection, owner = result

    if not user_has_collection_access(
        collection,
        current_user.id,
        db,
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this collection",
        )

    item_mappings = db.exec(
        select(CollectionItem)
        .where(
            CollectionItem.collection_id == collection.id
        )
        .order_by(CollectionItem.position.asc())
    ).all()

    resolved_items = []

    for mapping in item_mappings:

        if mapping.item_type == "note":
            item_id = parse_int_item_id(
                mapping.item_id,
                "note",
            )

            note = db.get(Note, item_id)

            if note:
                resolved_items.append({
                    "id": str(note.id),
                    "type": "note",
                    "title": note.title,
                    "subject": note.subject,
                })

        elif mapping.item_type == "set":
            item_id = parse_int_item_id(
                mapping.item_id,
                "set",
            )

            study_set = db.get(StudySet, item_id)

            if study_set:
                resolved_items.append({
                    "id": str(study_set.id),
                    "type": "set",
                    "title": study_set.title,
                    "card_count": study_set.card_count,
                })

        elif mapping.item_type == "canvas":
            canvas_id = parse_uuid_item_id(
                mapping.item_id
            )

            canvas = db.get(Canvas, canvas_id)

            if canvas:
                resolved_items.append({
                    "id": str(canvas.id),
                    "type": "canvas",
                    "title": canvas.name,
                    "node_count": canvas.node_count,
                })

    response_data = collection.model_dump()
    response_data["owner_username"] = owner.name

    return {
        "collection": response_data,
        "items": resolved_items,
    }


@router.patch(
    "/collections/{collection_id}",
    status_code=status.HTTP_200_OK,
)
def update_collection_settings(
    collection_id: int,
    payload: CollectionUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    collection = get_owned_collection(
        collection_id,
        current_user,
        db,
    )

    update_data = payload.model_dump(
        exclude_unset=True
    )

    if "visibility" in update_data:
        if update_data["visibility"] not in COLLECTION_VISIBILITIES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid collection visibility",
            )

    if "title" in update_data and update_data["title"]:
        update_data["title"] = update_data["title"].strip()

    if "subject" in update_data and update_data["subject"]:
        update_data["subject"] = update_data["subject"].strip()

    collection.sqlmodel_update(update_data)

    db.add(collection)
    db.commit()
    db.refresh(collection)

    return {
        "message": "Collection updated successfully"
    }


@router.delete(
    "/collections/{collection_id}",
    status_code=status.HTTP_200_OK,
)
def delete_collection(
    collection_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    collection = get_owned_collection(
        collection_id,
        current_user,
        db,
    )

    try:
        items = db.exec(
            select(CollectionItem).where(
                CollectionItem.collection_id == collection.id
            )
        ).all()

        for item in items:
            db.delete(item)

        access_rows = db.exec(
            select(CollectionAccess).where(
                CollectionAccess.collection_id == collection.id
            )
        ).all()

        for row in access_rows:
            db.delete(row)

        req_rows = db.exec(
            select(CollectionRequest).where(
                CollectionRequest.collection_id == collection.id
            )
        ).all()

        for row in req_rows:
            db.delete(row)

        db.delete(collection)
        db.commit()

        return {
            "message": (
                "Collection deleted successfully. "
                "Your items are safe."
            )
        }

    except Exception:
        db.rollback()
        raise


# ============================================================
# 3. MANAGE ITEMS IN COLLECTION
# ============================================================

@router.post(
    "/collections/{collection_id}/items",
    status_code=status.HTTP_200_OK,
)
def add_item_to_collection(
    collection_id: int,
    payload: ItemAddRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    collection = get_owned_collection(
        collection_id,
        current_user,
        db,
    )

    existing_item = db.exec(
        select(CollectionItem).where(
            CollectionItem.collection_id == collection.id,
            CollectionItem.item_id == payload.item_id,
            CollectionItem.item_type == payload.item_type,
        )
    ).first()

    if existing_item:
        return {
            "message": "Item already in collection"
        }

    max_pos = db.exec(
        select(func.max(CollectionItem.position)).where(
            CollectionItem.collection_id == collection.id
        )
    ).one()

    next_pos = 0 if max_pos is None else max_pos + 1

    new_item = CollectionItem(
        collection_id=collection.id,
        item_type=payload.item_type,
        item_id=payload.item_id,
        position=next_pos,
    )

    db.add(new_item)
    db.commit()

    return {
        "message": "Item added successfully"
    }


@router.patch(
    "/collections/{collection_id}/items",
    status_code=status.HTTP_200_OK,
)
def reorder_collection_items(
    collection_id: int,
    payload: ItemReorderRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    collection = get_owned_collection(
        collection_id,
        current_user,
        db,
    )

    mappings = db.exec(
        select(CollectionItem).where(
            CollectionItem.collection_id == collection.id
        )
    ).all()

    mapping_lookup = {
        (
            str(item.item_id),
            str(item.item_type),
        ): item
        for item in mappings
    }

    for reorder_item in payload.positions:
        item_id = str(reorder_item.item_id)

        item_type = getattr(
            reorder_item,
            "item_type",
            None,
        )

        mapping = None

        if item_type:
            mapping = mapping_lookup.get(
                (item_id, str(item_type))
            )
        else:
            # Backward-compatible fallback.
            candidates = [
                item
                for item in mappings
                if str(item.item_id) == item_id
            ]

            if len(candidates) == 1:
                mapping = candidates[0]

        if mapping:
            mapping.position = reorder_item.position
            db.add(mapping)

    db.commit()

    return {
        "message": "Items reordered successfully"
    }


# ============================================================
# 4. SHARE SETTINGS
# ============================================================

@router.post(
    "/collections/{collection_id}/share-token/regenerate",
    status_code=status.HTTP_200_OK,
)
def regenerate_share_token(
    collection_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    collection = get_owned_collection(
        collection_id,
        current_user,
        db,
    )

    collection.share_token = uuid.uuid4().hex[:12]

    db.add(collection)
    db.commit()
    db.refresh(collection)

    return {
        "new_share_token": collection.share_token
    }


@router.post(
    "/collections/{collection_id}/invites",
    status_code=status.HTTP_200_OK,
)
def invite_users_by_email(
    collection_id: int,
    payload: InviteRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    collection = get_owned_collection(
        collection_id,
        current_user,
        db,
    )

    invited = []
    already_had_access = []
    not_found = []

    # Normalize and remove duplicates while preserving order.
    normalized_emails = list(
        dict.fromkeys(
            email.strip().lower()
            for email in payload.emails
            if email.strip()
        )
    )

    for email in normalized_emails:
        user = db.exec(
            select(User).where(
                User.email == email
            )
        ).first()

        if not user:
            not_found.append(email)
            continue

        if user.id == current_user.id:
            already_had_access.append(email)
            continue

        existing_access = db.exec(
            select(CollectionAccess).where(
                CollectionAccess.collection_id == collection.id,
                CollectionAccess.user_id == user.id,
            )
        ).first()

        if existing_access:
            already_had_access.append(email)
            continue

        new_access = CollectionAccess(
            collection_id=collection.id,
            user_id=user.id,
        )

        db.add(new_access)
        invited.append(email)

        send_collection_notification(
            db=db,
            user_id=user.id,
            title="Collection Invitation",
            body=(
                f"@{current_user.name} invited you to "
                f"'{collection.title}'."
            ),
            deep_link=(
                f"/collections/{collection.id}/view"
            ),
        )

    db.commit()

    return {
        "invited": invited,
        "already_had_access": already_had_access,
        "not_found": not_found,
    }


@router.delete(
    "/collections/{collection_id}/access/{user_id}",
    status_code=status.HTTP_200_OK,
)
def revoke_access(
    collection_id: int,
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    collection = get_owned_collection(
        collection_id,
        current_user,
        db,
    )

    access_record = db.exec(
        select(CollectionAccess).where(
            CollectionAccess.collection_id == collection.id,
            CollectionAccess.user_id == user_id,
        )
    ).first()

    if access_record:
        db.delete(access_record)
        db.commit()

        send_collection_notification(
            db=db,
            user_id=user_id,
            title="Access Removed",
            body=(
                f"Your access to '{collection.title}' "
                f"has been removed by @{current_user.name}."
            ),
            deep_link=None,
        )

    return {
        "message": "Access revoked successfully"
    }


# ============================================================
# 5. ACCESS REQUESTS
# ============================================================

@router.post(
    "/collections/{collection_id}/requests",
    status_code=status.HTTP_200_OK,
)
def submit_access_request(
    collection_id: int,
    payload: AccessRequestSubmit,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    collection = db.get(Collection, collection_id)

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found",
        )

    if collection.user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You already own this collection",
        )

    if collection.visibility == "public":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Public collections do not require access requests",
        )

    if user_has_collection_access(
        collection,
        current_user.id,
        db,
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="You already have access to this collection",
        )

    existing_req = db.exec(
        select(CollectionRequest).where(
            CollectionRequest.collection_id == collection.id,
            CollectionRequest.user_id == current_user.id,
            CollectionRequest.status == "pending",
        )
    ).first()

    if existing_req:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "You already have a pending request "
                "for this collection."
            ),
        )

    new_request = CollectionRequest(
        collection_id=collection.id,
        user_id=current_user.id,
        message=payload.message,
        status="pending",
    )

    db.add(new_request)
    db.commit()
    db.refresh(new_request)

    send_collection_notification(
        db=db,
        user_id=collection.user_id,
        title="New Access Request",
        body=(
            f"{current_user.name} wants access to "
            f"'{collection.title}'"
        ),
        deep_link=(
            f"/collections/{collection.id}/share"
            f"?request_id={new_request.id}"
        ),
    )

    return {
        "message": "Request sent successfully",
        "request_id": new_request.id,
    }


@router.get(
    "/collections/{collection_id}/requests",
    status_code=status.HTTP_200_OK,
)
def get_pending_requests(
    collection_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    collection = get_owned_collection(
        collection_id,
        current_user,
        db,
    )

    requests = db.exec(
        select(CollectionRequest, User)
        .join(
            User,
            CollectionRequest.user_id == User.id,
        )
        .where(
            CollectionRequest.collection_id == collection.id,
            CollectionRequest.status == "pending",
        )
        .order_by(CollectionRequest.id.desc())
    ).all()

    formatted_requests = [
        {
            "request_id": request.id,
            "user_id": user.id,
            "username": user.name,
            "email": user.email,
            "message": request.message,
            "requested_at": request.requested_at,
        }
        for request, user in requests
    ]

    return {
        "requests": formatted_requests
    }


@router.patch(
    "/collections/requests/{request_id}",
    status_code=status.HTTP_200_OK,
)
def update_request_status(
    request_id: int,
    payload: RequestStatusUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    coll_request = db.get(
        CollectionRequest,
        request_id,
    )

    if not coll_request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Request not found",
        )

    collection = db.get(
        Collection,
        coll_request.collection_id,
    )

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found",
        )

    if collection.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "You do not have permission to manage "
                "this collection"
            ),
        )

    if coll_request.status != "pending":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"This request has already been "
                f"{coll_request.status}"
            ),
        )

    coll_request.status = payload.status
    db.add(coll_request)

    if payload.status == "approved":

        existing_access = db.exec(
            select(CollectionAccess).where(
                CollectionAccess.collection_id == collection.id,
                CollectionAccess.user_id == coll_request.user_id,
            )
        ).first()

        if not existing_access:
            access = CollectionAccess(
                collection_id=collection.id,
                user_id=coll_request.user_id,
            )

            db.add(access)

    db.commit()

    if payload.status == "approved":
        send_collection_notification(
            db=db,
            user_id=coll_request.user_id,
            title="Access Granted!",
            body=(
                f"@{current_user.name} approved your access "
                f"to '{collection.title}' — open it now!"
            ),
            deep_link=(
                f"/collections/{collection.id}/view"
            ),
        )

    else:
        send_collection_notification(
            db=db,
            user_id=coll_request.user_id,
            title="Access Denied",
            body=(
                f"Your request to '{collection.title}' "
                "was not approved."
            ),
            deep_link=None,
        )

    return {
        "message": (
            f"Request {payload.status} successfully"
        ),
        "request_id": coll_request.id,
        "status": coll_request.status,
    }


# ============================================================
# 6. SAVE / CLONE COLLECTIONS AND ITEMS
# ============================================================

def clone_note(
    original_note: Note,
    user_id: int,
    db: Session,
) -> Note:
    new_note = Note(
        user_id=user_id,
        title=original_note.title,
        subject=original_note.subject,
        content_text=original_note.content_text,
        content_html=original_note.content_html,
        word_count=original_note.word_count,
        snippet=original_note.snippet,
    )

    db.add(new_note)
    db.flush()

    return new_note


def clone_study_set(
    original_set: StudySet,
    user_id: int,
    db: Session,
) -> StudySet:
    original_cards = db.exec(
        select(Flashcard).where(
            Flashcard.study_set_id == original_set.id
        )
    ).all()

    new_set = StudySet(
        user_id=user_id,
        title=original_set.title,
        subject=original_set.subject,
        card_count=len(original_cards),
    )

    db.add(new_set)
    db.flush()

    for old_card in original_cards:
        new_card = Flashcard(
            study_set_id=new_set.id,
            question=old_card.question,
            answer=old_card.answer,
            subject=old_card.subject,
            difficulty=old_card.difficulty,
        )

        db.add(new_card)

    return new_set


def clone_canvas(
    original_canvas: Canvas,
    user_id: int,
    db: Session,
) -> Canvas:
    new_canvas = Canvas(
        user_id=user_id,
        name=original_canvas.name,
        subject=original_canvas.subject,
        source_type="manual",
    )

    db.add(new_canvas)
    db.flush()

    original_nodes = db.exec(
        select(CanvasNode).where(
            CanvasNode.canvas_id == original_canvas.id
        )
    ).all()

    id_map = {}

    for old_node in original_nodes:
        new_node = CanvasNode(
            canvas_id=new_canvas.id,
            label=old_node.label,
            x=old_node.x,
            y=old_node.y,
            size=old_node.size,
            is_hero=old_node.is_hero,
        )

        db.add(new_node)
        db.flush()

        id_map[old_node.id] = new_node.id

    original_connections = db.exec(
        select(CanvasConnection).where(
            CanvasConnection.canvas_id == original_canvas.id
        )
    ).all()

    for old_connection in original_connections:
        if (
            old_connection.from_node_id in id_map
            and old_connection.to_node_id in id_map
        ):
            new_connection = CanvasConnection(
                canvas_id=new_canvas.id,
                from_node_id=id_map[
                    old_connection.from_node_id
                ],
                to_node_id=id_map[
                    old_connection.to_node_id
                ],
                label=old_connection.label,
            )

            db.add(new_connection)

    return new_canvas


def clone_collection_item(
    item: CollectionItem,
    user_id: int,
    db: Session,
) -> Optional[str]:
    """
    Clone one collection item and return its new ID.
    """

    if item.item_type == "note":
        original = db.get(
            Note,
            parse_int_item_id(
                item.item_id,
                "note",
            ),
        )

        if not original:
            return None

        new_note = clone_note(
            original,
            user_id,
            db,
        )

        return str(new_note.id)

    if item.item_type == "set":
        original = db.get(
            StudySet,
            parse_int_item_id(
                item.item_id,
                "set",
            ),
        )

        if not original:
            return None

        new_set = clone_study_set(
            original,
            user_id,
            db,
        )

        return str(new_set.id)

    if item.item_type == "canvas":
        original = db.get(
            Canvas,
            parse_uuid_item_id(item.item_id),
        )

        if not original:
            return None

        new_canvas = clone_canvas(
            original,
            user_id,
            db,
        )

        return str(new_canvas.id)

    return None


@router.post(
    "/collections/{collection_id}/save",
    status_code=status.HTTP_200_OK,
)
def save_public_collection(
    collection_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    original_collection = db.get(
        Collection,
        collection_id,
    )

    if not original_collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found",
        )

    if original_collection.visibility != "public":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only public collections can be saved",
        )

    try:
        original_collection.save_count = (
            original_collection.save_count + 1
        )

        db.add(original_collection)

        new_collection = Collection(
            user_id=current_user.id,
            title=f"{original_collection.title} (Copy)",
            description=original_collection.description,
            subject=original_collection.subject,
            cover_emoji=original_collection.cover_emoji,
            visibility="private",
            share_token=uuid.uuid4().hex[:12],
        )

        db.add(new_collection)
        db.flush()

        original_items = db.exec(
            select(CollectionItem)
            .where(
                CollectionItem.collection_id
                == original_collection.id
            )
            .order_by(CollectionItem.position.asc())
        ).all()

        for item in original_items:
            new_item_id = clone_collection_item(
                item,
                current_user.id,
                db,
            )

            if not new_item_id:
                continue

            new_mapping = CollectionItem(
                collection_id=new_collection.id,
                item_type=item.item_type,
                item_id=new_item_id,
                position=item.position,
            )

            db.add(new_mapping)

        db.commit()
        db.refresh(new_collection)

        return {
            "message": (
                "Collection cloned to your library! "
                "You can now edit it safely."
            ),
            "new_collection_id": new_collection.id,
        }

    except Exception:
        db.rollback()
        raise


@router.post(
    "/users/me/library/items",
    status_code=status.HTTP_200_OK,
)
def save_individual_item(
    payload: SaveItemRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    try:
        if payload.item_type == "note":
            original = db.get(
                Note,
                parse_int_item_id(
                    payload.item_id,
                    "note",
                ),
            )

            if not original:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Note not found",
                )

            clone_note(
                original,
                current_user.id,
                db,
            )

        elif payload.item_type == "set":
            original_set = db.get(
                StudySet,
                parse_int_item_id(
                    payload.item_id,
                    "set",
                ),
            )

            if not original_set:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Study set not found",
                )

            clone_study_set(
                original_set,
                current_user.id,
                db,
            )

        elif payload.item_type == "canvas":
            original_canvas = db.get(
                Canvas,
                parse_uuid_item_id(payload.item_id),
            )

            if not original_canvas:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Canvas not found",
                )

            clone_canvas(
                original_canvas,
                current_user.id,
                db,
            )

        db.commit()

        return {
            "message": (
                f"{payload.item_type.capitalize()} "
                "saved to your library"
            )
        }

    except HTTPException:
        db.rollback()
        raise

    except Exception:
        db.rollback()
        raise


@router.get(
    "/collections/by-token/{share_token}",
    status_code=status.HTTP_200_OK,
)
def get_collection_metadata_by_token(
    share_token: str,
    db: Session = Depends(get_session),
):
    collection = db.exec(
        select(Collection).where(
            Collection.share_token == share_token
        )
    ).first()

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Link invalid or expired",
        )

    owner = db.get(
        User,
        collection.user_id,
    )

    return {
        "collection_id": collection.id,
        "title": collection.title,
        "owner_username": (
            owner.name
            if owner
            else "Unknown User"
        ),
        "visibility": collection.visibility,
    }


@router.get(
    "/collections/{collection_id}/requests/my",
    status_code=status.HTTP_200_OK,
)
def check_my_request_status(
    collection_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    collection = db.get(
        Collection,
        collection_id,
    )

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found",
        )

    req = db.exec(
        select(CollectionRequest)
        .where(
            CollectionRequest.collection_id
            == collection_id,
            CollectionRequest.user_id
            == current_user.id,
        )
        .order_by(
            CollectionRequest.id.desc()
        )
    ).first()

    return {
        "status": (
            req.status
            if req
            else "none"
        )
    }


@router.post(
    "/collections/{collection_id}/report",
    status_code=status.HTTP_200_OK,
)
def report_collection(
    collection_id: int,
    payload: ReportRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    collection = db.get(
        Collection,
        collection_id,
    )

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found",
        )

    # IMPORTANT:
    # This endpoint currently has no Report model in the
    # models imported by this router. Keep this logging behavior
    # until a persistent Report model/table is added.
    print(
        f"Collection {collection_id} "
        f"reported by {current_user.id}: "
        f"{payload.reason}"
    )

    return {
        "message": (
            "Collection reported. "
            "Our team will review it."
        )
    }


# ============================================================
# 7. COMMUNITY DISCOVERY
# ============================================================

@router.get(
    "/community/collections",
    status_code=status.HTTP_200_OK,
)
def get_discover_collections(
    search: Optional[str] = None,
    filter: str = Query("All"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """
    Community discovery should expose PUBLIC collections only.

    Private collections must never appear here.
    """

    conditions = [
        Collection.visibility == "public"
    ]

    if search:
        search_term = search.strip()

        if search_term:
            conditions.append(
                or_(
                    Collection.title.icontains(
                        search_term
                    ),
                    Collection.subject.icontains(
                        search_term
                    ),
                )
            )

    normalized_filter = (
        filter.strip().lower()
        if filter
        else "all"
    )

    if (
        normalized_filter != "all"
        and normalized_filter != "public"
    ):
        # Subject filtering.
        conditions.append(
            Collection.subject.ilike(
                normalized_filter
            )
        )

    query = (
        select(Collection)
        .where(and_(*conditions))
        .order_by(
            Collection.save_count.desc(),
            Collection.created_at.desc(),
        )
    )

    offset = (page - 1) * limit

    collections = db.exec(
        query.offset(offset).limit(limit)
    ).all()

    count_query = (
        select(func.count(Collection.id))
        .where(and_(*conditions))
    )

    total_count = db.exec(
        count_query
    ).one()

    return {
        "collections": collections,
        "total_count": total_count,
        "has_more": total_count > offset + limit,
        "page": page,
        "limit": limit,
    }


# ============================================================
# 8. STUDY GROUPS & CO-OP
# ============================================================

@router.post(
    "/community/groups",
    status_code=status.HTTP_201_CREATED,
)
def create_group(
    payload: GroupCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    try:
        new_group = StudyGroup(
            name=payload.name.strip(),
            invite_code=generate_unique_invite_code(db),
        )

        db.add(new_group)
        db.flush()

        member = GroupMember(
            group_id=new_group.id,
            user_id=current_user.id,
        )

        db.add(member)
        db.commit()
        db.refresh(new_group)

        return new_group

    except Exception:
        db.rollback()
        raise


@router.post(
    "/community/groups/join",
    status_code=status.HTTP_200_OK,
)
def join_group(
    payload: GroupJoin,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    invite_code = payload.invite_code.strip().upper()

    group = db.exec(
        select(StudyGroup).where(
            StudyGroup.invite_code == invite_code
        )
    ).first()

    if not group:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid invite code",
        )

    existing = db.exec(
        select(GroupMember).where(
            GroupMember.group_id == group.id,
            GroupMember.user_id == current_user.id,
        )
    ).first()

    if existing:
        return {
            "message": (
                "You are already a member "
                "of this group"
            )
        }

    db.add(
        GroupMember(
            group_id=group.id,
            user_id=current_user.id,
        )
    )

    db.commit()

    return {
        "message": (
            f"Successfully joined {group.name}!"
        )
    }


@router.get(
    "/community/quests/active",
    status_code=status.HTTP_200_OK,
)
def get_active_quests(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """
    Return active quests belonging to groups that the
    current user belongs to.

    The old implementation counted only the current
    user's membership because it filtered GroupMember
    before counting. This version uses a separate member
    count.
    """

    membership_subquery = (
        select(GroupMember.group_id)
        .where(
            GroupMember.user_id == current_user.id
        )
    )

    member_count_subquery = (
        select(
            GroupMember.group_id,
            func.count(
                GroupMember.user_id
            ).label("member_count"),
        )
        .group_by(GroupMember.group_id)
        .subquery()
    )

    statement = (
        select(
            CoopQuest,
            member_count_subquery.c.member_count,
        )
        .join(
            member_count_subquery,
            member_count_subquery.c.group_id
            == CoopQuest.group_id,
        )
        .where(
            CoopQuest.group_id.in_(
                membership_subquery
            )
        )
        .where(
            CoopQuest.is_completed == False
        )
        .order_by(CoopQuest.id.desc())
    )

    results = db.exec(
        statement
    ).all()

    return {
        "quests": [
            {
                "id": str(quest.id),
                "title": quest.title,
                "progress": quest.current_xp,
                "target": quest.target_xp,
                "type": "coop",
                "membersCount": int(
                    member_count or 0
                ),
            }
            for quest, member_count in results
        ]
    }


def contribute_to_group_quests(
    user_id: int,
    xp_amount: int,
    db: Session,
):
    """
    Add XP to active quests for every group the user
    belongs to.

    This function intentionally performs ONE commit
    at the end rather than committing once per group.
    """

    if xp_amount <= 0:
        return

    memberships = db.exec(
        select(GroupMember).where(
            GroupMember.user_id == user_id
        )
    ).all()

    if not memberships:
        return

    group_ids = {
        membership.group_id
        for membership in memberships
    }

    if not group_ids:
        return

    quests = db.exec(
        select(CoopQuest).where(
            CoopQuest.group_id.in_(group_ids),
            CoopQuest.is_completed == False,
        )
    ).all()

    for quest in quests:
        quest.current_xp += xp_amount

        if quest.current_xp >= quest.target_xp:
            quest.current_xp = quest.target_xp
            quest.is_completed = True

        db.add(quest)

    db.commit()
