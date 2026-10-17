# routers/notes.py
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlmodel import Session, select, func, or_
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timezone
import json
import os
from openai import OpenAI # FIXED: Switched to synchronous client

# Database, Models, and Authentication
from database import get_session, engine # FIXED: Imported engine for background task
from security import get_current_user 
from models import CollectionAccess, CollectionItem, Collection, User, Note, Flashcard, StudySet

router = APIRouter(tags=["Notes Section"])

# FIXED: Initialized Synchronous Gemini Client
gemini_client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY", "your-gemini-api-key"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# ==========================================
# PYDANTIC SCHEMAS
# ==========================================
class NoteCreate(BaseModel):
    title: str = "Untitled note"
    subject: str
    content_text: str = ""
    content_html: Optional[str] = None

class NoteUpdate(BaseModel):
    title: Optional[str] = None
    subject: Optional[str] = None
    content_text: Optional[str] = None
    content_html: Optional[str] = None
    is_public: Optional[bool] = None

class GenerateCardsOptions(BaseModel):
    definitions: bool = True
    questions: bool = True

class GenerateCardsRequest(BaseModel):
    options: GenerateCardsOptions

class ManualCardCreate(BaseModel):
    question: str
    answer: str

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def calculate_note_metadata(text: str):
    word_count = len(text.split()) if text else 0
    snippet = text[:77] + "..." if len(text) > 80 else text
    return word_count, snippet

# FIXED: Changed to sync `def` and handles its own DB session
def generate_cards_bg(note_id: int, user_id: int, options: GenerateCardsOptions):
    print(f"Starting Gemini flashcard generation for note {note_id}...")
    
    # Background task MUST open its own session, independent of the HTTP request
    with Session(engine) as db:
        note = db.get(Note, note_id)
        if not note or not note.content_text:
            return

        focus_instruction = "Generate both definition-style cards and conceptual questions."
        if options.definitions and not options.questions:
            focus_instruction = "Generate ONLY definition-style vocabulary cards (Term -> Definition)."
        elif options.questions and not options.definitions:
            focus_instruction = "Generate ONLY conceptual question/answer cards. Do not generate simple vocabulary definitions."

        prompt = f"""
        You are an expert tutor. Create high-quality flashcards from the provided study notes.
        {focus_instruction}
        
        Return ONLY raw, valid JSON. Format strictly as:
        {{
          "cards": [
            {{
              "question": "The front of the card",
              "answer": "The back of the card (clear and concise)",
              "difficulty": "easy" | "medium" | "hard"
            }}
          ]
        }}
        
        NOTES TEXT:
        {note.content_text}
        """

        try:
            # FIXED: Removed await, using sync client
            response = gemini_client.chat.completions.create(
                 model="gemini-3.8-flash",
                messages=[
                    {"role": "system", "content": "You are an API that only returns valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            ai_data = json.loads(response.choices[0].message.content)
            cards_data = ai_data.get("cards", [])
            
            if not cards_data:
                return

            study_set = db.exec(select(StudySet).where(StudySet.title == f"Set: {note.title}")).first()
            if not study_set:
                study_set = StudySet(
                    user_id=user_id,
                    title=f"Set: {note.title}",
                    subject=note.subject,
                    card_count=0
                )
                db.add(study_set)
                db.commit()
                db.refresh(study_set)

            for c in cards_data:
                new_card = Flashcard(
                    study_set_id=study_set.id,
                    note_id=note.id,
                    question=c.get("question", "")[:200], 
                    answer=c.get("answer", "")[:400],
                    subject=note.subject,
                    difficulty=c.get("difficulty", "medium")
                )
                db.add(new_card)
            
            study_set.card_count += len(cards_data)
            note.card_count += len(cards_data)
            
            db.commit()
            print(f"Successfully generated {len(cards_data)} cards for note {note_id}")

        except Exception as e:
            print(f"Gemini Error generating cards: {e}")

# ==========================================
# 1. GET ALL NOTES
# ==========================================
# FIXED: Changed all routes to sync `def` to prevent blocking the event loop
@router.get("/users/me/notes", status_code=status.HTTP_200_OK)
def get_all_notes(
    search: Optional[str] = None,
    sort: str = Query("recent"),
    filter: str = Query("all"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    query = select(Note).where(Note.user_id == current_user.id)
    
    if search:
        query = query.where((Note.title.icontains(search)) | (Note.content_text.icontains(search)))
        
    if filter.lower() != "all":
        query = query.where(Note.subject.ilike(filter))
        
    if sort == "recent":
        query = query.order_by(Note.updated_at.desc())
    elif sort == "created":
        query = query.order_by(Note.created_at.desc())
    elif sort == "a-z":
        query = query.order_by(Note.title.asc())
    elif sort == "most-cards":
        query = query.order_by(Note.card_count.desc())
        
    offset = (page - 1) * limit
    notes = db.exec(query.offset(offset).limit(limit)).all()
    
    total_count = db.exec(select(func.count(Note.id)).where(Note.user_id == current_user.id)).one()

    return {
        "notes": notes,
        "total_count": total_count,
        "has_more": total_count > (offset + limit)
    }

# ==========================================
# 2. CREATE NOTE
# ==========================================
@router.post("/notes", status_code=status.HTTP_201_CREATED)
def create_note(
    payload: NoteCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    word_count, snippet = calculate_note_metadata(payload.content_text)
    
    new_note = Note(
        user_id=current_user.id,
        title=payload.title,
        subject=payload.subject,
        content_text=payload.content_text,
        content_html=payload.content_html,
        word_count=word_count,
        snippet=snippet
    )
    
    db.add(new_note)
    db.commit()
    db.refresh(new_note)
    
    return {"note_id": new_note.id, "created_at": new_note.created_at}

# ==========================================
# 3. GET NOTE DETAIL
# ==========================================
@router.get("/notes/{note_id}", status_code=status.HTTP_200_OK)
def get_note(
    note_id: int, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_session)
):
    note = db.get(Note, note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    is_owner = False

    if note.user_id == current_user.id:
        is_owner = True
    elif getattr(note, "is_public", False):
        is_owner = False
    else:
        has_access = db.exec(
            select(CollectionItem.id)
            .join(Collection, Collection.id == CollectionItem.collection_id)
            .outerjoin(CollectionAccess, CollectionAccess.collection_id == Collection.id)
            .where(
                CollectionItem.item_type == "note",
                CollectionItem.item_id == str(note.id),
                or_(
                    Collection.visibility == "public",
                    CollectionAccess.user_id == current_user.id
                )
            )
        ).first()

        if has_access:
            is_owner = False
        else:
            raise HTTPException(status_code=404, detail="Note not found")

    response_data = note.model_dump()
    response_data["is_owner"] = is_owner

    return response_data

# ==========================================
# 4. UPDATE NOTE
# ==========================================
@router.patch("/notes/{note_id}", status_code=status.HTTP_200_OK)
def update_note(
    note_id: int,
    payload: NoteUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    note = db.get(Note, note_id)
    if not note or note.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Note not found")

    # FIXED: Replaced deprecated .dict() with .model_dump()
    update_data = payload.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(note, key, value)
        
    if "content_text" in update_data:
        note.word_count, note.snippet = calculate_note_metadata(note.content_text)

    # FIXED: Updated deprecated utcnow()
    note.updated_at = datetime.now(timezone.utc)
    db.add(note)
    db.commit()
    return {"note_id": note.id, "updated_at": note.updated_at}

# ==========================================
# 5. DELETE NOTE
# ==========================================
@router.delete("/notes/{note_id}", status_code=status.HTTP_200_OK)
def delete_note(
    note_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    note = db.exec(select(Note).where(Note.id == note_id, Note.user_id == current_user.id)).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
        
    flashcards = db.exec(select(Flashcard).where(Flashcard.note_id == note_id)).all()
    for card in flashcards:
        db.delete(card)

    db.delete(note)
    db.commit()
    return {"message": "Note and its flashcards deleted successfully"}

# ==========================================
# 6. GENERATE CARDS (AI)
# ==========================================
@router.post("/notes/{note_id}/generate-cards", status_code=status.HTTP_200_OK)
def generate_cards(
    note_id: int,
    payload: GenerateCardsRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    note = db.get(Note, note_id)
    if not note or note.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Note not found")

    estimated_cards = max(1, note.word_count // 60)
    
    # FIXED: We no longer pass `db` to the background task to avoid closed-session crashes
    background_tasks.add_task(
        generate_cards_bg, 
        note.id, 
        current_user.id, 
        payload.options
    )
    
    return {
        "set_id": f"temp_set_{note_id}", 
        "status": "processing", 
        "estimated_cards": estimated_cards
    }

# ==========================================
# 7. MANUALLY ADD CARD FROM HIGHLIGHT
# ==========================================
@router.post("/notes/{note_id}/cards", status_code=status.HTTP_201_CREATED)
def add_manual_card(
    note_id: int,
    payload: ManualCardCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    note = db.get(Note, note_id)
    if not note or note.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Note not found")

    study_set = db.exec(select(StudySet).where(StudySet.title == f"Set: {note.title}")).first()
    if not study_set:
        study_set = StudySet(user_id=current_user.id, title=f"Set: {note.title}", subject=note.subject)
        db.add(study_set)
        db.commit()
        db.refresh(study_set)

    new_card = Flashcard(
        study_set_id=study_set.id,
        note_id=note.id,
        question=payload.question,
        answer=payload.answer,
        subject=note.subject
    )
    db.add(new_card)
    
    study_set.card_count += 1
    note.card_count += 1
    
    db.commit()
    return {"message": "Card added to set successfully"}

# ==========================================
# 8. GET ALL CARDS FOR A NOTE
# ==========================================
@router.get("/notes/{note_id}/cards", status_code=status.HTTP_200_OK)
def get_note_cards(
    note_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    note = db.get(Note, note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    has_access = False

    if note.user_id == current_user.id:
        has_access = True
    elif getattr(note, "is_public", False):
        has_access = True
    else:
        shared = db.exec(
            select(CollectionItem.id)
            .join(Collection, Collection.id == CollectionItem.collection_id)
            .outerjoin(CollectionAccess, CollectionAccess.collection_id == Collection.id)
            .where(
                CollectionItem.item_type == "note",
                CollectionItem.item_id == str(note.id),
                or_(
                    Collection.visibility == "public",
                    CollectionAccess.user_id == current_user.id
                )
            )
        ).first()
        
        if shared:
            has_access = True

    if not has_access:
        raise HTTPException(status_code=404, detail="Note not found")

    cards = db.exec(select(Flashcard).where(Flashcard.note_id == note.id)).all()
    return {"cards": cards}