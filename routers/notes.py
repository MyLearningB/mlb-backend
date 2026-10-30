# routers/notes.py
from fastapi import (
    APIRouter, Depends, HTTPException, status,
    Query, BackgroundTasks, UploadFile, File,
)
from sqlmodel import Session, select, func, or_
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timezone
import json
import os
from openai import OpenAI

# Database, Models, and Authentication
from database import get_session, engine
from security import get_current_user
from models import CollectionAccess, CollectionItem, Collection, User, Note, Flashcard, StudySet

# Reuse the file-extraction guardrails from study.py so a 5MB cap
# and PDF/DOCX/TXT parsing aren't duplicated.
from routers.study import (
    extract_text_from_file,
    MAX_EXTRACTED_CHARS,
    MAX_TITLE_LEN,
    MAX_SUBJECT_LEN,
)

router = APIRouter(tags=["Notes Section"])

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

def generate_cards_bg(note_id: int, user_id: int, options: GenerateCardsOptions):
    print(f"Starting Gemini flashcard generation for note {note_id}...")
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

            study_set = db.exec(
                select(StudySet).where(StudySet.title == f"Set: {note.title}")
            ).first()
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
    total_count = db.exec(
        select(func.count(Note.id)).where(Note.user_id == current_user.id)
    ).one()

    return {
        "notes": notes,
        "total_count": total_count,
        "has_more": total_count > (offset + limit),
    }

# ==========================================
# 1B. UPLOAD DOCUMENT → CREATE NOTE
# ==========================================
@router.post("/notes/upload", status_code=status.HTTP_201_CREATED)
def upload_document_as_note(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """
    Upload PDF/DOCX/TXT → extract text → persist as a Note.

    Flow:
      1. Extract + size-guard via the shared helper in study.py
      2. Ask Gemini for a short title + subject (best-effort, non-fatal)
      3. Save a Note — NOT a StudySet
      4. Client can then call POST /notes/{id}/generate-cards

    Everything downstream (generate-cards, editor, delete) works
    unchanged because we're writing to the same Note model.
    """
    raw_text = extract_text_from_file(file)

    if not raw_text or len(raw_text.strip()) < 30:
        raise HTTPException(
            status_code=400,
            detail="Could not extract enough readable text from this file.",
        )

    if len(raw_text) > MAX_EXTRACTED_CHARS:
        raw_text = raw_text[:MAX_EXTRACTED_CHARS]

    fallback_title = (file.filename or "Untitled").rsplit(".", 1)[0]
    deck_title = fallback_title[:MAX_TITLE_LEN]
    deck_subject = "General"

    # Best-effort title/subject suggestion. Never blocks the upload.
    try:
        system_prompt = """
        You are analyzing a study document. Return ONLY valid JSON matching:
        {"title": "short engaging title, max 6 words", "subject": "academic subject or General"}
        No markdown. Just raw JSON.
        """
        response = gemini_client.chat.completions.create(
            model="gemini-3.8-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_text[:3000]},  # classification sample
            ],
            response_format={"type": "json_object"},
        )
        ai_data = json.loads(response.choices[0].message.content)
        deck_title = (ai_data.get("title") or fallback_title)[:MAX_TITLE_LEN]
        deck_subject = (ai_data.get("subject") or "General")[:MAX_SUBJECT_LEN]
    except Exception as e:
        print(f"AI title/subject suggestion failed, using filename: {e}")

    word_count, snippet = calculate_note_metadata(raw_text)

    new_note = Note(
        user_id=current_user.id,
        title=deck_title,
        subject=deck_subject,
        content_text=raw_text,
        word_count=word_count,
        snippet=snippet,
    )
    db.add(new_note)
    db.commit()
    db.refresh(new_note)

    return {
        "note_id": new_note.id,
        "title": new_note.title,
        "subject": new_note.subject,
        "word_count": word_count,
        "status": "success",
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
        snippet=snippet,
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

    update_data = payload.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(note, key, value)

    if "content_text" in update_data:
        note.word_count, note.snippet = calculate_note_metadata(note.content_text)

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
    note = db.exec(
        select(Note).where(Note.id == note_id, Note.user_id == current_user.id)
    ).first()
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

    # Create the parent StudySet synchronously so we can return its real
    # ID. The background task then just appends cards to it, and the
    # client polls GET /study/sets until card_count > 0.
    study_set = db.exec(
        select(StudySet).where(StudySet.title == f"Set: {note.title}")
    ).first()
    if not study_set:
        study_set = StudySet(
            user_id=current_user.id,
            title=f"Set: {note.title}",
            subject=note.subject,
            card_count=0,
        )
        db.add(study_set)
        db.commit()
        db.refresh(study_set)

    background_tasks.add_task(
        generate_cards_bg,
        note.id,
        current_user.id,
        payload.options,
    )

    return {
        "id": study_set.id,
        "status": "processing",
        "card_count": study_set.card_count,
        "estimated_cards": estimated_cards,
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

    study_set = db.exec(
        select(StudySet).where(StudySet.title == f"Set: {note.title}")
    ).first()
    if not study_set:
        study_set = StudySet(
            user_id=current_user.id,
            title=f"Set: {note.title}",
            subject=note.subject,
        )
        db.add(study_set)
        db.commit()
        db.refresh(study_set)

    new_card = Flashcard(
        study_set_id=study_set.id,
        note_id=note.id,
        question=payload.question,
        answer=payload.answer,
        subject=note.subject,
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