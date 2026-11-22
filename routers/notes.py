# routers/notes.py — v2
"""
What changed vs v1:
  * SECURITY: study sets were looked up by title only
    (`StudySet.title == "Set: <note title>"`) with no user filter, so two
    users with the same note title shared one set and wrote cards into
    each other's decks. Sets are now found through the note's own
    flashcards and always checked against the owner.
  * Deleting cards (note delete / regenerate) first deletes their
    FeynmanSession rows. Before, the ORM tried to NULL a NOT NULL
    card_id and the delete crashed with a 500 for any note that had
    been studied with Feynman.
  * Card generation has a real status on the note
    (idle | generating | ready | failed) + error message, so the app
    can show progress and failures instead of polling blind for 30 s.
  * "Regenerate" replaces the note's cards instead of piling duplicates.
  * AI output is validated (empty questions skipped, difficulty
    normalised, lengths capped) — one bad card no longer kills the run.
  * card_count / weak_card_count are recounted, never hand-incremented.
  * Titles are capped at 60 chars (the DB column is VARCHAR(60) — longer
    titles used to make every save fail).
  * List: total_count respects search/filter, content isn't sent,
    full subject list returned for chips, study_set_id included.
  * Timestamps serialised as UTC with "Z".
  * GET /notes/{id}/cards honours ?limit=.
  * Background title enrichment no longer overwrites a title the user
    already changed.
"""

from datetime import datetime, timedelta, timezone
import json
import os
from typing import Dict, List, Optional

from fastapi import (
    APIRouter, BackgroundTasks, Depends, File, HTTPException, Query,
    UploadFile, status,
)
from httpx import Timeout
from openai import OpenAI
from pydantic import BaseModel, Field
from sqlmodel import Session, func, or_, select

from database import engine, get_session
from models import (
    Collection, CollectionAccess, CollectionItem, FeynmanSession, Flashcard,
    Note, StudySet, User,
)
from routers.study import (
    MAX_EXTRACTED_CHARS,
    MAX_SUBJECT_LEN,
    MAX_TITLE_LEN,
    extract_text_from_file,
)
from security import get_current_user

router = APIRouter(tags=["Notes Section"])

# ==========================================
# CONFIG
# ==========================================

AI_MODEL = os.environ.get("NOTES_AI_MODEL", "gemini-3.8-flash")

NOTE_TITLE_MAX = 60          # Note.title is VARCHAR(60)
QUESTION_MAX = 200           # Flashcard.question is VARCHAR(200)
ANSWER_MAX = 400             # Flashcard.answer is VARCHAR(400)
MAX_PROMPT_CHARS = 30_000
MIN_WORDS_FOR_CARDS = 20
CARD_STALE_AFTER = timedelta(minutes=4)

CARD_IDLE = "idle"
CARD_GENERATING = "generating"
CARD_READY = "ready"
CARD_FAILED = "failed"

gemini_client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY", "your-gemini-api-key"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=Timeout(60.0, connect=10.0),
    max_retries=1,
)


# ==========================================
# PYDANTIC SCHEMAS
# ==========================================

class NoteCreate(BaseModel):
    title: str = "Untitled note"
    subject: str = "General"
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
    options: GenerateCardsOptions = GenerateCardsOptions()
    # True = delete this note's existing cards first ("Regenerate").
    replace: bool = False


class ManualCardCreate(BaseModel):
    question: str = Field(..., min_length=1, max_length=QUESTION_MAX)
    answer: str = Field(..., min_length=1, max_length=ANSWER_MAX)


# ==========================================
# HELPERS
# ==========================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    d = _as_utc(dt)
    return d.isoformat().replace("+00:00", "Z") if d else None


def _clean_title(value: Optional[str]) -> str:
    t = (value or "").strip()
    return (t or "Untitled note")[:NOTE_TITLE_MAX]


def _clean_subject(value: Optional[str]) -> str:
    s = (value or "").strip()
    return (s or "General")[:MAX_SUBJECT_LEN]


def calculate_note_metadata(text: str):
    text = text or ""
    word_count = len(text.split()) if text else 0
    flat = " ".join(text.split())
    snippet = flat[:77] + "..." if len(flat) > 80 else flat
    return word_count, snippet


def _card_dict(c: Flashcard) -> dict:
    return {
        "id": c.id,
        "study_set_id": c.study_set_id,
        "note_id": c.note_id,
        "question": c.question,
        "answer": c.answer,
        "subject": c.subject,
        "difficulty": getattr(c.difficulty, "value", c.difficulty),
        "is_weak": c.is_weak,
    }


def _note_dict(
    note: Note,
    *,
    is_owner: Optional[bool] = None,
    study_set_id: Optional[int] = None,
    include_content: bool = True,
) -> dict:
    data = {
        "id": note.id,
        "title": note.title,
        "subject": note.subject,
        "word_count": note.word_count,
        "card_count": note.card_count,
        "weak_card_count": note.weak_card_count,
        "has_canvas": note.has_canvas,
        "snippet": note.snippet,
        "is_public": note.is_public,
        "created_at": _iso(note.created_at),
        "updated_at": _iso(note.updated_at),
        "card_status": note.card_status or CARD_IDLE,
        "card_error": note.card_error,
        "study_set_id": study_set_id,
    }
    if include_content:
        data["content_text"] = note.content_text
        data["content_html"] = note.content_html
    if is_owner is not None:
        data["is_owner"] = is_owner
    return data


def _note_set_id(db: Session, note: Note) -> Optional[int]:
    return db.exec(
        select(Flashcard.study_set_id)
        .where(Flashcard.note_id == note.id)
        .limit(1)
    ).first()


def _get_or_create_note_set(db: Session, note: Note, user_id: int) -> StudySet:
    """The note's deck = the set its cards already live in (owner-checked).
    Otherwise a fresh set. Never matched by title."""
    set_id = _note_set_id(db, note)
    if set_id:
        existing = db.get(StudySet, set_id)
        if existing and existing.user_id == user_id:
            return existing

    study_set = StudySet(
        user_id=user_id,
        title=f"Set: {note.title}",
        subject=note.subject,
        card_count=0,
    )
    db.add(study_set)
    db.flush()
    return study_set


def _delete_cards(db: Session, cards: List[Flashcard]) -> None:
    """Delete FeynmanSessions first: card_id is NOT NULL, so letting the
    ORM null it out crashes the whole delete."""
    if not cards:
        return
    ids = [c.id for c in cards]
    for s in db.exec(select(FeynmanSession).where(FeynmanSession.card_id.in_(ids))).all():
        db.delete(s)
    db.flush()
    for c in cards:
        db.delete(c)
    db.flush()


def _recount_note(db: Session, note: Note) -> None:
    db.flush()
    flags = db.exec(select(Flashcard.is_weak).where(Flashcard.note_id == note.id)).all()
    note.card_count = len(flags)
    note.weak_card_count = sum(1 for f in flags if f)
    db.add(note)


def _recount_set(db: Session, study_set: Optional[StudySet]) -> None:
    if study_set is None:
        return
    db.flush()
    flags = db.exec(
        select(Flashcard.is_weak).where(Flashcard.study_set_id == study_set.id)
    ).all()
    study_set.card_count = len(flags)
    study_set.weak_cards_count = sum(1 for f in flags if f)
    db.add(study_set)


def _expire_stale_generation(note: Note) -> bool:
    if note.card_status != CARD_GENERATING:
        return False
    started = _as_utc(note.card_status_at)
    if started is None or utc_now() - started > CARD_STALE_AFTER:
        note.card_status = CARD_FAILED
        note.card_error = "Card generation timed out. Try again."
        return True
    return False


def _has_read_access(db: Session, note: Note, user: User) -> Optional[bool]:
    """Returns is_owner (True/False) or None when the user can't see it."""
    if note.user_id == user.id:
        return True
    if getattr(note, "is_public", False):
        return False
    shared = db.exec(
        select(CollectionItem.id)
        .join(Collection, Collection.id == CollectionItem.collection_id)
        .outerjoin(CollectionAccess, CollectionAccess.collection_id == Collection.id)
        .where(
            CollectionItem.item_type == "note",
            CollectionItem.item_id == str(note.id),
            or_(
                Collection.visibility == "public",
                CollectionAccess.user_id == user.id,
            ),
        )
    ).first()
    return False if shared else None


def _get_owned_note(db: Session, note_id: int, user: User) -> Note:
    note = db.get(Note, note_id)
    if not note or note.user_id != user.id:
        raise HTTPException(status_code=404, detail="Note not found")
    return note


_DIFFICULTIES = {"easy", "medium", "hard"}


def _normalise_cards(raw_cards) -> List[dict]:
    if not isinstance(raw_cards, list):
        return []
    out, seen = [], set()
    for c in raw_cards:
        if not isinstance(c, dict):
            continue
        q = " ".join(str(c.get("question") or "").split())
        a = " ".join(str(c.get("answer") or "").split())
        if not q or not a:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        diff = str(c.get("difficulty") or "medium").strip().lower()
        out.append({
            "question": q[:QUESTION_MAX],
            "answer": a[:ANSWER_MAX],
            "difficulty": diff if diff in _DIFFICULTIES else "medium",
        })
    return out


# ==========================================
# BACKGROUND TASKS
# ==========================================

def enrich_note_metadata(note_id: int, sample_text: str, original_title: str):
    """Nicer AI title/subject for uploaded docs. Only touches fields the
    user hasn't changed in the meantime."""
    with Session(engine) as db:
        note = db.get(Note, note_id)
        if not note:
            return
        try:
            system_prompt = """
            You are analyzing a study document. Return ONLY valid JSON matching:
            {"title": "short engaging title, max 6 words", "subject": "academic subject or General"}
            No markdown. Just raw JSON.
            """
            response = gemini_client.chat.completions.create(
                model=AI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": sample_text},
                ],
                response_format={"type": "json_object"},
            )
            ai_data = json.loads(response.choices[0].message.content)

            db.refresh(note)
            changed = False
            if ai_data.get("title") and note.title == original_title:
                note.title = _clean_title(str(ai_data["title"]))[:MAX_TITLE_LEN]
                changed = True
            if ai_data.get("subject") and note.subject == "General":
                note.subject = _clean_subject(str(ai_data["subject"]))
                changed = True
            if changed:
                note.updated_at = utc_now()
                db.add(note)
                db.commit()
            print(f"Enriched note {note_id}: {note.title!r} / {note.subject!r}")
        except Exception as e:
            print(f"Note enrichment failed (non-fatal): {e}")


def _mark_cards_failed(db: Session, note_id: int, message: str) -> None:
    try:
        db.rollback()
        note = db.get(Note, note_id)
        if note:
            note.card_status = CARD_FAILED
            note.card_error = message[:300]
            db.add(note)
            db.commit()
    except Exception as e:
        print(f"[cards-bg] could not mark failed note={note_id}: {e}")
        db.rollback()


def generate_cards_bg(note_id: int, user_id: int, options: GenerateCardsOptions, replace: bool):
    print(f"[cards-bg] START note={note_id} replace={replace}")
    with Session(engine) as db:
        note = db.get(Note, note_id)
        if not note:
            return
        text = (note.content_text or "").strip()
        if not text:
            _mark_cards_failed(db, note_id, "This note is empty.")
            return

        focus = "Generate both definition-style cards and conceptual questions."
        if options.definitions and not options.questions:
            focus = "Generate ONLY definition-style vocabulary cards (Term -> Definition)."
        elif options.questions and not options.definitions:
            focus = "Generate ONLY conceptual question/answer cards. Do not generate simple vocabulary definitions."

        target = max(5, min(30, (note.word_count or 0) // 60))

        prompt = f"""
        You are an expert tutor. Create about {target} high-quality flashcards from the study notes below.
        {focus}
        Questions must be under 180 characters and answers under 350 characters.
        Every card must be answerable from the notes.

        Return ONLY raw, valid JSON:
        {{
          "cards": [
            {{ "question": "front of the card", "answer": "back of the card", "difficulty": "easy" | "medium" | "hard" }}
          ]
        }}

        NOTES TEXT:
        {text[:MAX_PROMPT_CHARS]}
        """

        try:
            response = gemini_client.chat.completions.create(
                model=AI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an API that only returns valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            ai_data = json.loads(response.choices[0].message.content)
            cards = _normalise_cards(ai_data.get("cards") if isinstance(ai_data, dict) else None)
        except Exception as e:
            print(f"[cards-bg] AI FAILURE note={note_id}: {type(e).__name__}: {e}")
            _mark_cards_failed(db, note_id, "The AI couldn't read this note. Try again.")
            return

        if not cards:
            _mark_cards_failed(db, note_id, "The AI didn't return any usable cards. Try again.")
            return

        try:
            note = db.get(Note, note_id)
            study_set = _get_or_create_note_set(db, note, user_id)

            if replace:
                old = db.exec(select(Flashcard).where(Flashcard.note_id == note.id)).all()
                old_sets = {c.study_set_id for c in old}
                _delete_cards(db, old)
                for sid in old_sets:
                    if sid != study_set.id:
                        _recount_set(db, db.get(StudySet, sid))

            for c in cards:
                db.add(Flashcard(
                    study_set_id=study_set.id,
                    note_id=note.id,
                    question=c["question"],
                    answer=c["answer"],
                    subject=note.subject,
                    difficulty=c["difficulty"],
                ))

            _recount_note(db, note)
            _recount_set(db, study_set)
            note.card_status = CARD_READY
            note.card_error = None
            db.add(note)
            db.commit()
            print(f"[cards-bg] DONE note={note_id}: {len(cards)} cards")
        except Exception as e:
            print(f"[cards-bg] SAVE FAILURE note={note_id}: {type(e).__name__}: {e}")
            _mark_cards_failed(db, note_id, "Cards were generated but couldn't be saved. Try again.")


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
    db: Session = Depends(get_session),
):
    conditions = [Note.user_id == current_user.id]
    s = (search or "").strip()
    if s:
        conditions.append(or_(Note.title.icontains(s), Note.content_text.icontains(s)))
    if filter and filter.lower() != "all":
        conditions.append(func.lower(Note.subject) == filter.lower())

    query = select(Note).where(*conditions)
    if sort == "created":
        query = query.order_by(Note.created_at.desc())
    elif sort == "a-z":
        query = query.order_by(Note.title.asc())
    elif sort == "most-cards":
        query = query.order_by(Note.card_count.desc())
    else:
        query = query.order_by(Note.updated_at.desc())

    offset = (page - 1) * limit
    notes = db.exec(query.offset(offset).limit(limit)).all()

    # FIX: count with the SAME filters, or has_more is wrong.
    total_count = db.exec(select(func.count(Note.id)).where(*conditions)).one()

    note_ids = [n.id for n in notes]
    set_map: Dict[int, int] = {}
    if note_ids:
        for nid, sid in db.exec(
            select(Flashcard.note_id, Flashcard.study_set_id)
            .where(Flashcard.note_id.in_(note_ids))
            .distinct()
        ).all():
            set_map.setdefault(nid, sid)

    changed = False
    for n in notes:
        if _expire_stale_generation(n):
            db.add(n)
            changed = True
    if changed:
        db.commit()

    subjects = db.exec(
        select(Note.subject).where(Note.user_id == current_user.id).distinct()
    ).all()

    return {
        "notes": [
            _note_dict(n, study_set_id=set_map.get(n.id), include_content=False)
            for n in notes
        ],
        "subjects": sorted({x for x in subjects if x}, key=str.lower),
        "total_count": total_count,
        "page": page,
        "has_more": total_count > offset + len(notes),
    }


# ==========================================
# 1B. UPLOAD DOCUMENT → CREATE NOTE (FAST)
# ==========================================

@router.post("/notes/upload", status_code=status.HTTP_201_CREATED)
def upload_document_as_note(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    raw_text = extract_text_from_file(file)
    if not raw_text or len(raw_text.strip()) < 30:
        raise HTTPException(
            status_code=400,
            detail="Could not extract enough readable text from this file.",
        )
    if len(raw_text) > MAX_EXTRACTED_CHARS:
        raw_text = raw_text[:MAX_EXTRACTED_CHARS]

    fallback_title = _clean_title((file.filename or "Untitled").rsplit(".", 1)[0])
    word_count, snippet = calculate_note_metadata(raw_text)

    new_note = Note(
        user_id=current_user.id,
        title=fallback_title,
        subject="General",
        content_text=raw_text,
        word_count=word_count,
        snippet=snippet,
    )
    db.add(new_note)
    db.commit()
    db.refresh(new_note)

    background_tasks.add_task(
        enrich_note_metadata, new_note.id, raw_text[:3000], fallback_title,
    )

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
    db: Session = Depends(get_session),
):
    word_count, snippet = calculate_note_metadata(payload.content_text)
    new_note = Note(
        user_id=current_user.id,
        title=_clean_title(payload.title),
        subject=_clean_subject(payload.subject),
        content_text=payload.content_text or "",
        content_html=payload.content_html,
        word_count=word_count,
        snippet=snippet,
    )
    db.add(new_note)
    db.commit()
    db.refresh(new_note)
    return {
        "note_id": new_note.id,
        "title": new_note.title,
        "created_at": _iso(new_note.created_at),
        "updated_at": _iso(new_note.updated_at),
    }


# ==========================================
# 3. GET NOTE DETAIL
# ==========================================

@router.get("/notes/{note_id}", status_code=status.HTTP_200_OK)
def get_note(
    note_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    note = db.get(Note, note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    is_owner = _has_read_access(db, note, current_user)
    if is_owner is None:
        raise HTTPException(status_code=404, detail="Note not found")

    if _expire_stale_generation(note):
        db.add(note)
        db.commit()
        db.refresh(note)

    return _note_dict(note, is_owner=is_owner, study_set_id=_note_set_id(db, note))


# ==========================================
# 4. UPDATE NOTE
# ==========================================

@router.patch("/notes/{note_id}", status_code=status.HTTP_200_OK)
def update_note(
    note_id: int,
    payload: NoteUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    note = _get_owned_note(db, note_id, current_user)
    data = payload.model_dump(exclude_unset=True)

    if "title" in data:
        note.title = _clean_title(data["title"])
        set_id = _note_set_id(db, note)
        if set_id:
            study_set = db.get(StudySet, set_id)
            if study_set and study_set.user_id == current_user.id:
                study_set.title = f"Set: {note.title}"
                db.add(study_set)
    if "subject" in data:
        note.subject = _clean_subject(data["subject"])
    if "content_text" in data:
        note.content_text = data["content_text"] or ""
        note.word_count, note.snippet = calculate_note_metadata(note.content_text)
    if "content_html" in data:
        note.content_html = data["content_html"]
    if data.get("is_public") is not None:
        note.is_public = data["is_public"]

    note.updated_at = utc_now()
    db.add(note)
    db.commit()
    db.refresh(note)
    return {
        "note_id": note.id,
        "title": note.title,
        "subject": note.subject,
        "word_count": note.word_count,
        "is_public": note.is_public,
        "updated_at": _iso(note.updated_at),
    }


# ==========================================
# 5. DELETE NOTE
# ==========================================

@router.delete("/notes/{note_id}", status_code=status.HTTP_200_OK)
def delete_note(
    note_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    note = _get_owned_note(db, note_id, current_user)

    cards = db.exec(select(Flashcard).where(Flashcard.note_id == note.id)).all()
    set_ids = {c.study_set_id for c in cards}
    _delete_cards(db, cards)

    # Keep deck counters right; drop decks this note emptied.
    for sid in set_ids:
        study_set = db.get(StudySet, sid)
        if not study_set or study_set.user_id != current_user.id:
            continue
        _recount_set(db, study_set)
        if study_set.card_count == 0:
            for s in db.exec(
                select(FeynmanSession).where(FeynmanSession.study_set_id == sid)
            ).all():
                db.delete(s)
            db.flush()
            db.delete(study_set)

    db.delete(note)
    db.commit()
    return {"message": "Note and its flashcards deleted successfully"}


# ==========================================
# 6. GENERATE CARDS (AI)
# ==========================================

@router.post("/notes/{note_id}/generate-cards", status_code=status.HTTP_200_OK)
def generate_cards(
    note_id: int,
    background_tasks: BackgroundTasks,
    payload: GenerateCardsRequest = GenerateCardsRequest(),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    note = _get_owned_note(db, note_id, current_user)

    if (note.word_count or 0) < MIN_WORDS_FOR_CARDS:
        raise HTTPException(
            status_code=400,
            detail=f"Write at least {MIN_WORDS_FOR_CARDS} words before generating cards.",
        )

    _expire_stale_generation(note)
    if note.card_status == CARD_GENERATING:
        raise HTTPException(status_code=409, detail="Cards are already being generated.")

    note.card_status = CARD_GENERATING
    note.card_error = None
    note.card_status_at = utc_now()
    db.add(note)
    db.commit()

    background_tasks.add_task(
        generate_cards_bg, note.id, current_user.id, payload.options, payload.replace,
    )

    return {
        "status": CARD_GENERATING,
        "study_set_id": _note_set_id(db, note),
        "card_count": note.card_count,
        "estimated_cards": max(5, min(30, (note.word_count or 0) // 60)),
    }


# ==========================================
# 7. MANUALLY ADD CARD
# ==========================================

@router.post("/notes/{note_id}/cards", status_code=status.HTTP_201_CREATED)
def add_manual_card(
    note_id: int,
    payload: ManualCardCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    note = _get_owned_note(db, note_id, current_user)
    question, answer = payload.question.strip(), payload.answer.strip()
    if not question or not answer:
        raise HTTPException(status_code=400, detail="Question and answer can't be empty.")

    study_set = _get_or_create_note_set(db, note, current_user.id)
    card = Flashcard(
        study_set_id=study_set.id,
        note_id=note.id,
        question=question,
        answer=answer,
        subject=note.subject,
    )
    db.add(card)
    _recount_note(db, note)
    _recount_set(db, study_set)
    db.commit()
    db.refresh(card)
    return {"message": "Card added to set successfully", "card": _card_dict(card)}


# ==========================================
# 8. GET CARDS FOR A NOTE
# ==========================================

@router.get("/notes/{note_id}/cards", status_code=status.HTTP_200_OK)
def get_note_cards(
    note_id: int,
    limit: Optional[int] = Query(None, ge=1, le=500),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    note = db.get(Note, note_id)
    if not note or _has_read_access(db, note, current_user) is None:
        raise HTTPException(status_code=404, detail="Note not found")

    query = select(Flashcard).where(Flashcard.note_id == note.id).order_by(Flashcard.id)
    if limit:
        query = query.limit(limit)
    cards = db.exec(query).all()
    return {"cards": [_card_dict(c) for c in cards], "total": note.card_count}