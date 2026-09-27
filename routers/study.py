# routers/study.py
import io
import os
import json
import uuid
from typing import List, Optional
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Header
from sqlmodel import Session, select, or_
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# Database, Models, Authentication, and Helpers
from database import get_session
from security import get_current_user
from models import User, StudySet, Flashcard, FeynmanSession, DailyActivity, Pet, Collection, CollectionItem, CollectionAccess

# ✨ IMPORT THE STREAK HELPER ✨
from routers.profile import update_user_streak

# 👉 NEW: IMPORT THE CO-OP XP HOOK
from routers.community import contribute_to_group_quests

load_dotenv()

router = APIRouter(prefix="/study", tags=["Study Tab"])

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "fallback-key-for-dev"),
    base_url="https://api.deepseek.com"
)

# ==========================================
# UPLOAD LIMITS
# ==========================================

MAX_UPLOAD_BYTES = 5 * 1024 * 1024           # 5 MB hard cap
MAX_EXTRACTED_CHARS = 25000                  # ~6k tokens going to DeepSeek
MAX_TITLE_LEN = 60
MAX_SUBJECT_LEN = 40


# ==========================================
# PYDANTIC SCHEMAS
# ==========================================

class SwipeResponse(BaseModel):
    card_id: int
    response: str
    response_time_ms: Optional[int] = None

class FlashcardCompleteRequest(BaseModel):
    cards_correct: int
    cards_incorrect: int
    duration_seconds: int

class FeynmanStartRequest(BaseModel):
    set_id: int
    card_id: int

class FeynmanMessageRequest(BaseModel):
    message: str
    voice_transcript: Optional[str] = None

class FeynmanCompleteRequest(BaseModel):
    final_score: int
    duration_seconds: int


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_local_now(timezone_str: str = "UTC") -> datetime:
    """Helper to ensure streaks are calculated in the user's local timezone"""
    try:
        tz = ZoneInfo(timezone_str)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo("UTC")
    return datetime.now(tz)


def extract_text_from_file(file: UploadFile) -> str:
    """Reads uploaded files directly from RAM without hitting the disk.

    Enforces MAX_UPLOAD_BYTES BEFORE parsing, so a malicious or
    accidental 100MB upload can't blow up the container.
    """
    content = file.file.read()
    filename = (file.filename or "").lower()

    # --------------------------------------------------------
    # SIZE GUARD
    # --------------------------------------------------------
    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail="That file is empty.",
        )

    if len(content) > MAX_UPLOAD_BYTES:
        size_mb = len(content) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large ({size_mb:.1f}MB). "
                f"Max {MAX_UPLOAD_BYTES // (1024 * 1024)}MB."
            ),
        )

    # --------------------------------------------------------
    # PARSE
    # --------------------------------------------------------
    if filename.endswith(".txt"):
        return content.decode("utf-8", errors="ignore")

    elif filename.endswith(".pdf"):
        import pypdf
        pdf_reader = pypdf.PdfReader(io.BytesIO(content))
        text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
        return text

    elif filename.endswith((".docx", ".doc")):
        import docx
        doc = docx.Document(io.BytesIO(content))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text

    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload PDF, TXT, or DOCX.",
        )


def _recompute_weak_cards_count(db: Session, study_set_id: int) -> int:
    """Recount weak cards for a set and update the denormalized field.

    Called any time a card's `is_weak` flag changes so the mode
    selection screen's 'X weak' badge stays truthful.
    """
    cards = db.exec(
        select(Flashcard).where(Flashcard.study_set_id == study_set_id)
    ).all()
    return sum(1 for c in cards if c.is_weak)


# ==========================================
# 6A: MODE SELECTION & SET ENDPOINTS
# ==========================================

@router.get("/sets")
def get_user_study_sets(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """Fetches all study sets belonging strictly to the authenticated user.

    Sorts newest first by ID so freshly uploaded/generated sets
    always surface at the top of the mode selection screen.
    """
    statement = (
        select(StudySet)
        .where(StudySet.user_id == current_user.id)
        .order_by(StudySet.id.desc())
    )
    return db.exec(statement).all()


@router.get("/sets/{set_id}")
def get_study_set(
    set_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    study_set = db.get(StudySet, set_id)
    if not study_set:
        raise HTTPException(status_code=404, detail="Study set not found")

    is_owner = False

    if study_set.user_id == current_user.id:
        is_owner = True
    else:
        has_access = db.exec(
            select(CollectionItem.id)
            .join(Collection, Collection.id == CollectionItem.collection_id)
            .outerjoin(CollectionAccess, CollectionAccess.collection_id == Collection.id)
            .where(
                CollectionItem.item_type == "set",
                CollectionItem.item_id == str(study_set.id),
                or_(
                    Collection.visibility == "public",
                    CollectionAccess.user_id == current_user.id
                )
            )
        ).first()

        if has_access:
            is_owner = False
        else:
            raise HTTPException(status_code=404, detail="Study set not found")

    cards = db.exec(select(Flashcard).where(Flashcard.study_set_id == study_set.id)).all()

    response_data = study_set.model_dump()
    response_data["is_owner"] = is_owner
    response_data["cards"] = cards

    return response_data


@router.delete("/sets/{set_id}", status_code=200)
def delete_study_set(
    set_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    study_set = db.get(StudySet, set_id)
    if not study_set or study_set.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Study set not found")

    feynman_sessions = db.exec(select(FeynmanSession).where(FeynmanSession.study_set_id == set_id)).all()
    for session in feynman_sessions:
        db.delete(session)

    flashcards = db.exec(select(Flashcard).where(Flashcard.study_set_id == set_id)).all()
    for card in flashcards:
        db.delete(card)

    db.delete(study_set)
    db.commit()

    return {"message": "Study set and all associated data deleted successfully"}


# ========================================================
# ✨ 6C: AI DOCUMENT UPLOAD & EXTRACTION ENDPOINT ✨
# ========================================================

@router.post("/upload", status_code=201)
def upload_and_generate_study_set(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """Takes a raw PDF/DOCX/TXT, extracts the text, and commands DeepSeek to turn it into flashcards.

    Synchronous: DeepSeek runs inside the request. The client
    has a 90s timeout to accommodate cold starts and 5MB docs.
    """

    # 1. Extract raw text from the uploaded binary (with size guard)
    raw_text = extract_text_from_file(file)

    if not raw_text or len(raw_text.strip()) < 30:
        raise HTTPException(
            status_code=400,
            detail="Could not extract enough readable text from this file.",
        )

    # Guardrail: Cap the text sent to DeepSeek
    if len(raw_text) > MAX_EXTRACTED_CHARS:
        raw_text = raw_text[:MAX_EXTRACTED_CHARS]

    system_prompt = """
    You are an expert AI professor. Your job is to read the provided text and turn it into a high-yield study set.

    Analyze the text and output ONLY valid JSON matching this exact schema:
    {
      "title": "A short, engaging title for the deck (max 5 words)",
      "subject": "The academic subject (e.g. Biology, Macroeconomics, Literature, or General)",
      "flashcards": [
        {
          "question": "Clear, direct conceptual question",
          "answer": "Concise, highly accurate technical answer"
        }
      ]
    }

    Rules:
    - Generate between 6 and 15 flashcards depending on the document's density.
    - Focus on core concepts, definitions, formulas, or pivotal relationships.
    - Do not include markdown wrappers around the JSON.
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the source text:\n\n{raw_text}"}
            ],
            response_format={"type": "json_object"}
        )

        ai_data = json.loads(response.choices[0].message.content)

        fallback_title = (file.filename or "Untitled").rsplit(".", 1)[0]
        deck_title = (ai_data.get("title") or fallback_title)[:MAX_TITLE_LEN]
        deck_subject = (ai_data.get("subject") or "General")[:MAX_SUBJECT_LEN]
        generated_cards = ai_data.get("flashcards", [])

        if not generated_cards:
            raise ValueError("LLM returned an empty flashcards array.")

    except HTTPException:
        # Let our own HTTP errors (e.g. 400 from size check) bubble up
        raise
    except Exception as e:
        print(f"DeepSeek Parsing Error: {e}")
        raise HTTPException(
            status_code=500,
            detail="The AI couldn't parse the concepts out of this document. Try a cleaner PDF."
        )

    # 2. Save the new parent StudySet to the DB.
    # `last_studied` is intentionally left None — it should reflect
    # real study activity, not upload time.
    new_study_set = StudySet(
        user_id=current_user.id,
        title=deck_title,
        subject=deck_subject,
        card_count=len(generated_cards),
        weak_cards_count=0,
    )
    db.add(new_study_set)
    db.commit()
    db.refresh(new_study_set)

    # 3. Save all children Flashcards attached to that set
    for card_data in generated_cards:
        new_card = Flashcard(
            study_set_id=new_study_set.id,
            question=card_data.get("question", "Undefined Question"),
            answer=card_data.get("answer", "Undefined Answer"),
            subject=deck_subject,
            is_weak=False
        )
        db.add(new_card)

    db.commit()

    return {
        "id": new_study_set.id,
        "title": new_study_set.title,
        "card_count": new_study_set.card_count,
        "status": "success"
    }


# ==========================================
# 6B: STANDARD FLASHCARDS ENDPOINTS
# ==========================================

@router.get("/sets/{set_id}/cards")
def get_flashcards(
    set_id: int,
    order: str = "spaced_repetition",
    limit: int = 40,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    study_set = db.get(StudySet, set_id)
    if not study_set:
        raise HTTPException(status_code=404, detail="Study set not found")

    has_access = False

    if study_set.user_id == current_user.id:
        has_access = True
    else:
        shared = db.exec(
            select(CollectionItem.id)
            .join(Collection, Collection.id == CollectionItem.collection_id)
            .outerjoin(CollectionAccess, CollectionAccess.collection_id == Collection.id)
            .where(
                CollectionItem.item_type == "set",
                CollectionItem.item_id == str(study_set.id),
                or_(
                    Collection.visibility == "public",
                    CollectionAccess.user_id == current_user.id
                )
            )
        ).first()

        if shared:
            has_access = True

    if not has_access:
        raise HTTPException(status_code=404, detail="Study set not found")

    query = select(Flashcard).where(Flashcard.study_set_id == set_id)

    if order == "spaced_repetition":
        query = query.order_by(Flashcard.is_weak.desc())

    cards = db.exec(query.limit(limit)).all()
    session_id = str(uuid.uuid4())

    return {
        "session_id": session_id,
        "total_cards": len(cards),
        "cards": cards
    }


@router.post("/sessions/{session_id}/responses")
def record_swipe(
    session_id: str,
    payload: SwipeResponse,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    card = db.get(Flashcard, payload.card_id)
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")

    if payload.response == "correct":
        card.is_weak = False
        xp_earned = 5
    elif payload.response == "incorrect":
        card.is_weak = True
        xp_earned = 1
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid response type. Use 'correct' or 'incorrect'."
        )

    db.add(card)

    # --------------------------------------------------------
    # KEEP THE PARENT'S weak_cards_count IN SYNC
    # --------------------------------------------------------
    # Without this, the "X weak" badge on the mode selection
    # screen never lights up.
    study_set = db.get(StudySet, card.study_set_id)
    if study_set:
        study_set.weak_cards_count = _recompute_weak_cards_count(
            db, study_set.id
        )
        db.add(study_set)

    db.commit()

    return {
        "is_weak": card.is_weak,
        "xp_earned": xp_earned
    }


@router.post("/sessions/{session_id}/complete")
def complete_flashcard_session(
    session_id: str,
    payload: FlashcardCompleteRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
    x_timezone: str = Header("UTC")
):
    base_xp = (payload.cards_correct * 5) + (payload.cards_incorrect * 1)
    pet_xp_awarded = int(base_xp * 0.5)

    pet = db.exec(select(Pet).where(Pet.user_id == current_user.id)).first()
    pet_type = "nova"
    pet_level = 1

    if pet:
        pet.xp += pet_xp_awarded
        db.add(pet)
        pet_type = pet.pet_type
        pet_level = pet.level

    today_str = get_local_now(x_timezone).date().isoformat()
    daily_activity = db.exec(select(DailyActivity).where(
        DailyActivity.user_id == current_user.id,
        DailyActivity.date == today_str
    )).first()

    if daily_activity:
        daily_activity.xp_earned += base_xp
    else:
        daily_activity = DailyActivity(
            user_id=current_user.id,
            date=today_str,
            xp_earned=base_xp
        )

    db.add(daily_activity)
    update_user_streak(current_user, db)

    # Funnel the earned XP into active Co-op Quests
    contribute_to_group_quests(user_id=current_user.id, xp_amount=base_xp, db=db)

    db.commit()

    return {
        "session_summary": {
            "xp_earned": base_xp,
            "pet_xp": pet_xp_awarded,
            "pet_type": pet_type,
            "pet_level": pet_level,
            "streak_updated": True,
            "next_suggestions": [
                {"label": "Tackle your weak cards", "action_type": "review_weak"},
                {"label": "Deep dive with Feynman", "action_type": "feynman_mode"}
            ]
        }
    }


# ==========================================
# 7: FEYNMAN MODE (AI CHAT) ENDPOINTS
# ==========================================

@router.post("/feynman/start")
def start_feynman_session(
    payload: FeynmanStartRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    card = db.get(Flashcard, payload.card_id)
    if not card:
        raise HTTPException(status_code=404, detail="Flashcard not found")

    feynman_session = FeynmanSession(
        user_id=current_user.id,
        study_set_id=payload.set_id,
        card_id=payload.card_id,
        comprehension_score=0
    )
    db.add(feynman_session)
    db.commit()
    db.refresh(feynman_session)

    first_prompt = (
        f"Explain '{card.question}' as if I've never heard of it before. "
        f"Break it down simply!"
    )

    return {
        "session_id": feynman_session.id,
        "first_prompt": first_prompt,
        "card_concept": card.subject
    }


@router.post("/feynman/{session_id}/message")
def feynman_chat_message(
    session_id: int,
    payload: FeynmanMessageRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    feynman_session = db.get(FeynmanSession, session_id)

    if not feynman_session or feynman_session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")

    card = db.get(Flashcard, feynman_session.card_id)
    user_input = payload.voice_transcript if payload.voice_transcript else payload.message

    system_prompt = f"""
    You are the 'myLB AI Feynman Coach', an expert tutor testing a student's comprehension.
    The concept they must explain is: "{card.question}".
    The correct technical answer is: "{card.answer}".

    Your current state:
    - Previous Score: {feynman_session.comprehension_score}/100
    - Previous Gaps: {feynman_session.gaps_identified}

    INSTRUCTIONS:
    1. Read the student's explanation.
    2. Respond with an encouraging, conversational tone (max 3 sentences).
    3. If they missed something, ask a probing follow-up question.
    4. Calculate a live comprehension score (0-100).
    5. Calculate the score_delta (+/- change from the previous score).
    6. Update the arrays of strong_points and gaps_identified.
    7. Set 'session_complete' to true ONLY IF the score is > 90 AND they have covered all core concepts.

    YOU MUST RESPOND ONLY IN VALID JSON FORMAT matching this structure exactly:
    {{
      "ai_reply": "Your conversational reply here",
      "comprehension_score": 85,
      "score_delta": 10,
      "session_complete": false,
      "gaps_identified": ["gap 1"],
      "strong_points": ["point 1"]
    }}
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            response_format={"type": "json_object"}
        )

        ai_data = json.loads(response.choices[0].message.content)
        comp_score = ai_data.get(
            "comprehension_score",
            feynman_session.comprehension_score
        )

        feynman_session.comprehension_score = comp_score
        feynman_session.is_complete = ai_data.get("session_complete", False)
        feynman_session.gaps_identified = json.dumps(ai_data.get("gaps_identified", []))
        feynman_session.strong_points = json.dumps(ai_data.get("strong_points", []))

        db.add(feynman_session)
        db.commit()

        return {
            "ai_reply": ai_data.get("ai_reply", "I see. Could you elaborate on that?"),
            "comprehension_score": comp_score,
            "score_delta": ai_data.get("score_delta", 0),
            "session_complete": feynman_session.is_complete
        }

    except Exception as e:
        print(f"DeepSeek Feynman Error: {e}")
        raise HTTPException(
            status_code=500,
            detail="AI failed to process the response. Please try again."
        )


@router.get("/feynman/{session_id}/score")
def get_feynman_score(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    session = db.get(FeynmanSession, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        gaps = json.loads(session.gaps_identified) if session.gaps_identified else []
        strengths = json.loads(session.strong_points) if session.strong_points else []
    except json.JSONDecodeError:
        gaps, strengths = [], []

    return {
        "comprehension_score": session.comprehension_score,
        "gaps_identified": gaps,
        "strong_points": strengths
    }


@router.post("/feynman/{session_id}/complete")
def complete_feynman_session(
    session_id: int,
    payload: FeynmanCompleteRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
    x_timezone: str = Header("UTC")
):
    feynman_session = db.get(FeynmanSession, session_id)
    if not feynman_session or feynman_session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")

    feynman_session.is_complete = True
    db.add(feynman_session)

    base_xp = payload.final_score
    pet_xp = int(base_xp * 0.5)

    today_str = get_local_now(x_timezone).date().isoformat()
    daily_activity = db.exec(select(DailyActivity).where(
        DailyActivity.user_id == current_user.id,
        DailyActivity.date == today_str
    )).first()

    if daily_activity:
        daily_activity.xp_earned += base_xp
    else:
        daily_activity = DailyActivity(
            user_id=current_user.id,
            date=today_str,
            xp_earned=base_xp
        )

    db.add(daily_activity)

    pet = db.exec(select(Pet).where(Pet.user_id == current_user.id)).first()
    pet_type = "nova"
    pet_level = 1

    if pet:
        pet.xp += pet_xp
        db.add(pet)
        pet_type = pet.pet_type
        pet_level = pet.level

    update_user_streak(current_user, db)
    contribute_to_group_quests(user_id=current_user.id, xp_amount=base_xp, db=db)

    db.commit()

    return {
        "session_summary": {
            "xp_earned": base_xp,
            "pet_xp": pet_xp,
            "pet_type": pet_type,
            "pet_level": pet_level,
            "comprehension_score": payload.final_score,
            "next_suggestions": [
                {"label": "Review your gaps", "action_type": "review_gaps"}
            ]
        }
    }