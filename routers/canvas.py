import uuid
import json
import networkx as nx
import os
import io
from pypdf import PdfReader
import docx
from httpx import Timeout

from openai import OpenAI
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, UploadFile, File, Form
from sqlmodel import Session, select, or_
from typing import List, Optional

from schemas import (
    CanvasCreate, CanvasResponse, CanvasStatusResponse,
    NodeCreate, NodeResponse, NodeUpdate,
)
from models import (
    Canvas, CanvasNode, CanvasConnection, CanvasSourceType,
    Collection, CollectionAccess, CollectionItem, Note, User,
)
from security import get_current_user
from database import get_session, engine

router = APIRouter(tags=["Canvas Section"])

# ==========================================
# GEMINI CLIENT — BOUNDED TIMEOUT IS THE FIX
# ==========================================
# Previously: no timeout → chat.completions.create() can hang forever
# when Gemini is slow / rate-limited / the network blips mid-response.
# That's exactly what your log showed: task started, no error, no exit,
# canvas stuck at 0 nodes.
#
# timeout=Timeout(60.0, connect=10.0):
#   * 10 s to establish the TCP/TLS connection
#   * 60 s total for the request (headers + body streaming)
# max_retries=1: retry once on transient 5xx / connection error,
#                 then raise so we actually log the failure.

gemini_client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=Timeout(60.0, connect=10.0),
    max_retries=1,
)

# ==========================================
# FILE TEXT EXTRACTION
# ==========================================

def extract_text_from_file(file: UploadFile) -> str:
    """Reads PDF, DOCX, or TXT directly from memory without saving to disk."""
    content = file.file.read()

    # FIXED: `file.filename` can be None on some Android providers.
    filename = (file.filename or "").lower()
    text = ""

    try:
        if filename.endswith(".pdf"):
            pdf = PdfReader(io.BytesIO(content))
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        elif filename.endswith(".docx"):
            doc = docx.Document(io.BytesIO(content))
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif filename.endswith(".txt"):
            text = content.decode("utf-8", errors="ignore")
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Use PDF, DOCX, or TXT.",
            )
    except HTTPException:
        # FIXED: don't re-wrap our own 400 as a 500
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse file: {str(e)}",
        )

    return text.strip()


# ==========================================
# BACKGROUND AI PROCESSING — CRASH-SAFE
# ==========================================

def process_note_to_canvas_bg(canvas_id: uuid.UUID, note_content: str):
    """
    Two-phase, fully-guarded background task.

    Phase 1: Gemini call (bounded by client timeout).
    Phase 2: Build graph + persist nodes/connections.

    Any exception in either phase is logged with a distinct prefix
    (`[canvas-bg] AI FAILURE` / `[canvas-bg] BUILD FAILURE`) so you
    can grep Render logs and see exactly which phase broke.
    """
    print(f"[canvas-bg] START canvas={canvas_id}")

    with Session(engine) as db:
        canvas = db.exec(
            select(Canvas).where(Canvas.id == canvas_id)
        ).first()
        if not canvas:
            print(f"[canvas-bg] canvas {canvas_id} not found — aborting")
            return

        prompt = f"""
        You are an expert educational AI. Analyze the following study notes and extract a mind map structure.
        Return ONLY raw, valid JSON. Do not use markdown blocks like ```json.

        Structure the JSON exactly like this:
        {{
          "hero_concept": "The single main topic (string)",
          "concepts": [
            {{ "id": "c1", "label": "Concept Name", "is_weak": false, "definition": "Short definition" }}
          ],
          "relationships": [
            {{ "from_id": "hero", "to_id": "c1", "label": "relates to" }},
            {{ "from_id": "c1", "to_id": "c2", "label": "produces" }}
          ]
        }}

        NOTES TO ANALYZE:
        {note_content}
        """

        # ---------- PHASE 1: GEMINI ----------
        try:
            response = gemini_client.chat.completions.create(
                model="gemini-3.8-flash",
                messages=[
                    {"role": "system", "content": "You are an API that only returns valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            ai_data = json.loads(response.choices[0].message.content)
            concepts_count = len(ai_data.get("concepts") or [])
            rels_count = len(ai_data.get("relationships") or [])
            print(
                f"[canvas-bg] AI OK canvas={canvas_id}: "
                f"{concepts_count} concepts, {rels_count} relations"
            )
        except Exception as e:
            print(
                f"[canvas-bg] AI FAILURE canvas={canvas_id}: "
                f"{type(e).__name__}: {e}"
            )
            # node_count stays at 0 → client will treat this as failed
            # after its 90s polling window.
            return

        # ---------- PHASE 2: BUILD & PERSIST ----------
        try:
            _persist_canvas_from_ai_data(db, canvas_id, canvas, ai_data)
            print(
                f"[canvas-bg] DONE canvas={canvas_id} → "
                f"{canvas.node_count} nodes"
            )
        except Exception as e:
            print(
                f"[canvas-bg] BUILD FAILURE canvas={canvas_id}: "
                f"{type(e).__name__}: {e}"
            )
            db.rollback()
            # If we got partway through, wipe any orphan nodes/connections
            # that might have been added before the crash.
            try:
                stray_nodes = db.exec(
                    select(CanvasNode).where(CanvasNode.canvas_id == canvas_id)
                ).all()
                for n in stray_nodes:
                    db.delete(n)
                stray_conns = db.exec(
                    select(CanvasConnection).where(
                        CanvasConnection.canvas_id == canvas_id
                    )
                ).all()
                for c in stray_conns:
                    db.delete(c)
                canvas = db.exec(
                    select(Canvas).where(Canvas.id == canvas_id)
                ).first()
                if canvas:
                    canvas.node_count = 0
                    canvas.weak_node_count = 0
                    db.add(canvas)
                db.commit()
            except Exception as cleanup_err:
                print(
                    f"[canvas-bg] cleanup also failed canvas={canvas_id}: "
                    f"{cleanup_err}"
                )
                db.rollback()


def _persist_canvas_from_ai_data(
    db: Session,
    canvas_id: uuid.UUID,
    canvas: Canvas,
    ai_data: dict,
):
    """
    All dict access is .get()-based. No KeyError, no TypeError on
    None labels, no node-id collisions with "hero".
    """
    raw_concepts = ai_data.get("concepts") or []
    raw_rels = ai_data.get("relationships") or []
    if not isinstance(raw_concepts, list):
        raw_concepts = []
    if not isinstance(raw_rels, list):
        raw_rels = []

    hero_label = str(ai_data.get("hero_concept") or "Main Topic")[:40]

    # --- Normalise concept ids so nothing collides with "hero" ---
    seen_ids = {"hero"}
    safe_concepts = []
    for i, c in enumerate(raw_concepts):
        if not isinstance(c, dict):
            continue
        raw_id = str(c.get("id") or "").strip()
        if not raw_id or raw_id in seen_ids:
            raw_id = f"c{i}_{uuid.uuid4().hex[:4]}"
        seen_ids.add(raw_id)
        safe_concepts.append({
            "id": raw_id,
            "label": str(c.get("label") or "Untitled")[:40],
            "is_weak": bool(c.get("is_weak", False)),
            "definition": c.get("definition"),
        })

    # --- Build the graph (only edges between known ids) ---
    G = nx.Graph()
    G.add_node("hero")
    for c in safe_concepts:
        G.add_node(c["id"])
    for rel in raw_rels:
        if not isinstance(rel, dict):
            continue
        a = rel.get("from_id")
        b = rel.get("to_id")
        if a in G.nodes and b in G.nodes and a != b:
            G.add_edge(a, b)

    positions = nx.spring_layout(
        G, center=(1500, 1500), scale=600, seed=42
    )

    # --- Hero node ---
    hero_uuid = uuid.uuid4()
    hx, hy = positions.get("hero", (1500.0, 1500.0))
    db.add(CanvasNode(
        id=hero_uuid,
        canvas_id=canvas_id,
        label=hero_label,
        x=float(hx),
        y=float(hy),
        size="large",
        is_hero=True,
    ))
    id_map = {"hero": hero_uuid}

    # --- Concept nodes ---
    for c in safe_concepts:
        node_uuid = uuid.uuid4()
        id_map[c["id"]] = node_uuid
        x, y = positions.get(c["id"], (1500.0, 1500.0))
        db.add(CanvasNode(
            id=node_uuid,
            canvas_id=canvas_id,
            label=c["label"],
            x=float(x),
            y=float(y),
            size="medium",
            is_weak=c["is_weak"],
            definition=c["definition"],
        ))

    # --- Edges ---
    for rel in raw_rels:
        if not isinstance(rel, dict):
            continue
        a = id_map.get(rel.get("from_id"))
        b = id_map.get(rel.get("to_id"))
        if a and b and a != b:
            db.add(CanvasConnection(
                canvas_id=canvas_id,
                from_node_id=a,
                to_node_id=b,
                label=rel.get("label"),
            ))

    canvas.node_count = len(safe_concepts) + 1
    canvas.weak_node_count = sum(
        1 for c in safe_concepts if c["is_weak"]
    )
    db.add(canvas)
    db.commit()


# ==========================================
# API ENDPOINTS
# ==========================================

# 1. GET ALL CANVASES
@router.get("/users/me/canvases", status_code=200)
def get_user_canvases(
    search: Optional[str] = "",
    filter: Optional[str] = "All",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    query = select(Canvas).where(Canvas.user_id == current_user.id)
    if search:
        query = query.where(
            (Canvas.name.icontains(search)) | (Canvas.subject.icontains(search))
        )
    if filter and filter.lower() != "all":
        query = query.where(Canvas.subject.ilike(filter))
    query = query.order_by(Canvas.id.desc())
    canvases = db.exec(query).all()
    return {"canvases": canvases}


# 2. CREATE CANVAS FROM NOTES (AI)
@router.post("/canvases/from-notes", response_model=CanvasStatusResponse)
def create_canvas_from_note(
    payload: CanvasCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    if not payload.note_id:
        raise HTTPException(status_code=400, detail="note_id is required")

    note = db.exec(
        select(Note).where(
            Note.id == payload.note_id,
            Note.user_id == current_user.id,
        )
    ).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    db_canvas = Canvas(
        user_id=current_user.id,
        name=payload.name,
        subject=payload.subject,
        source_type=CanvasSourceType.notes,
        source_id=payload.note_id,
    )
    db.add(db_canvas)
    db.commit()
    db.refresh(db_canvas)

    background_tasks.add_task(
        process_note_to_canvas_bg, db_canvas.id, note.content_text
    )
    return {"status": "processing", "node_count": 0, "nodes": []}


# 3. CREATE CANVAS FROM FILE UPLOAD (AI)
@router.post("/canvases/from-file", response_model=CanvasStatusResponse)
def create_canvas_from_file(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    subject: str = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    document_text = extract_text_from_file(file)
    if not document_text:
        raise HTTPException(
            status_code=400,
            detail="Could not extract any text from the uploaded file.",
        )

    db_canvas = Canvas(
        user_id=current_user.id,
        name=name,
        subject=subject,
        # FIXED: was CanvasSourceType.manual — that made uploads
        # indistinguishable from intentional empty manual canvases,
        # so the client couldn't tell "still AI-ing" from "done".
        source_type=CanvasSourceType.upload,
        source_id=None,
    )
    db.add(db_canvas)
    db.commit()
    db.refresh(db_canvas)

    background_tasks.add_task(
        process_note_to_canvas_bg, db_canvas.id, document_text
    )
    return {"status": "processing", "node_count": 0, "nodes": []}


# 4. GET CANVAS STATUS / CONTENT
@router.get("/canvases/{canvas_id}/status", response_model=CanvasStatusResponse)
def get_canvas_status(
    canvas_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    canvas = db.exec(
        select(Canvas).where(
            Canvas.id == canvas_id, Canvas.user_id == current_user.id
        )
    ).first()
    if not canvas:
        raise HTTPException(status_code=404, detail="Canvas not found")

    if canvas.node_count > 0:
        return {
            "status": "ready",
            "node_count": canvas.node_count,
            "nodes": canvas.nodes,
        }
    return {"status": "processing", "node_count": 0, "nodes": []}


# 5. CREATE MANUAL CANVAS
@router.post("/canvases", response_model=CanvasResponse)
def create_manual_canvas(
    payload: CanvasCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    db_canvas = Canvas(
        user_id=current_user.id,
        name=payload.name,
        subject=payload.subject,
        source_type=CanvasSourceType.manual,
    )
    db.add(db_canvas)
    db.commit()
    db.refresh(db_canvas)
    return db_canvas


# 6. ADD NODE
@router.post("/canvases/{canvas_id}/nodes", response_model=NodeResponse)
def add_node_to_canvas(
    canvas_id: uuid.UUID,
    node: NodeCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    canvas = db.exec(
        select(Canvas).where(
            Canvas.id == canvas_id, Canvas.user_id == current_user.id
        )
    ).first()
    if not canvas:
        raise HTTPException(status_code=404, detail="Canvas not found")

    db_node = CanvasNode(
        canvas_id=canvas_id,
        label=node.label,
        x=node.x,
        y=node.y,
        size=node.size,
        is_hero=node.is_hero,
        is_weak=node.is_weak,
    )
    db.add(db_node)
    canvas.node_count += 1
    db.commit()
    db.refresh(db_node)
    return db_node


# 7. GET SINGLE CANVAS
@router.get("/canvases/{canvas_id}", status_code=200)
def get_single_canvas(
    canvas_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    canvas = db.get(Canvas, canvas_id)
    if not canvas:
        raise HTTPException(status_code=404, detail="Canvas not found")

    is_owner = False
    if canvas.user_id == current_user.id:
        is_owner = True
    elif getattr(canvas, "is_public", False):
        is_owner = False
    else:
        has_access = db.exec(
            select(CollectionItem.id)
            .join(Collection, Collection.id == CollectionItem.collection_id)
            .outerjoin(
                CollectionAccess,
                CollectionAccess.collection_id == Collection.id,
            )
            .where(
                CollectionItem.item_type == "canvas",
                CollectionItem.item_id == str(canvas.id),
                or_(
                    Collection.visibility == "public",
                    CollectionAccess.user_id == current_user.id,
                ),
            )
        ).first()
        if has_access:
            is_owner = False
        else:
            raise HTTPException(status_code=404, detail="Canvas not found")

    nodes = db.exec(
        select(CanvasNode).where(CanvasNode.canvas_id == canvas.id)
    ).all()
    connections = db.exec(
        select(CanvasConnection).where(
            CanvasConnection.canvas_id == canvas.id
        )
    ).all()

    response_data = canvas.model_dump()
    response_data["is_owner"] = is_owner
    response_data["nodes"] = nodes
    response_data["connections"] = connections
    return response_data


# 8. DELETE CANVAS
@router.delete("/canvases/{canvas_id}", status_code=200)
def delete_canvas(
    canvas_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    canvas = db.exec(
        select(Canvas).where(
            Canvas.id == canvas_id, Canvas.user_id == current_user.id
        )
    ).first()
    if not canvas:
        raise HTTPException(status_code=404, detail="Canvas not found")

    connections = db.exec(
        select(CanvasConnection).where(
            CanvasConnection.canvas_id == canvas_id
        )
    ).all()
    for conn in connections:
        db.delete(conn)

    nodes = db.exec(
        select(CanvasNode).where(CanvasNode.canvas_id == canvas_id)
    ).all()
    for node in nodes:
        db.delete(node)

    db.delete(canvas)
    db.commit()
    return {"message": "Canvas and all its nodes deleted successfully"}


# 9. UPDATE NODE
@router.patch("/canvases/{canvas_id}/nodes/{node_id}", status_code=200)
def update_canvas_node(
    canvas_id: uuid.UUID,
    node_id: uuid.UUID,
    payload: NodeUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    canvas = db.exec(
        select(Canvas).where(
            Canvas.id == canvas_id, Canvas.user_id == current_user.id
        )
    ).first()
    if not canvas:
        raise HTTPException(status_code=404, detail="Canvas not found")

    node = db.exec(
        select(CanvasNode).where(
            CanvasNode.id == node_id, CanvasNode.canvas_id == canvas_id
        )
    ).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(node, key, value)

    db.add(node)
    db.commit()
    db.refresh(node)
    return node


# 10. DELETE NODE
@router.delete("/canvases/{canvas_id}/nodes/{node_id}", status_code=200)
def delete_canvas_node(
    canvas_id: uuid.UUID,
    node_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    canvas = db.exec(
        select(Canvas).where(
            Canvas.id == canvas_id, Canvas.user_id == current_user.id
        )
    ).first()
    if not canvas:
        raise HTTPException(status_code=404, detail="Canvas not found")

    node = db.exec(
        select(CanvasNode).where(
            CanvasNode.id == node_id, CanvasNode.canvas_id == canvas_id
        )
    ).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    connections = db.exec(
        select(CanvasConnection).where(
            (CanvasConnection.from_node_id == node_id)
            | (CanvasConnection.to_node_id == node_id)
        )
    ).all()
    for conn in connections:
        db.delete(conn)

    db.delete(node)
    if canvas.node_count > 0:
        canvas.node_count -= 1
        db.add(canvas)
    db.commit()
    return {"message": "Node deleted successfully"}