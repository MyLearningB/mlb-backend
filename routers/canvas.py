"""
Canvas routes — v2.

What changed vs v1:
  * Real `status` column (processing | ready | failed) + error_message.
    The client no longer guesses "failed" from node_count + its own clock.
  * Stale detection: a generation stuck in "processing" for > 4 min
    (e.g. Render restarted mid-task) is flipped to "failed" on read.
  * POST /canvases/{id}/retry actually re-runs the AI (source text is
    stored on the canvas, so uploads can be retried without re-picking
    the file).
  * AI prompt asks for a TREE (parent_id) + a few cross-links, and the
    layout is a radial tidy-tree, so related ideas sit together.
  * POST /canvases/{id}/layout re-runs that layout on any canvas.
  * Connections can be created/deleted. Adding a node with parent_id
    auto-connects it.
  * node_count / weak_node_count are recounted from the DB, never
    hand-incremented.
  * List is ordered by created_at (was UUID = random order) and
    returns the full subject list for filter chips.
  * Timestamps are serialised as UTC with a trailing "Z" so Dart's
    DateTime.parse doesn't treat them as local time.
  * source_text is never sent to the client.
"""

import io
import json
import math
import os
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import docx
from fastapi import (
    APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile,
)
from httpx import Timeout
from openai import OpenAI
from pypdf import PdfReader
from sqlmodel import Session, func, or_, select

from database import engine, get_session
from models import (
    Canvas, CanvasConnection, CanvasNode, CanvasSourceType,
    Collection, CollectionAccess, CollectionItem, Note, User,
)
from schemas import (
    CanvasCreate, CanvasUpdate, ConnectionCreate, NodeCreate, NodeUpdate,
)
from security import get_current_user

router = APIRouter(tags=["Canvas Section"])

# ==========================================
# CONFIG
# ==========================================

# Override on Render with CANVAS_AI_MODEL if you switch models.
AI_MODEL = os.environ.get("CANVAS_AI_MODEL", "gemini-3.8-flash")

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_SOURCE_CHARS = 60_000          # what we send to the AI
STALE_AFTER = timedelta(minutes=4)  # processing longer than this = dead

CANVAS_SIZE = 5000.0
LAYOUT_CENTER = 2500.0

LABEL_MAX = 40
DEFINITION_MAX = 500
EDGE_LABEL_MAX = 60
ERROR_MAX = 300

STATUS_PROCESSING = "processing"
STATUS_READY = "ready"
STATUS_FAILED = "failed"

gemini_client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=Timeout(60.0, connect=10.0),
    max_retries=1,
)


# ==========================================
# TIME HELPERS
# ==========================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Columns are TIMESTAMP WITHOUT TIME ZONE, so values come back
    naive. They were written as UTC — re-attach the zone."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    d = _as_utc(dt)
    return d.isoformat().replace("+00:00", "Z") if d else None


# ==========================================
# SERIALISERS (never include source_text)
# ==========================================

def _enum_value(v):
    return getattr(v, "value", v)


def _canvas_summary(c: Canvas) -> dict:
    return {
        "id": str(c.id),
        "name": c.name,
        "subject": c.subject,
        "status": c.status or STATUS_READY,
        "error_message": c.error_message,
        "node_count": c.node_count,
        "weak_node_count": c.weak_node_count,
        "thumbnail_url": c.thumbnail_url,
        "source_type": _enum_value(c.source_type),
        "source_id": c.source_id,
        "last_studied_at": _iso(c.last_studied_at),
        "created_at": _iso(c.created_at),
        "updated_at": _iso(c.updated_at),
        "is_public": c.is_public,
    }


def _node_dict(n: CanvasNode) -> dict:
    return {
        "id": str(n.id),
        "label": n.label,
        "x": n.x,
        "y": n.y,
        "size": _enum_value(n.size),
        "is_hero": n.is_hero,
        "is_weak": n.is_weak,
        "definition": n.definition,
        "card_id": n.card_id,
    }


def _conn_dict(c: CanvasConnection) -> dict:
    return {
        "id": str(c.id),
        "from_node_id": str(c.from_node_id),
        "to_node_id": str(c.to_node_id),
        "label": c.label,
    }


def _canvas_detail(db: Session, canvas: Canvas, is_owner: bool) -> dict:
    nodes = db.exec(
        select(CanvasNode).where(CanvasNode.canvas_id == canvas.id)
    ).all()
    conns = db.exec(
        select(CanvasConnection).where(CanvasConnection.canvas_id == canvas.id)
    ).all()
    data = _canvas_summary(canvas)
    data["is_owner"] = is_owner
    data["nodes"] = [_node_dict(n) for n in nodes]
    data["connections"] = [_conn_dict(c) for c in conns]
    return data


# ==========================================
# DB HELPERS
# ==========================================

def _get_owned_canvas(db: Session, canvas_id: uuid.UUID, user: User) -> Canvas:
    canvas = db.exec(
        select(Canvas).where(Canvas.id == canvas_id, Canvas.user_id == user.id)
    ).first()
    if not canvas:
        raise HTTPException(status_code=404, detail="Canvas not found")
    return canvas


def _get_node(db: Session, canvas_id: uuid.UUID, node_id: uuid.UUID) -> CanvasNode:
    node = db.exec(
        select(CanvasNode).where(
            CanvasNode.id == node_id, CanvasNode.canvas_id == canvas_id
        )
    ).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return node


def _recount(db: Session, canvas: Canvas) -> None:
    db.flush()
    weak_flags = db.exec(
        select(CanvasNode.is_weak).where(CanvasNode.canvas_id == canvas.id)
    ).all()
    canvas.node_count = len(weak_flags)
    canvas.weak_node_count = sum(1 for w in weak_flags if w)
    db.add(canvas)


def _delete_canvas_children(db: Session, canvas_id: uuid.UUID) -> None:
    for c in db.exec(
        select(CanvasConnection).where(CanvasConnection.canvas_id == canvas_id)
    ).all():
        db.delete(c)
    db.flush()
    for n in db.exec(
        select(CanvasNode).where(CanvasNode.canvas_id == canvas_id)
    ).all():
        db.delete(n)
    db.flush()


def _expire_if_stale(canvas: Canvas) -> bool:
    """Flip a dead 'processing' canvas to 'failed'. Caller commits."""
    if canvas.status != STATUS_PROCESSING:
        return False
    ref = _as_utc(canvas.updated_at) or _as_utc(canvas.created_at)
    if ref and utc_now() - ref > STALE_AFTER:
        canvas.status = STATUS_FAILED
        canvas.error_message = "Generation timed out. Tap Try again."
        canvas.updated_at = utc_now()
        return True
    return False


def _clamp(v: float, lo: float = 40.0, hi: float = CANVAS_SIZE - 40.0) -> float:
    return max(lo, min(hi, float(v)))


def _clean_text(v: Optional[str], limit: int) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s[:limit] if s else None


# ==========================================
# FILE TEXT EXTRACTION
# ==========================================

def extract_text_from_file(file: UploadFile) -> str:
    """Reads PDF, DOCX or TXT in memory."""
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Max 5 MB.")

    filename = (file.filename or "").lower()
    text_parts: List[str] = []

    try:
        if filename.endswith(".pdf"):
            pdf = PdfReader(io.BytesIO(content))
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text_parts.append(extracted)
        elif filename.endswith(".docx"):
            doc = docx.Document(io.BytesIO(content))
            text_parts.extend(p.text for p in doc.paragraphs if p.text)
            for table in doc.tables:
                for row in table.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        text_parts.append(" | ".join(cells))
        elif filename.endswith(".txt"):
            text_parts.append(content.decode("utf-8", errors="ignore"))
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Use PDF, DOCX, or TXT.",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Couldn't read that file: {e}")

    return "\n".join(text_parts).strip()


# ==========================================
# LAYOUT — RADIAL TIDY TREE
# ==========================================

def compute_radial_tree_layout(
    node_ids: List,
    root,
    edges: Iterable[Tuple],
) -> Dict:
    """
    Root at the centre. Every subtree gets an angular wedge proportional
    to its number of leaves, so siblings stay together and branches
    don't overlap. Nodes not reachable from the root are attached to it.
    Deterministic, O(n), no numpy.
    """
    ids = list(node_ids)
    if not ids:
        return {}
    if root not in ids:
        root = ids[0]

    adj: Dict = {n: [] for n in ids}
    for a, b in edges:
        if a in adj and b in adj and a != b:
            adj[a].append(b)
            adj[b].append(a)

    children: Dict = {n: [] for n in ids}
    depth: Dict = {root: 0}
    order: List = [root]

    def bfs(start):
        q = deque([start])
        while q:
            cur = q.popleft()
            for nb in adj[cur]:
                if nb not in depth:
                    depth[nb] = depth[cur] + 1
                    children[cur].append(nb)
                    order.append(nb)
                    q.append(nb)

    bfs(root)
    for n in ids:
        if n not in depth:
            depth[n] = 1
            children[root].append(n)
            order.append(n)
            bfs(n)

    leaves: Dict = {}
    for n in reversed(order):
        leaves[n] = sum(leaves[c] for c in children[n]) or 1

    max_depth = max(depth.values()) or 1
    # Ring spacing grows with the number of leaves so outer nodes have
    # ~170 px of arc each, but the map never exceeds the virtual canvas.
    step = max(300.0, leaves[root] * 170.0 / (2 * math.pi * max_depth))
    step = min(step, (LAYOUT_CENTER - 200.0) / max_depth)

    pos: Dict = {root: (LAYOUT_CENTER, LAYOUT_CENTER)}
    wedge: Dict = {root: (0.0, 2 * math.pi)}
    for n in order:
        start, size = wedge[n]
        cursor = start
        for c in children[n]:
            w = size * leaves[c] / leaves[n]
            wedge[c] = (cursor, w)
            angle = cursor + w / 2 - math.pi / 2
            r = depth[c] * step
            pos[c] = (
                LAYOUT_CENTER + r * math.cos(angle),
                LAYOUT_CENTER + r * math.sin(angle),
            )
            cursor += w
    return pos


# ==========================================
# AI GENERATION
# ==========================================

def _build_prompt(text: str) -> str:
    return f"""
You are an expert teacher turning study material into a mind map.
Return ONLY a JSON object (no markdown) with exactly this shape:
{{
  "hero_concept": "the single main topic, max 5 words",
  "hero_definition": "one-sentence summary of the whole topic",
  "concepts": [
    {{
      "id": "c1",
      "label": "Concept name, max 5 words",
      "parent_id": "hero",
      "relation": "short verb phrase from parent to this concept",
      "definition": "1-2 plain-English sentences explaining it"
    }}
  ],
  "cross_links": [
    {{ "from_id": "c2", "to_id": "c5", "label": "short verb phrase" }}
  ]
}}
Rules:
- 8 to 24 concepts. Every parent_id is "hero" or another concept's id, forming a tree.
- Group related ideas under sub-topics instead of attaching everything to "hero".
- At most 6 cross_links, only for important relationships outside the tree.
- Labels must be short. Definitions must be accurate to the material.

MATERIAL:
{text}
"""


def _strip_fences(raw: str) -> str:
    s = (raw or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


def _parse_ai_graph(ai_data: dict) -> dict:
    """Normalise whatever the model returned into a safe structure."""
    if not isinstance(ai_data, dict):
        raise ValueError("AI response is not an object")

    raw_concepts = ai_data.get("concepts") or []
    if not isinstance(raw_concepts, list):
        raw_concepts = []

    seen = {"hero"}
    concepts = []
    for i, c in enumerate(raw_concepts):
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "").strip()
        if not cid or cid in seen:
            cid = f"c{i}_{uuid.uuid4().hex[:4]}"
        seen.add(cid)
        concepts.append({
            "id": cid,
            "label": (str(c.get("label") or "Untitled").strip() or "Untitled")[:LABEL_MAX],
            "definition": _clean_text(c.get("definition"), DEFINITION_MAX),
            "parent_id": str(c.get("parent_id") or "").strip() or None,
            "relation": _clean_text(c.get("relation"), EDGE_LABEL_MAX),
        })

    valid_ids = {"hero"} | {c["id"] for c in concepts}

    # Tree edges from parent_id (fallback to hero).
    tree_edges = []
    for c in concepts:
        parent = c["parent_id"]
        if parent not in valid_ids or parent == c["id"]:
            parent = "hero"
        tree_edges.append((parent, c["id"], c["relation"]))

    # Cross links (+ legacy "relationships" key from the v1 prompt).
    extra = []
    for key in ("cross_links", "relationships"):
        items = ai_data.get(key) or []
        if not isinstance(items, list):
            continue
        for rel in items:
            if not isinstance(rel, dict):
                continue
            a, b = rel.get("from_id"), rel.get("to_id")
            if a in valid_ids and b in valid_ids and a != b:
                extra.append((a, b, _clean_text(rel.get("label"), EDGE_LABEL_MAX)))

    return {
        "hero_label": (str(ai_data.get("hero_concept") or "Main Topic").strip() or "Main Topic")[:LABEL_MAX],
        "hero_definition": _clean_text(ai_data.get("hero_definition"), DEFINITION_MAX),
        "concepts": concepts,
        "tree_edges": tree_edges,
        "extra_edges": extra,
    }


def _persist_graph(db: Session, canvas: Canvas, graph: dict) -> None:
    _delete_canvas_children(db, canvas.id)

    all_ids = ["hero"] + [c["id"] for c in graph["concepts"]]
    positions = compute_radial_tree_layout(
        all_ids, "hero", [(a, b) for a, b, _ in graph["tree_edges"]]
    )

    id_map: Dict[str, uuid.UUID] = {}

    hero_id = uuid.uuid4()
    id_map["hero"] = hero_id
    hx, hy = positions.get("hero", (LAYOUT_CENTER, LAYOUT_CENTER))
    db.add(CanvasNode(
        id=hero_id, canvas_id=canvas.id, label=graph["hero_label"],
        x=float(hx), y=float(hy), size="large", is_hero=True,
        definition=graph["hero_definition"],
    ))

    for c in graph["concepts"]:
        nid = uuid.uuid4()
        id_map[c["id"]] = nid
        x, y = positions.get(c["id"], (LAYOUT_CENTER, LAYOUT_CENTER))
        db.add(CanvasNode(
            id=nid, canvas_id=canvas.id, label=c["label"],
            x=float(x), y=float(y), size="medium",
            # is_weak is NOT taken from the AI any more — it's set by the
            # user's own study answers in Study mode.
            is_weak=False, definition=c["definition"],
        ))

    db.flush()

    seen_pairs = set()
    for a, b, label in graph["tree_edges"] + graph["extra_edges"]:
        ua, ub = id_map.get(a), id_map.get(b)
        if not ua or not ub or ua == ub:
            continue
        key = frozenset((ua, ub))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        db.add(CanvasConnection(
            canvas_id=canvas.id, from_node_id=ua, to_node_id=ub, label=label,
        ))

    _recount(db, canvas)
    canvas.status = STATUS_READY
    canvas.error_message = None
    canvas.updated_at = utc_now()
    db.add(canvas)
    db.commit()


def _mark_failed(db: Session, canvas_id: uuid.UUID, message: str) -> None:
    try:
        db.rollback()
        canvas = db.get(Canvas, canvas_id)
        if not canvas:
            return
        _delete_canvas_children(db, canvas_id)
        canvas.node_count = 0
        canvas.weak_node_count = 0
        canvas.status = STATUS_FAILED
        canvas.error_message = message[:ERROR_MAX]
        canvas.updated_at = utc_now()
        db.add(canvas)
        db.commit()
    except Exception as e:
        print(f"[canvas-bg] could not mark failed canvas={canvas_id}: {e}")
        db.rollback()


def process_canvas_bg(canvas_id: uuid.UUID) -> None:
    """Background task: AI call -> parse -> layout -> persist."""
    print(f"[canvas-bg] START canvas={canvas_id}")
    with Session(engine) as db:
        canvas = db.get(Canvas, canvas_id)
        if not canvas:
            print(f"[canvas-bg] canvas {canvas_id} not found — aborting")
            return

        text = (canvas.source_text or "").strip()
        if not text and canvas.source_id:
            note = db.get(Note, canvas.source_id)
            if note:
                text = (note.content_text or "").strip()
        if not text:
            _mark_failed(db, canvas_id, "There was no text to build a mind map from.")
            return

        # ---------- PHASE 1: AI ----------
        try:
            response = gemini_client.chat.completions.create(
                model=AI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an API that only returns valid JSON."},
                    {"role": "user", "content": _build_prompt(text[:MAX_SOURCE_CHARS])},
                ],
                response_format={"type": "json_object"},
            )
            ai_data = json.loads(_strip_fences(response.choices[0].message.content))
        except Exception as e:
            print(f"[canvas-bg] AI FAILURE canvas={canvas_id}: {type(e).__name__}: {e}")
            _mark_failed(db, canvas_id, "The AI couldn't process this material. Tap Try again.")
            return

        # ---------- PHASE 2: BUILD & PERSIST ----------
        try:
            graph = _parse_ai_graph(ai_data)
            if not graph["concepts"]:
                raise ValueError("AI returned no concepts")
            canvas = db.get(Canvas, canvas_id)
            if not canvas:
                return
            _persist_graph(db, canvas, graph)
            print(f"[canvas-bg] DONE canvas={canvas_id} -> {canvas.node_count} nodes")
        except Exception as e:
            print(f"[canvas-bg] BUILD FAILURE canvas={canvas_id}: {type(e).__name__}: {e}")
            _mark_failed(db, canvas_id, "The mind map couldn't be built from the AI's answer. Tap Try again.")


# ==========================================
# ENDPOINTS — CANVASES
# ==========================================

@router.get("/users/me/canvases", status_code=200)
def get_user_canvases(
    search: Optional[str] = "",
    filter: Optional[str] = "All",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    query = select(Canvas).where(Canvas.user_id == current_user.id)
    s = (search or "").strip()
    if s:
        query = query.where(or_(Canvas.name.icontains(s), Canvas.subject.icontains(s)))
    if filter and filter.lower() != "all":
        query = query.where(func.lower(Canvas.subject) == filter.lower())
    query = query.order_by(Canvas.created_at.desc())

    canvases = db.exec(query).all()
    changed = False
    for c in canvases:
        if _expire_if_stale(c):
            db.add(c)
            changed = True
    if changed:
        db.commit()

    # Full subject list (unfiltered) so chips don't vanish when one is picked.
    subjects = db.exec(
        select(Canvas.subject).where(Canvas.user_id == current_user.id).distinct()
    ).all()
    subjects = sorted({x for x in subjects if x}, key=lambda x: x.lower())

    return {
        "canvases": [_canvas_summary(c) for c in canvases],
        "subjects": subjects,
    }


@router.post("/canvases/from-notes", status_code=200)
def create_canvas_from_note(
    payload: CanvasCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    note_id = payload.note_id or payload.source_id
    if not note_id:
        raise HTTPException(status_code=400, detail="note_id is required")

    note = db.exec(
        select(Note).where(Note.id == note_id, Note.user_id == current_user.id)
    ).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    text = (note.content_text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="This note is empty. Write something first.")

    name = (payload.name or "").strip() or f"{note.title} Canvas"
    subject = (payload.subject or "").strip() or note.subject or "General"

    canvas = Canvas(
        user_id=current_user.id,
        name=name[:80],
        subject=subject[:100],
        source_type=CanvasSourceType.notes,
        source_id=note.id,
        status=STATUS_PROCESSING,
        source_text=text[:MAX_SOURCE_CHARS],
        updated_at=utc_now(),
    )
    note.has_canvas = True
    db.add(note)
    db.add(canvas)
    db.commit()
    db.refresh(canvas)

    background_tasks.add_task(process_canvas_bg, canvas.id)
    return {"status": STATUS_PROCESSING, "node_count": 0, "canvas": _canvas_summary(canvas)}


@router.post("/canvases/from-file", status_code=200)
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
            detail="Couldn't find any text in that file. Scanned PDFs aren't supported yet.",
        )

    canvas = Canvas(
        user_id=current_user.id,
        name=(name.strip() or "Uploaded Document")[:80],
        subject=(subject.strip() or "Document Upload")[:100],
        source_type=CanvasSourceType.upload,
        source_id=None,
        status=STATUS_PROCESSING,
        source_text=document_text[:MAX_SOURCE_CHARS],
        updated_at=utc_now(),
    )
    db.add(canvas)
    db.commit()
    db.refresh(canvas)

    background_tasks.add_task(process_canvas_bg, canvas.id)
    return {"status": STATUS_PROCESSING, "node_count": 0, "canvas": _canvas_summary(canvas)}


@router.get("/canvases/{canvas_id}/status", status_code=200)
def get_canvas_status(
    canvas_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    canvas = _get_owned_canvas(db, canvas_id, current_user)
    if _expire_if_stale(canvas):
        db.add(canvas)
        db.commit()
    return {
        "status": canvas.status or STATUS_READY,
        "node_count": canvas.node_count,
        "error": canvas.error_message,
    }


@router.post("/canvases", status_code=200)
def create_manual_canvas(
    payload: CanvasCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    canvas = Canvas(
        user_id=current_user.id,
        name=((payload.name or "").strip() or "Untitled Canvas")[:80],
        subject=((payload.subject or "").strip() or "General")[:100],
        source_type=CanvasSourceType.manual,
        status=STATUS_READY,
    )
    db.add(canvas)
    db.commit()
    db.refresh(canvas)
    return _canvas_summary(canvas)


@router.patch("/canvases/{canvas_id}", status_code=200)
def update_canvas(
    canvas_id: uuid.UUID,
    payload: CanvasUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    canvas = _get_owned_canvas(db, canvas_id, current_user)
    data = payload.model_dump(exclude_unset=True)
    if "name" in data and data["name"] is not None:
        canvas.name = data["name"].strip()[:80] or canvas.name
    if "subject" in data and data["subject"] is not None:
        canvas.subject = data["subject"].strip()[:100] or canvas.subject
    db.add(canvas)
    db.commit()
    db.refresh(canvas)
    return _canvas_summary(canvas)


@router.post("/canvases/{canvas_id}/retry", status_code=200)
def retry_canvas_generation(
    canvas_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    canvas = _get_owned_canvas(db, canvas_id, current_user)
    _expire_if_stale(canvas)

    if canvas.status == STATUS_PROCESSING:
        raise HTTPException(status_code=409, detail="This canvas is already generating.")
    if _enum_value(canvas.source_type) == "manual":
        raise HTTPException(status_code=400, detail="Manual canvases aren't AI-generated.")

    has_text = bool((canvas.source_text or "").strip())
    if not has_text and canvas.source_id:
        note = db.get(Note, canvas.source_id)
        has_text = bool(note and (note.content_text or "").strip())
    if not has_text:
        raise HTTPException(
            status_code=400,
            detail="The original text for this canvas isn't stored. Delete it and upload the file again.",
        )

    canvas.status = STATUS_PROCESSING
    canvas.error_message = None
    canvas.updated_at = utc_now()
    db.add(canvas)
    db.commit()
    db.refresh(canvas)

    background_tasks.add_task(process_canvas_bg, canvas.id)
    return _canvas_summary(canvas)


@router.post("/canvases/{canvas_id}/layout", status_code=200)
def auto_layout_canvas(
    canvas_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    canvas = _get_owned_canvas(db, canvas_id, current_user)
    nodes = db.exec(select(CanvasNode).where(CanvasNode.canvas_id == canvas.id)).all()
    if not nodes:
        return _canvas_detail(db, canvas, True)

    conns = db.exec(
        select(CanvasConnection).where(CanvasConnection.canvas_id == canvas.id)
    ).all()
    edges = [(c.from_node_id, c.to_node_id) for c in conns]

    hero = next((n for n in nodes if n.is_hero), None)
    if hero is not None:
        root = hero.id
    else:
        degree: Dict = {n.id: 0 for n in nodes}
        for a, b in edges:
            if a in degree:
                degree[a] += 1
            if b in degree:
                degree[b] += 1
        root = max(degree, key=lambda k: degree[k])

    positions = compute_radial_tree_layout([n.id for n in nodes], root, edges)
    for n in nodes:
        x, y = positions.get(n.id, (n.x, n.y))
        n.x, n.y = float(x), float(y)
        db.add(n)
    db.commit()
    return _canvas_detail(db, canvas, True)


@router.post("/canvases/{canvas_id}/studied", status_code=200)
def mark_canvas_studied(
    canvas_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    canvas = _get_owned_canvas(db, canvas_id, current_user)
    canvas.last_studied_at = utc_now()
    db.add(canvas)
    db.commit()
    db.refresh(canvas)
    return _canvas_summary(canvas)


@router.get("/canvases/{canvas_id}", status_code=200)
def get_single_canvas(
    canvas_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    canvas = db.get(Canvas, canvas_id)
    if not canvas:
        raise HTTPException(status_code=404, detail="Canvas not found")

    if canvas.user_id == current_user.id:
        is_owner = True
    elif getattr(canvas, "is_public", False):
        is_owner = False
    else:
        has_access = db.exec(
            select(CollectionItem.id)
            .join(Collection, Collection.id == CollectionItem.collection_id)
            .outerjoin(CollectionAccess, CollectionAccess.collection_id == Collection.id)
            .where(
                CollectionItem.item_type == "canvas",
                CollectionItem.item_id == str(canvas.id),
                or_(
                    Collection.visibility == "public",
                    CollectionAccess.user_id == current_user.id,
                ),
            )
        ).first()
        if not has_access:
            raise HTTPException(status_code=404, detail="Canvas not found")
        is_owner = False

    if _expire_if_stale(canvas):
        db.add(canvas)
        db.commit()
        db.refresh(canvas)

    return _canvas_detail(db, canvas, is_owner)


@router.delete("/canvases/{canvas_id}", status_code=200)
def delete_canvas(
    canvas_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    canvas = _get_owned_canvas(db, canvas_id, current_user)
    _delete_canvas_children(db, canvas.id)
    db.delete(canvas)
    db.commit()
    return {"message": "Canvas and all its nodes deleted successfully"}


# ==========================================
# ENDPOINTS — NODES
# ==========================================

@router.post("/canvases/{canvas_id}/nodes", status_code=200)
def add_node_to_canvas(
    canvas_id: uuid.UUID,
    node: NodeCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    canvas = _get_owned_canvas(db, canvas_id, current_user)

    label = node.label.strip()[:LABEL_MAX]
    if not label:
        raise HTTPException(status_code=400, detail="Label can't be empty.")

    parent = None
    if node.parent_id:
        parent = _get_node(db, canvas.id, node.parent_id)

    if node.is_hero:
        for other in db.exec(
            select(CanvasNode).where(CanvasNode.canvas_id == canvas.id, CanvasNode.is_hero == True)  # noqa: E712
        ).all():
            other.is_hero = False
            db.add(other)

    db_node = CanvasNode(
        canvas_id=canvas.id,
        label=label,
        x=_clamp(node.x),
        y=_clamp(node.y),
        size=node.size,
        is_hero=node.is_hero,
        is_weak=node.is_weak,
        definition=_clean_text(node.definition, DEFINITION_MAX),
    )
    db.add(db_node)
    db.flush()

    conn = None
    if parent is not None:
        conn = CanvasConnection(
            canvas_id=canvas.id, from_node_id=parent.id, to_node_id=db_node.id,
        )
        db.add(conn)

    _recount(db, canvas)
    db.commit()
    db.refresh(db_node)
    if conn is not None:
        db.refresh(conn)

    return {
        "node": _node_dict(db_node),
        "connection": _conn_dict(conn) if conn is not None else None,
        "node_count": canvas.node_count,
    }


@router.patch("/canvases/{canvas_id}/nodes/{node_id}", status_code=200)
def update_canvas_node(
    canvas_id: uuid.UUID,
    node_id: uuid.UUID,
    payload: NodeUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    canvas = _get_owned_canvas(db, canvas_id, current_user)
    node = _get_node(db, canvas.id, node_id)
    data = payload.model_dump(exclude_unset=True)

    if "label" in data:
        label = (data["label"] or "").strip()[:LABEL_MAX]
        if not label:
            raise HTTPException(status_code=400, detail="Label can't be empty.")
        node.label = label
    if "definition" in data:
        node.definition = _clean_text(data["definition"], DEFINITION_MAX)
    if data.get("x") is not None:
        node.x = _clamp(data["x"])
    if data.get("y") is not None:
        node.y = _clamp(data["y"])
    if data.get("size") is not None:
        node.size = data["size"]
    if data.get("is_weak") is not None:
        node.is_weak = data["is_weak"]
    if data.get("is_hero") is not None:
        if data["is_hero"]:
            for other in db.exec(
                select(CanvasNode).where(
                    CanvasNode.canvas_id == canvas.id,
                    CanvasNode.is_hero == True,  # noqa: E712
                    CanvasNode.id != node.id,
                )
            ).all():
                other.is_hero = False
                db.add(other)
        node.is_hero = data["is_hero"]

    db.add(node)
    if "is_weak" in data:
        _recount(db, canvas)
    db.commit()
    db.refresh(node)
    return _node_dict(node)


@router.delete("/canvases/{canvas_id}/nodes/{node_id}", status_code=200)
def delete_canvas_node(
    canvas_id: uuid.UUID,
    node_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    canvas = _get_owned_canvas(db, canvas_id, current_user)
    node = _get_node(db, canvas.id, node_id)

    for conn in db.exec(
        select(CanvasConnection).where(
            (CanvasConnection.from_node_id == node_id)
            | (CanvasConnection.to_node_id == node_id)
        )
    ).all():
        db.delete(conn)
    db.flush()
    db.delete(node)
    _recount(db, canvas)
    db.commit()
    return {"message": "Node deleted successfully", "node_count": canvas.node_count}


# ==========================================
# ENDPOINTS — CONNECTIONS
# ==========================================

@router.post("/canvases/{canvas_id}/connections", status_code=200)
def create_connection(
    canvas_id: uuid.UUID,
    payload: ConnectionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    canvas = _get_owned_canvas(db, canvas_id, current_user)
    if payload.from_node_id == payload.to_node_id:
        raise HTTPException(status_code=400, detail="A concept can't link to itself.")

    _get_node(db, canvas.id, payload.from_node_id)
    _get_node(db, canvas.id, payload.to_node_id)

    existing = db.exec(
        select(CanvasConnection).where(
            CanvasConnection.canvas_id == canvas.id,
            or_(
                (CanvasConnection.from_node_id == payload.from_node_id)
                & (CanvasConnection.to_node_id == payload.to_node_id),
                (CanvasConnection.from_node_id == payload.to_node_id)
                & (CanvasConnection.to_node_id == payload.from_node_id),
            ),
        )
    ).first()
    if existing:
        return _conn_dict(existing)

    conn = CanvasConnection(
        canvas_id=canvas.id,
        from_node_id=payload.from_node_id,
        to_node_id=payload.to_node_id,
        label=_clean_text(payload.label, EDGE_LABEL_MAX),
    )
    db.add(conn)
    db.commit()
    db.refresh(conn)
    return _conn_dict(conn)


@router.delete("/canvases/{canvas_id}/connections/{connection_id}", status_code=200)
def delete_connection(
    canvas_id: uuid.UUID,
    connection_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    canvas = _get_owned_canvas(db, canvas_id, current_user)
    conn = db.exec(
        select(CanvasConnection).where(
            CanvasConnection.id == connection_id,
            CanvasConnection.canvas_id == canvas.id,
        )
    ).first()
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found")
    db.delete(conn)
    db.commit()
    return {"message": "Connection deleted"}