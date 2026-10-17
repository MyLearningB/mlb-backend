import uuid
import json
import networkx as nx
import os
import io
from pypdf import PdfReader
import docx

from openai import OpenAI  # FIXED: Switched to synchronous client
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, UploadFile, File, Form
from sqlmodel import Session, select, or_
from typing import List, Optional

# Import your schemas and models
from schemas import CanvasCreate, CanvasResponse, CanvasStatusResponse, NodeCreate, NodeResponse, NodeUpdate
from models import Canvas, CanvasNode, CanvasConnection, CanvasSourceType, Collection, CollectionAccess, CollectionItem, Note, User
from security import get_current_user
from database import get_session, engine  # FIXED: Imported engine for the background task

router = APIRouter(tags=["Canvas Section"])

# FIXED: Initialize the Synchronous Gemini Client
gemini_client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"), 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# ==========================================
# HELPER FUNCTIONS & BACKGROUND TASKS
# ==========================================

# FIXED: Made synchronous to avoid blocking event loop with mixed IO
def extract_text_from_file(file: UploadFile) -> str:
    """Reads PDF, DOCX, or TXT directly from memory without saving to disk."""
    content = file.file.read() # Synchronous read
    text = ""
    
    try:
        if file.filename.lower().endswith(".pdf"):
            pdf = PdfReader(io.BytesIO(content))
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        elif file.filename.lower().endswith(".docx"):
            doc = docx.Document(io.BytesIO(content))
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file.filename.lower().endswith(".txt"):
            text = content.decode("utf-8")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use PDF, DOCX, or TXT.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse file: {str(e)}")
        
    return text.strip()


# FIXED: Made synchronous and handles its own independent DB session
def process_note_to_canvas_bg(canvas_id: uuid.UUID, note_content: str):
    print(f"Starting Gemini processing for canvas {canvas_id}...")
    
    with Session(engine) as db:
        canvas = db.exec(select(Canvas).where(Canvas.id == canvas_id)).first()
        if not canvas:
            print("Canvas not found, aborting background task.")
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
            
        except Exception as e:
            print(f"Gemini Error: {e}")
            return

        G = nx.Graph()
        
        G.add_node("hero")
        for concept in ai_data.get("concepts", []):
            G.add_node(concept["id"])
            
        for rel in ai_data.get("relationships", []):
            G.add_edge(rel["from_id"], rel["to_id"])
            
        positions = nx.spring_layout(G, center=(1500, 1500), scale=600, seed=42)

        id_map = {} 
        
        hero_uuid = uuid.uuid4()
        id_map["hero"] = hero_uuid
        hero_node = CanvasNode(
            id=hero_uuid,
            canvas_id=canvas_id,
            label=ai_data.get("hero_concept", "Main Topic")[:40],
            x=float(positions["hero"][0]),
            y=float(positions["hero"][1]),
            size="large",
            is_hero=True
        )
        db.add(hero_node)
        
        for concept in ai_data.get("concepts", []):
            node_uuid = uuid.uuid4()
            id_map[concept["id"]] = node_uuid
            
            node = CanvasNode(
                id=node_uuid,
                canvas_id=canvas_id,
                label=concept["label"][:40],
                x=float(positions[concept["id"]][0]),
                y=float(positions[concept["id"]][1]),
                size="medium",
                is_weak=concept.get("is_weak", False),
                definition=concept.get("definition")
            )
            db.add(node)
            
        for rel in ai_data.get("relationships", []):
            from_uuid = id_map.get(rel["from_id"])
            to_uuid = id_map.get(rel["to_id"])
            
            if from_uuid and to_uuid:
                connection = CanvasConnection(
                    canvas_id=canvas_id,
                    from_node_id=from_uuid,
                    to_node_id=to_uuid,
                    label=rel.get("label")
                )
                db.add(connection)
                
        canvas.node_count = len(ai_data.get("concepts", [])) + 1
        db.commit()
        print(f"Successfully processed and mapped canvas {canvas_id}!")


# ==========================================
# API ENDPOINTS
# ==========================================

# 1. GET ALL CANVASES
@router.get("/users/me/canvases", status_code=200)
def get_user_canvases(
    search: Optional[str] = "",
    filter: Optional[str] = "All",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """Fetch user's canvases with search and filter capabilities."""
    query = select(Canvas).where(Canvas.user_id == current_user.id)
    
    if search:
        query = query.where((Canvas.name.icontains(search)) | (Canvas.subject.icontains(search)))
        
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
    db: Session = Depends(get_session)
):
    if not payload.note_id:
        raise HTTPException(status_code=400, detail="note_id is required")
        
    note = db.exec(select(Note).where(Note.id == payload.note_id, Note.user_id == current_user.id)).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
        
    db_canvas = Canvas(
        user_id=current_user.id,
        name=payload.name,
        subject=payload.subject,
        source_type=CanvasSourceType.notes,
        source_id=payload.note_id
    )
    db.add(db_canvas)
    db.commit()
    db.refresh(db_canvas)

    # FIXED: Do not pass the `db` session into the background task
    background_tasks.add_task(process_note_to_canvas_bg, db_canvas.id, note.content_text)
    return {"status": "processing", "node_count": 0, "nodes": []}

# 3. CREATE CANVAS FROM FILE UPLOAD (AI)
@router.post("/canvases/from-file", response_model=CanvasStatusResponse)
def create_canvas_from_file(  # FIXED: Removed async
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    subject: str = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    document_text = extract_text_from_file(file)
    
    if not document_text:
        raise HTTPException(status_code=400, detail="Could not extract any text from the uploaded file.")

    db_canvas = Canvas(
        user_id=current_user.id,
        name=name,
        subject=subject,
        source_type=CanvasSourceType.manual,
        source_id=None
    )
    db.add(db_canvas)
    db.commit()
    db.refresh(db_canvas)

    # FIXED: Do not pass the `db` session into the background task
    background_tasks.add_task(process_note_to_canvas_bg, db_canvas.id, document_text)
    
    return {"status": "processing", "node_count": 0, "nodes": []}

# 4. GET CANVAS STATUS / CONTENT
@router.get("/canvases/{canvas_id}/status", response_model=CanvasStatusResponse)
def get_canvas_status(
    canvas_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    canvas = db.exec(select(Canvas).where(Canvas.id == canvas_id, Canvas.user_id == current_user.id)).first()
    if not canvas: 
        raise HTTPException(status_code=404, detail="Canvas not found")
    
    if canvas.node_count > 0:
        return {"status": "ready", "node_count": canvas.node_count, "nodes": canvas.nodes}
    else:
        return {"status": "processing", "node_count": 0, "nodes": []}

# 5. CREATE MANUAL CANVAS
@router.post("/canvases", response_model=CanvasResponse)
def create_manual_canvas(
    payload: CanvasCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    db_canvas = Canvas(
        user_id=current_user.id,
        name=payload.name,
        subject=payload.subject,
        source_type=CanvasSourceType.manual
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
    db: Session = Depends(get_session)
):
    canvas = db.exec(select(Canvas).where(Canvas.id == canvas_id, Canvas.user_id == current_user.id)).first()
    if not canvas:
        raise HTTPException(status_code=404, detail="Canvas not found")

    db_node = CanvasNode(
        canvas_id=canvas_id,
        label=node.label,
        x=node.x,
        y=node.y,
        size=node.size,
        is_hero=node.is_hero,
        is_weak=node.is_weak
    )
    db.add(db_node)
    
    canvas.node_count += 1
    db.commit()
    db.refresh(db_node)
    return db_node
 
# 7. GET SINGLE CANVAS
@router.get("/canvases/{canvas_id}", status_code=200)
def get_single_canvas(  # FIXED: Removed async
    canvas_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
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
            .outerjoin(CollectionAccess, CollectionAccess.collection_id == Collection.id)
            .where(
                CollectionItem.item_type == "canvas",
                CollectionItem.item_id == str(canvas.id),
                or_(
                    Collection.visibility == "public",
                    CollectionAccess.user_id == current_user.id
                )
            )
        ).first()

        if has_access:
            is_owner = False
        else:
            raise HTTPException(status_code=404, detail="Canvas not found")

    nodes = db.exec(select(CanvasNode).where(CanvasNode.canvas_id == canvas.id)).all()
    connections = db.exec(select(CanvasConnection).where(CanvasConnection.canvas_id == canvas.id)).all()

    response_data = canvas.model_dump()
    response_data["is_owner"] = is_owner
    response_data["nodes"] = nodes
    response_data["connections"] = connections

    return response_data

# 8. DELETE CANVAS (SAFE CASCADE DELETE)
@router.delete("/canvases/{canvas_id}", status_code=200)
def delete_canvas(
    canvas_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    canvas = db.exec(select(Canvas).where(Canvas.id == canvas_id, Canvas.user_id == current_user.id)).first()
    if not canvas:
        raise HTTPException(status_code=404, detail="Canvas not found")
        
    connections = db.exec(select(CanvasConnection).where(CanvasConnection.canvas_id == canvas_id)).all()
    for conn in connections:
        db.delete(conn)

    nodes = db.exec(select(CanvasNode).where(CanvasNode.canvas_id == canvas_id)).all()
    for node in nodes:
        db.delete(node)

    db.delete(canvas)
    db.commit()
    
    return {"message": "Canvas and all its nodes deleted successfully"}

# --------------------------------------------------
# RENAME / MOVE A CANVAS NODE
# --------------------------------------------------
@router.patch("/canvases/{canvas_id}/nodes/{node_id}", status_code=200)
def update_canvas_node(
    canvas_id: uuid.UUID,
    node_id: uuid.UUID,
    payload: NodeUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    # 1. Verify the user actually owns this canvas
    canvas = db.exec(select(Canvas).where(Canvas.id == canvas_id, Canvas.user_id == current_user.id)).first()
    if not canvas:
        raise HTTPException(status_code=404, detail="Canvas not found")
        
    # 2. Find the specific node
    node = db.exec(select(CanvasNode).where(CanvasNode.id == node_id, CanvasNode.canvas_id == canvas_id)).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
        
    # 3. Safely apply only the fields provided by the Flutter app
    update_data = payload.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(node, key, value)
        
    db.add(node)
    db.commit()
    db.refresh(node)
    
    return node


# --------------------------------------------------
# DELETE A CANVAS NODE
# --------------------------------------------------
@router.delete("/canvases/{canvas_id}/nodes/{node_id}", status_code=200)
def delete_canvas_node(
    canvas_id: uuid.UUID,
    node_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    # 1. Verify ownership
    canvas = db.exec(select(Canvas).where(Canvas.id == canvas_id, Canvas.user_id == current_user.id)).first()
    if not canvas:
        raise HTTPException(status_code=404, detail="Canvas not found")
        
    # 2. Find the node
    node = db.exec(select(CanvasNode).where(CanvasNode.id == node_id, CanvasNode.canvas_id == canvas_id)).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
        
    # 3. Clean up any Connections (Lines) attached to this node
    # This prevents invisible "ghost lines" from crashing the UI renderer later
    connections = db.exec(
        select(CanvasConnection)
        .where((CanvasConnection.from_node_id == node_id) | (CanvasConnection.to_node_id == node_id))
    ).all()
    
    for conn in connections:
        db.delete(conn)
        
    # 4. Delete the node itself
    db.delete(node)
    
    # 5. Keep the parent Canvas metadata accurate
    if canvas.node_count > 0:
        canvas.node_count -= 1
        db.add(canvas)
        
    db.commit()
    
    return {"message": "Node deleted successfully"}