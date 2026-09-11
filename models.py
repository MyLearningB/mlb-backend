import uuid
from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
from datetime import date, datetime, timezone
from enum import Enum

# ==========================================
# HELPER FUNCTION (FIXED TIMEZONES)
# ==========================================
def utc_now():
    """Generates a timezone-aware UTC datetime. Replaces deprecated datetime.utcnow()"""
    return datetime.now(timezone.utc)

# ==========================================
# ENUMS
# ==========================================

class DifficultyLevel(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"

class CanvasSourceType(str, Enum):
    notes = "notes"
    upload = "upload"
    manual = "manual"

class NodeSize(str, Enum):
    small = "small"
    medium = "medium"
    large = "large"

# ==========================================
# MULTIPLAYER LINK MODELS (Define first for relationships)
# ==========================================

class GroupMember(SQLModel, table=True):
    group_id: uuid.UUID = Field(foreign_key="studygroup.id", primary_key=True)
    user_id: int = Field(foreign_key="user.id", primary_key=True)

class UserRelic(SQLModel, table=True):
    user_id: int = Field(foreign_key="user.id", primary_key=True)
    relic_id: uuid.UUID = Field(foreign_key="relic.id", primary_key=True)
    unlocked_at: datetime = Field(default_factory=utc_now) # FIXED

# ==========================================
# USER & PET MODELS (ONBOARDING)
# ==========================================

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    email: str = Field(unique=True, index=True)
    hashed_password: str
    is_verified: bool = False
    token_version: int = Field(default=1)

    username: Optional[str] = Field(default=None, unique=True, index=True)
    bio: Optional[str] = Field(default=None, max_length=120)
    study_goal: Optional[str] = None

    # True until onboarding is successfully completed.
    is_first_session: bool = True

    fcm_token: Optional[str] = Field(default=None)

    srs_intensity: str = Field(default="Standard")
    daily_goal_mins: int = Field(default=30)
    public_profile: bool = Field(default=True)
    push_notifications: bool = Field(default=True)
    quest_updates: bool = Field(default=True)
    access_requests_alerts: bool = Field(default=True)

    current_streak: int = Field(default=0)
    longest_streak: int = Field(default=0)
    last_active_date: Optional[date] = Field(default=None)

    pets: List["Pet"] = Relationship(
        back_populates="user",
        cascade_delete=True
    )

    quests: List["Quest"] = Relationship(
        back_populates="user",
        cascade_delete=True
    )

    study_plans: List["StudyPlan"] = Relationship(
        back_populates="user",
        cascade_delete=True
    )

    study_sets: List["StudySet"] = Relationship(
        back_populates="user",
        cascade_delete=True
    )

    notes: List["Note"] = Relationship(
        back_populates="user",
        cascade_delete=True
    )

    canvases: List["Canvas"] = Relationship(
        back_populates="user",
        cascade_delete=True
    )

    reminders: List["Reminder"] = Relationship(
        back_populates="user",
        cascade_delete=True
    )

    study_groups: List["StudyGroup"] = Relationship(
        back_populates="members",
        link_model=GroupMember
    )

    relics: List["Relic"] = Relationship(
        back_populates="users",
        link_model=UserRelic
    )

class Pet(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", ondelete="CASCADE")
    
    pet_type: str
    pet_name: str
    level: int = 1
    xp: int = 0
    
    user: Optional["User"] = Relationship(back_populates="pets")

# ==========================================
# DASHBOARD, STUDY PLAN & REMINDER MODELS
# ==========================================

class Reminder(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", ondelete="CASCADE")
    
    type: str 
    title: str
    schedule: str 
    time: Optional[str] = None 
    days_of_week: Optional[str] = None 
    enabled: bool = True
    color: str 
    
    user: Optional["User"] = Relationship(back_populates="reminders")

class Quest(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", ondelete="CASCADE")
    
    title: str
    type: str 
    progress: int = 0
    target: int
    members_count: Optional[int] = None
    
    user: Optional["User"] = Relationship(back_populates="quests")

class StudyPlan(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", ondelete="CASCADE")
    
    subject: str
    deadline: date
    is_approved: bool = False
    
    user: Optional["User"] = Relationship(back_populates="study_plans")
    sessions: List["StudySession"] = Relationship(back_populates="plan", cascade_delete=True)

class StudySession(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    plan_id: int = Field(foreign_key="studyplan.id", ondelete="CASCADE")
    user_id: int = Field(foreign_key="user.id", ondelete="CASCADE")
    
    date: str 
    time: Optional[str] = None 
    subject: str
    duration_mins: int
    mode: str 
    priority: str 
    completed: bool = False
    skipped: bool = False
    
    plan: Optional["StudyPlan"] = Relationship(back_populates="sessions")

class DailyActivity(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", ondelete="CASCADE")
    
    date: str 
    xp_earned: int = 0

# ==========================================
# STUDY TAB & NOTES MODELS
# ==========================================

class StudySet(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", ondelete="CASCADE")
    
    title: str
    subject: str
    card_count: int = Field(default=0)
    last_studied: Optional[datetime] = Field(default=None)
    weak_cards_count: int = Field(default=0)

    user: Optional["User"] = Relationship(back_populates="study_sets")
    flashcards: List["Flashcard"] = Relationship(back_populates="study_set", cascade_delete=True)
    feynman_sessions: List["FeynmanSession"] = Relationship(back_populates="study_set", cascade_delete=True)

class Flashcard(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    study_set_id: int = Field(foreign_key="studyset.id", ondelete="CASCADE")
    note_id: Optional[int] = Field(default=None, foreign_key="note.id", ondelete="CASCADE")
    
    question: str = Field(max_length=200)
    answer: str = Field(max_length=400)
    subject: str
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.medium)
    is_weak: bool = Field(default=False)

    study_set: Optional["StudySet"] = Relationship(back_populates="flashcards")
    note: Optional["Note"] = Relationship(back_populates="flashcards")
    feynman_sessions: List["FeynmanSession"] = Relationship(back_populates="flashcard")
    canvas_nodes: List["CanvasNode"] = Relationship(back_populates="flashcard")

class FeynmanSession(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", ondelete="CASCADE")
    study_set_id: int = Field(foreign_key="studyset.id", ondelete="CASCADE")
    card_id: int = Field(foreign_key="flashcard.id", ondelete="CASCADE")
    
    comprehension_score: int = Field(default=0, ge=0, le=100)
    is_complete: bool = Field(default=False)
    
    gaps_identified: str = Field(default="[]") 
    strong_points: str = Field(default="[]")
    
    created_at: datetime = Field(default_factory=utc_now) # FIXED

    study_set: Optional["StudySet"] = Relationship(back_populates="feynman_sessions")
    flashcard: Optional["Flashcard"] = Relationship(back_populates="feynman_sessions")

class Note(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", ondelete="CASCADE")
    
    title: str = Field(default="Untitled note", max_length=60)
    subject: str
    content_text: str = Field(default="")
    content_html: Optional[str] = Field(default=None)
    
    word_count: int = Field(default=0)
    card_count: int = Field(default=0)
    weak_card_count: int = Field(default=0)
    has_canvas: bool = Field(default=False)
    snippet: str = Field(default="", max_length=80)
    is_public: bool = Field(default=False)
    
    created_at: datetime = Field(default_factory=utc_now) # FIXED
    updated_at: datetime = Field(default_factory=utc_now) # FIXED

    user: Optional["User"] = Relationship(back_populates="notes")
    flashcards: List["Flashcard"] = Relationship(back_populates="note", cascade_delete=True)
    canvases: List["Canvas"] = Relationship(back_populates="note")

# ==========================================
# CANVAS MODELS
# ==========================================

class Canvas(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: int = Field(foreign_key="user.id", ondelete="CASCADE")
    
    name: str
    subject: str
    node_count: int = Field(default=0)
    weak_node_count: int = Field(default=0)
    thumbnail_url: Optional[str] = Field(default=None)
    
    source_type: CanvasSourceType = Field(default=CanvasSourceType.manual)
    source_id: Optional[int] = Field(default=None, foreign_key="note.id", ondelete="SET NULL")
    
    last_studied_at: Optional[datetime] = Field(default=None)
    created_at: datetime = Field(default_factory=utc_now) # FIXED
    is_public: bool = Field(default=False)
    
    user: Optional["User"] = Relationship(back_populates="canvases")
    note: Optional["Note"] = Relationship(back_populates="canvases")
    nodes: List["CanvasNode"] = Relationship(back_populates="canvas", cascade_delete=True)
    connections: List["CanvasConnection"] = Relationship(back_populates="canvas", cascade_delete=True)

class CanvasNode(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    canvas_id: uuid.UUID = Field(foreign_key="canvas.id", ondelete="CASCADE")
    
    label: str = Field(max_length=40)
    x: float
    y: float
    size: NodeSize = Field(default=NodeSize.medium)
    is_hero: bool = Field(default=False)
    is_weak: bool = Field(default=False)
    definition: Optional[str] = Field(default=None)
    
    card_id: Optional[int] = Field(default=None, foreign_key="flashcard.id", ondelete="SET NULL")
    
    canvas: Optional["Canvas"] = Relationship(back_populates="nodes")
    flashcard: Optional["Flashcard"] = Relationship(back_populates="canvas_nodes")
    
    outgoing_connections: List["CanvasConnection"] = Relationship(
        back_populates="from_node",
        sa_relationship_kwargs={"foreign_keys": "[CanvasConnection.from_node_id]", "cascade": "all, delete-orphan"}
    )
    incoming_connections: List["CanvasConnection"] = Relationship(
        back_populates="to_node",
        sa_relationship_kwargs={"foreign_keys": "[CanvasConnection.to_node_id]", "cascade": "all, delete-orphan"}
    )

class CanvasConnection(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    canvas_id: uuid.UUID = Field(foreign_key="canvas.id", ondelete="CASCADE")
    from_node_id: uuid.UUID = Field(foreign_key="canvasnode.id", ondelete="CASCADE")
    to_node_id: uuid.UUID = Field(foreign_key="canvasnode.id", ondelete="CASCADE")
    
    label: Optional[str] = Field(default=None)
    
    canvas: Optional["Canvas"] = Relationship(back_populates="connections")
    from_node: Optional["CanvasNode"] = Relationship(
        back_populates="outgoing_connections",
        sa_relationship_kwargs={"foreign_keys": "[CanvasConnection.from_node_id]"}
    )
    to_node: Optional["CanvasNode"] = Relationship(
        back_populates="incoming_connections",
        sa_relationship_kwargs={"foreign_keys": "[CanvasConnection.to_node_id]"}
    )

# ==========================================
# COLLECTIONS & NOTIFICATIONS
# ==========================================

class Collection(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", ondelete="CASCADE")
    
    title: str
    description: Optional[str] = None
    subject: str
    cover_emoji: Optional[str] = None
    
    visibility: str = Field(default="private") 
    share_token: str = Field(default_factory=lambda: uuid.uuid4().hex[:12], unique=True)
    save_count: int = Field(default=0)
    
    created_at: datetime = Field(default_factory=utc_now) # FIXED
    updated_at: datetime = Field(default_factory=utc_now) # FIXED

class CollectionItem(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    collection_id: int = Field(foreign_key="collection.id", ondelete="CASCADE")
    
    item_type: str 
    item_id: str 
    position: int = Field(default=0) 

class CollectionAccess(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    collection_id: int = Field(foreign_key="collection.id", ondelete="CASCADE")
    user_id: int = Field(foreign_key="user.id", ondelete="CASCADE") 
    granted_at: datetime = Field(default_factory=utc_now) # FIXED

class CollectionRequest(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    collection_id: int = Field(foreign_key="collection.id", ondelete="CASCADE")
    user_id: int = Field(foreign_key="user.id", ondelete="CASCADE") 
    message: Optional[str] = None
    status: str = Field(default="pending") 
    requested_at: datetime = Field(default_factory=utc_now) # FIXED

class Notification(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", ondelete="CASCADE") 
    
    title: str
    body: str
    deep_link: Optional[str] = None 
    is_read: bool = Field(default=False)
    
    created_at: datetime = Field(default_factory=utc_now) # FIXED

class UserTrophy(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", ondelete="CASCADE")
    
    trophy_id: str 
    earned_at: datetime = Field(default_factory=utc_now) # FIXED

class Feedback(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", ondelete="CASCADE")
    message: str = Field(max_length=1000)
    status: str = Field(default="unread") 
    created_at: datetime = Field(default_factory=utc_now) # FIXED

# ==========================================
# NEW: MULTIPLAYER & CO-OP MODELS 
# ==========================================

class StudyGroup(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str
    invite_code: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=utc_now) # FIXED
    
    # Relationships
    members: List["User"] = Relationship(back_populates="study_groups", link_model=GroupMember)
    active_quests: List["CoopQuest"] = Relationship(back_populates="group", cascade_delete=True)

class CoopQuest(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    group_id: uuid.UUID = Field(foreign_key="studygroup.id", ondelete="CASCADE")
    title: str
    target_xp: int
    current_xp: int = Field(default=0)
    is_completed: bool = Field(default=False)
    
    group: Optional["StudyGroup"] = Relationship(back_populates="active_quests")

class Relic(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str
    description: str

    users: List["User"] = Relationship(back_populates="relics", link_model=UserRelic)