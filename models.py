import uuid
from datetime import date, datetime, timezone
from enum import Enum
from typing import List, Optional

from sqlmodel import SQLModel, Field, Relationship


# ==========================================
# HELPER
# ==========================================

def utc_now() -> datetime:
    """Return a timezone-aware UTC datetime."""
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
# MULTIPLAYER LINK MODELS
# ==========================================

class GroupMember(SQLModel, table=True):
    group_id: uuid.UUID = Field(
        foreign_key="studygroup.id",
        primary_key=True,
        ondelete="CASCADE",
    )

    user_id: int = Field(
        foreign_key="user.id",
        primary_key=True,
        ondelete="CASCADE",
    )


class UserRelic(SQLModel, table=True):
    user_id: int = Field(
        foreign_key="user.id",
        primary_key=True,
        ondelete="CASCADE",
    )

    relic_id: uuid.UUID = Field(
        foreign_key="relic.id",
        primary_key=True,
        ondelete="CASCADE",
    )

    unlocked_at: datetime = Field(
        default_factory=utc_now,
    )


# ==========================================
# USER
# ==========================================

class User(SQLModel, table=True):
    id: Optional[int] = Field(
        default=None,
        primary_key=True,
    )

    name: str = Field(
        min_length=2,
        max_length=60,
    )

    email: str = Field(
        unique=True,
        index=True,
    )

    hashed_password: str

    is_verified: bool = Field(
        default=False,
    )

    token_version: int = Field(
        default=1,
    )

    username: Optional[str] = Field(
        default=None,
        unique=True,
        index=True,
    )

    bio: Optional[str] = Field(
        default=None,
        max_length=120,
    )

    # --------------------------------------
    # ONBOARDING
    # --------------------------------------

    study_goal: Optional[str] = None

    goal_type: Optional[str] = None

    target_date: Optional[date] = None

    is_first_session: bool = Field(
        default=True,
    )

    # --------------------------------------
    # NOTIFICATIONS
    # --------------------------------------

    fcm_token: Optional[str] = None

    srs_intensity: str = Field(
        default="Standard",
    )

    daily_goal_mins: int = Field(
        default=30,
        ge=0,
    )

    public_profile: bool = Field(
        default=True,
    )

    push_notifications: bool = Field(
        default=True,
    )

    quest_updates: bool = Field(
        default=True,
    )

    access_requests_alerts: bool = Field(
        default=True,
    )

    # --------------------------------------
    # STREAKS
    # --------------------------------------

    current_streak: int = Field(
        default=0,
        ge=0,
    )

    longest_streak: int = Field(
        default=0,
        ge=0,
    )

    last_active_date: Optional[date] = None

    # ======================================
    # RELATIONSHIPS
    # ======================================

    pets: List["Pet"] = Relationship(
        back_populates="user",
        cascade_delete=True,
    )

    quests: List["Quest"] = Relationship(
        back_populates="user",
        cascade_delete=True,
    )

    study_plans: List["StudyPlan"] = Relationship(
        back_populates="user",
        cascade_delete=True,
    )

    study_sessions: List["StudySession"] = Relationship(
        back_populates="user",
        cascade_delete=True,
    )

    study_sets: List["StudySet"] = Relationship(
        back_populates="user",
        cascade_delete=True,
    )

    notes: List["Note"] = Relationship(
        back_populates="user",
        cascade_delete=True,
    )

    canvases: List["Canvas"] = Relationship(
        back_populates="user",
        cascade_delete=True,
    )

    reminders: List["Reminder"] = Relationship(
        back_populates="user",
        cascade_delete=True,
    )

    daily_activities: List["DailyActivity"] = Relationship(
        back_populates="user",
        cascade_delete=True,
    )

    notifications: List["Notification"] = Relationship(
        back_populates="user",
        cascade_delete=True,
    )

    collections: List["Collection"] = Relationship(
        back_populates="user",
        cascade_delete=True,
    )

    collection_access: List["CollectionAccess"] = Relationship(
        back_populates="user",
        cascade_delete=True,
    )

    collection_requests: List["CollectionRequest"] = Relationship(
        back_populates="user",
        cascade_delete=True,
    )

    study_groups: List["StudyGroup"] = Relationship(
        back_populates="members",
        link_model=GroupMember,
    )

    relics: List["Relic"] = Relationship(
        back_populates="users",
        link_model=UserRelic,
    )

    trophies: List["UserTrophy"] = Relationship(
        back_populates="user",
        cascade_delete=True,
    )


# ==========================================
# PET
# ==========================================

class Pet(SQLModel, table=True):
    id: Optional[int] = Field(
        default=None,
        primary_key=True,
    )

    user_id: int = Field(
        foreign_key="user.id",
        ondelete="CASCADE",
        index=True,
    )

    pet_type: str

    pet_name: str

    level: int = Field(
        default=1,
        ge=1,
    )

    xp: int = Field(
        default=0,
        ge=0,
    )

    user: Optional["User"] = Relationship(
        back_populates="pets",
    )


# ==========================================
# REMINDERS
# ==========================================

class Reminder(SQLModel, table=True):
    id: Optional[int] = Field(
        default=None,
        primary_key=True,
    )

    user_id: int = Field(
        foreign_key="user.id",
        ondelete="CASCADE",
        index=True,
    )

    type: str

    title: str

    schedule: str

    time: Optional[str] = None

    days_of_week: Optional[str] = None

    enabled: bool = Field(
        default=True,
    )

    color: str

    user: Optional["User"] = Relationship(
        back_populates="reminders",
    )


# ==========================================
# QUEST
# ==========================================

class Quest(SQLModel, table=True):
    id: Optional[int] = Field(
        default=None,
        primary_key=True,
    )

    user_id: int = Field(
        foreign_key="user.id",
        ondelete="CASCADE",
        index=True,
    )

    title: str

    type: str

    progress: int = Field(
        default=0,
        ge=0,
    )

    target: int = Field(
        ge=1,
    )

    members_count: Optional[int] = Field(
        default=None,
        ge=0,
    )

    user: Optional["User"] = Relationship(
        back_populates="quests",
    )


# ==========================================
# STUDY PLAN
# ==========================================

class StudyPlan(SQLModel, table=True):
    id: Optional[int] = Field(
        default=None,
        primary_key=True,
    )

    user_id: int = Field(
        foreign_key="user.id",
        ondelete="CASCADE",
        index=True,
    )

    subject: str

    deadline: Optional[date] = None

    is_approved: bool = Field(
        default=False,
    )

    user: Optional["User"] = Relationship(
        back_populates="study_plans",
    )

    sessions: List["StudySession"] = Relationship(
        back_populates="plan",
        cascade_delete=True,
    )


class StudySession(SQLModel, table=True):
    id: Optional[int] = Field(
        default=None,
        primary_key=True,
    )

    plan_id: int = Field(
        foreign_key="studyplan.id",
        ondelete="CASCADE",
        index=True,
    )

    user_id: int = Field(
        foreign_key="user.id",
        ondelete="CASCADE",
        index=True,
    )

    date: str

    time: Optional[str] = None

    subject: str

    duration_mins: int = Field(
        ge=1,
    )

    mode: str

    priority: str

    completed: bool = Field(
        default=False,
    )

    skipped: bool = Field(
        default=False,
    )

    plan: Optional["StudyPlan"] = Relationship(
        back_populates="sessions",
    )

    user: Optional["User"] = Relationship(
        back_populates="study_sessions",
    )


class DailyActivity(SQLModel, table=True):
    id: Optional[int] = Field(
        default=None,
        primary_key=True,
    )

    user_id: int = Field(
        foreign_key="user.id",
        ondelete="CASCADE",
        index=True,
    )

    date: str

    xp_earned: int = Field(
        default=0,
        ge=0,
    )

    user: Optional["User"] = Relationship(
        back_populates="daily_activities",
    )


# ==========================================
# STUDY SETS
# ==========================================

class StudySet(SQLModel, table=True):
    id: Optional[int] = Field(
        default=None,
        primary_key=True,
    )

    user_id: int = Field(
        foreign_key="user.id",
        ondelete="CASCADE",
        index=True,
    )

    title: str

    subject: str

    card_count: int = Field(
        default=0,
        ge=0,
    )

    last_studied: Optional[datetime] = None

    weak_cards_count: int = Field(
        default=0,
        ge=0,
    )

    user: Optional["User"] = Relationship(
        back_populates="study_sets",
    )

    flashcards: List["Flashcard"] = Relationship(
        back_populates="study_set",
        cascade_delete=True,
    )

    feynman_sessions: List["FeynmanSession"] = Relationship(
        back_populates="study_set",
        cascade_delete=True,
    )


# ==========================================
# FLASHCARDS
# ==========================================

class Flashcard(SQLModel, table=True):
    id: Optional[int] = Field(
        default=None,
        primary_key=True,
    )

    study_set_id: int = Field(
        foreign_key="studyset.id",
        ondelete="CASCADE",
        index=True,
    )

    note_id: Optional[int] = Field(
        default=None,
        foreign_key="note.id",
        ondelete="CASCADE",
        index=True,
    )

    question: str = Field(
        max_length=200,
    )

    answer: str = Field(
        max_length=400,
    )

    subject: str

    difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.medium,
    )

    is_weak: bool = Field(
        default=False,
    )

    study_set: Optional["StudySet"] = Relationship(
        back_populates="flashcards",
    )

    note: Optional["Note"] = Relationship(
        back_populates="flashcards",
    )

    feynman_sessions: List["FeynmanSession"] = Relationship(
        back_populates="flashcard",
    )

    canvas_nodes: List["CanvasNode"] = Relationship(
        back_populates="flashcard",
    )


# ==========================================
# FEYNMAN
# ==========================================

class FeynmanSession(SQLModel, table=True):
    id: Optional[int] = Field(
        default=None,
        primary_key=True,
    )

    user_id: int = Field(
        foreign_key="user.id",
        ondelete="CASCADE",
        index=True,
    )

    study_set_id: int = Field(
        foreign_key="studyset.id",
        ondelete="CASCADE",
        index=True,
    )

    card_id: int = Field(
        foreign_key="flashcard.id",
        ondelete="CASCADE",
        index=True,
    )

    comprehension_score: int = Field(
        default=0,
        ge=0,
        le=100,
    )

    is_complete: bool = Field(
        default=False,
    )

    gaps_identified: str = Field(
        default="[]",
    )

    strong_points: str = Field(
        default="[]",
    )

    created_at: datetime = Field(
        default_factory=utc_now,
    )

    study_set: Optional["StudySet"] = Relationship(
        back_populates="feynman_sessions",
    )

    flashcard: Optional["Flashcard"] = Relationship(
        back_populates="feynman_sessions",
    )


# ==========================================
# NOTES
# ==========================================

class Note(SQLModel, table=True):
    id: Optional[int] = Field(
        default=None,
        primary_key=True,
    )

    user_id: int = Field(
        foreign_key="user.id",
        ondelete="CASCADE",
        index=True,
    )

    title: str = Field(
        default="Untitled note",
        max_length=60,
    )

    subject: str

    content_text: str = Field(
        default="",
    )

    content_html: Optional[str] = None

    word_count: int = Field(
        default=0,
        ge=0,
    )

    card_count: int = Field(
        default=0,
        ge=0,
    )

    weak_card_count: int = Field(
        default=0,
        ge=0,
    )

    has_canvas: bool = Field(
        default=False,
    )

    snippet: str = Field(
        default="",
        max_length=80,
    )

    is_public: bool = Field(
        default=False,
    )

    created_at: datetime = Field(
        default_factory=utc_now,
    )

    updated_at: datetime = Field(
        default_factory=utc_now,
    )

    user: Optional["User"] = Relationship(
        back_populates="notes",
    )

    flashcards: List["Flashcard"] = Relationship(
        back_populates="note",
        cascade_delete=True,
    )

    canvases: List["Canvas"] = Relationship(
        back_populates="note",
    )


# ==========================================
# CANVAS
# ==========================================

class Canvas(SQLModel, table=True):
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
    )

    user_id: int = Field(
        foreign_key="user.id",
        ondelete="CASCADE",
        index=True,
    )

    name: str

    subject: str

    node_count: int = Field(
        default=0,
        ge=0,
    )

    weak_node_count: int = Field(
        default=0,
        ge=0,
    )

    thumbnail_url: Optional[str] = None

    source_type: CanvasSourceType = Field(
        default=CanvasSourceType.manual,
    )

    source_id: Optional[int] = Field(
        default=None,
        foreign_key="note.id",
        ondelete="SET NULL",
    )

    last_studied_at: Optional[datetime] = None

    created_at: datetime = Field(
        default_factory=utc_now,
    )

    is_public: bool = Field(
        default=False,
    )

    user: Optional["User"] = Relationship(
        back_populates="canvases",
    )

    note: Optional["Note"] = Relationship(
        back_populates="canvases",
    )

    nodes: List["CanvasNode"] = Relationship(
        back_populates="canvas",
        cascade_delete=True,
    )

    connections: List["CanvasConnection"] = Relationship(
        back_populates="canvas",
        cascade_delete=True,
    )


class CanvasNode(SQLModel, table=True):
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
    )

    canvas_id: uuid.UUID = Field(
        foreign_key="canvas.id",
        ondelete="CASCADE",
    )

    label: str = Field(
        max_length=40,
    )

    x: float

    y: float

    size: NodeSize = Field(
        default=NodeSize.medium,
    )

    is_hero: bool = Field(
        default=False,
    )

    is_weak: bool = Field(
        default=False,
    )

    definition: Optional[str] = None

    card_id: Optional[int] = Field(
        default=None,
        foreign_key="flashcard.id",
        ondelete="SET NULL",
    )

    canvas: Optional["Canvas"] = Relationship(
        back_populates="nodes",
    )

    flashcard: Optional["Flashcard"] = Relationship(
        back_populates="canvas_nodes",
    )

    outgoing_connections: List["CanvasConnection"] = Relationship(
        back_populates="from_node",
        sa_relationship_kwargs={
            "foreign_keys": "[CanvasConnection.from_node_id]",
            "cascade": "all, delete-orphan",
        },
    )

    incoming_connections: List["CanvasConnection"] = Relationship(
        back_populates="to_node",
        sa_relationship_kwargs={
            "foreign_keys": "[CanvasConnection.to_node_id]",
            "cascade": "all, delete-orphan",
        },
    )


class CanvasConnection(SQLModel, table=True):
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
    )

    canvas_id: uuid.UUID = Field(
        foreign_key="canvas.id",
        ondelete="CASCADE",
    )

    from_node_id: uuid.UUID = Field(
        foreign_key="canvasnode.id",
        ondelete="CASCADE",
    )

    to_node_id: uuid.UUID = Field(
        foreign_key="canvasnode.id",
        ondelete="CASCADE",
    )

    label: Optional[str] = None

    canvas: Optional["Canvas"] = Relationship(
        back_populates="connections",
    )

    from_node: Optional["CanvasNode"] = Relationship(
        back_populates="outgoing_connections",
        sa_relationship_kwargs={
            "foreign_keys": "[CanvasConnection.from_node_id]",
        },
    )

    to_node: Optional["CanvasNode"] = Relationship(
        back_populates="incoming_connections",
        sa_relationship_kwargs={
            "foreign_keys": "[CanvasConnection.to_node_id]",
        },
    )


# ==========================================
# COLLECTIONS
# ==========================================

class Collection(SQLModel, table=True):
    id: Optional[int] = Field(
        default=None,
        primary_key=True,
    )

    user_id: int = Field(
        foreign_key="user.id",
        ondelete="CASCADE",
        index=True,
    )

    title: str = Field(
        max_length=60,
    )

    description: Optional[str] = Field(
        default=None,
        max_length=300,
    )

    subject: str = Field(
        max_length=100,
    )

    cover_emoji: Optional[str] = Field(
        default=None,
        max_length=10,
    )

    visibility: str = Field(
        default="private",
        max_length=20,
        index=True,
    )

    share_token: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:12],
        unique=True,
        index=True,
    )

    save_count: int = Field(
        default=0,
        ge=0,
    )

    created_at: datetime = Field(
        default_factory=utc_now,
    )

    updated_at: datetime = Field(
        default_factory=utc_now,
    )

    user: Optional["User"] = Relationship(
        back_populates="collections",
    )

    items: List["CollectionItem"] = Relationship(
        back_populates="collection",
        cascade_delete=True,
    )

    access_list: List["CollectionAccess"] = Relationship(
        back_populates="collection",
        cascade_delete=True,
    )

    requests: List["CollectionRequest"] = Relationship(
        back_populates="collection",
        cascade_delete=True,
    )


class CollectionItem(SQLModel, table=True):
    id: Optional[int] = Field(
        default=None,
        primary_key=True,
    )

    collection_id: int = Field(
        foreign_key="collection.id",
        ondelete="CASCADE",
        index=True,
    )

    item_type: str = Field(
        max_length=20,
    )

    item_id: str = Field(
        max_length=100,
    )

    position: int = Field(
        default=0,
        ge=0,
    )

    collection: Optional["Collection"] = Relationship(
        back_populates="items",
    )


class CollectionAccess(SQLModel, table=True):
    id: Optional[int] = Field(
        default=None,
        primary_key=True,
    )

    collection_id: int = Field(
        foreign_key="collection.id",
        ondelete="CASCADE",
        index=True,
    )

    user_id: int = Field(
        foreign_key="user.id",
        ondelete="CASCADE",
        index=True,
    )

    granted_at: datetime = Field(
        default_factory=utc_now,
    )

    collection: Optional["Collection"] = Relationship(
        back_populates="access_list",
    )

    user: Optional["User"] = Relationship(
        back_populates="collection_access",
    )


class CollectionRequest(SQLModel, table=True):
    id: Optional[int] = Field(
        default=None,
        primary_key=True,
    )

    collection_id: int = Field(
        foreign_key="collection.id",
        ondelete="CASCADE",
        index=True,
    )

    user_id: int = Field(
        foreign_key="user.id",
        ondelete="CASCADE",
        index=True,
    )

    message: Optional[str] = Field(
        default=None,
        max_length=120,
    )

    status: str = Field(
        default="pending",
        max_length=20,
        index=True,
    )

    requested_at: datetime = Field(
        default_factory=utc_now,
    )

    collection: Optional["Collection"] = Relationship(
        back_populates="requests",
    )

    user: Optional["User"] = Relationship(
        back_populates="collection_requests",
    )


# ==========================================
# NOTIFICATIONS
# ==========================================

class Notification(SQLModel, table=True):
    id: Optional[int] = Field(
        default=None,
        primary_key=True,
    )

    user_id: int = Field(
        foreign_key="user.id",
        ondelete="CASCADE",
        index=True,
    )

    title: str

    body: str

    deep_link: Optional[str] = None

    is_read: bool = Field(
        default=False,
    )

    created_at: datetime = Field(
        default_factory=utc_now,
    )

    user: Optional["User"] = Relationship(
        back_populates="notifications",
    )


# ==========================================
# GAMIFICATION
# ==========================================

class TrophyDefinition(SQLModel, table=True):
    id: str = Field(
        primary_key=True,
    )

    title: str

    description: str

    icon: str

    condition_type: str

    condition_value: int

    is_coop: bool = Field(
        default=False,
    )


class UserTrophy(SQLModel, table=True):
    id: Optional[int] = Field(
        default=None,
        primary_key=True,
    )

    user_id: int = Field(
        foreign_key="user.id",
        ondelete="CASCADE",
        index=True,
    )

    trophy_id: str = Field(
        foreign_key="trophydefinition.id",
        ondelete="CASCADE",
    )

    earned_at: datetime = Field(
        default_factory=utc_now,
    )

    user: Optional["User"] = Relationship(
        back_populates="trophies",
    )


class Feedback(SQLModel, table=True):
    id: Optional[int] = Field(
        default=None,
        primary_key=True,
    )

    user_id: int = Field(
        foreign_key="user.id",
        ondelete="CASCADE",
        index=True,
    )

    message: str = Field(
        max_length=1000,
    )

    status: str = Field(
        default="unread",
        max_length=20,
    )

    created_at: datetime = Field(
        default_factory=utc_now,
    )


# ==========================================
# STUDY GROUPS / CO-OP
# ==========================================

class StudyGroup(SQLModel, table=True):
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
    )

    name: str = Field(
        max_length=50,
    )

    invite_code: str = Field(
        index=True,
        unique=True,
        max_length=6,
    )

    created_at: datetime = Field(
        default_factory=utc_now,
    )

    members: List["User"] = Relationship(
        back_populates="study_groups",
        link_model=GroupMember,
    )

    active_quests: List["CoopQuest"] = Relationship(
        back_populates="group",
        cascade_delete=True,
    )


class CoopQuest(SQLModel, table=True):
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
    )

    group_id: uuid.UUID = Field(
        foreign_key="studygroup.id",
        ondelete="CASCADE",
        index=True,
    )

    title: str

    target_xp: int = Field(
        ge=1,
    )

    current_xp: int = Field(
        default=0,
        ge=0,
    )

    is_completed: bool = Field(
        default=False,
    )

    group: Optional["StudyGroup"] = Relationship(
        back_populates="active_quests",
    )


class Relic(SQLModel, table=True):
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
    )

    name: str

    description: str

    users: List["User"] = Relationship(
        back_populates="relics",
        link_model=UserRelic,
    )