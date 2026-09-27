from datetime import date, datetime
from enum import Enum
from typing import List, Literal, Optional
from uuid import UUID

from pydantic import (
    BaseModel,
    EmailStr,
    Field,
    ConfigDict,
    field_validator,
)


# ==========================================
# NORMALIZERS (shared with main.py)
# ==========================================
#
# The AI sometimes returns "Review", "Flashcards", "Feynman",
# "Medium", "Weak Area". The Literal types below only accept
# lowercase singular forms. These normalizers map every variant
# to the canonical form so Pydantic validation never explodes
# on AI output.

_MODE_ALIASES = {
    "review": "review",
    "reviews": "review",
    "revision": "review",
    "flashcard": "flashcard",
    "flashcards": "flashcard",
    "flash card": "flashcard",
    "flash cards": "flashcard",
    "feynman": "feynman",
    "feynman technique": "feynman",
}

_PRIORITY_ALIASES = {
    "normal": "normal",
    "medium": "normal",
    "regular": "normal",
    "high": "high",
    "urgent": "high",
    "important": "high",
    "weak_area": "weak_area",
    "weak area": "weak_area",
    "weak": "weak_area",
}


def normalize_mode(value) -> str:
    if not isinstance(value, str):
        return "review"
    return _MODE_ALIASES.get(value.strip().lower(), "review")


def normalize_priority(value) -> str:
    if not isinstance(value, str):
        return "normal"
    return _PRIORITY_ALIASES.get(
        value.strip().lower().replace("_", " "),
        "normal",
    )


# ==========================================
# AUTH
# ==========================================

class UserRegister(BaseModel):
    name: str = Field(..., min_length=2, max_length=60)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=72)


class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(..., max_length=72)


class UserBasicInfo(BaseModel):
    id: int
    email: EmailStr
    name: str
    is_first_session: bool


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserBasicInfo


class TokenRefreshRequest(BaseModel):
    refresh_token: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    password: str = Field(..., min_length=8, max_length=72)


class UserProfileUpdate(BaseModel):
    study_goal: str


class FCMTokenUpdate(BaseModel):
    fcm_token: str


class PetAdoptionRequest(BaseModel):
    pet_type: Literal["nova", "pip", "luna", "zap"]
    pet_name: str = Field(
        ...,
        min_length=1,
        max_length=20,
        pattern=r"^[a-zA-Z0-9 \-']+$",
    )


class FirstSessionUpdate(BaseModel):
    is_first_session: bool


# ==========================================
# ONBOARDING / STUDY PROFILE
# ==========================================

class OnboardingQuizSubmit(BaseModel):
    """Used during first-run onboarding (guarded by is_first_session)."""
    goal_type: str
    study_goal: Optional[str] = None
    target_date: Optional[date] = None


class StudyProfileUpdate(BaseModel):
    """
    Always-available endpoint for updating the user's default persona.
    Unlike OnboardingQuizSubmit, this is not gated by is_first_session —
    it's how a user later changes their primary goal, or how the
    client keeps User.goal_type in sync with the persona picker.
    """
    goal_type: Optional[str] = None
    study_goal: Optional[str] = None
    target_date: Optional[date] = None


# ==========================================
# DASHBOARD
# ==========================================

class UserDashboardInfo(BaseModel):
    first_name: str
    is_first_session: bool


class PetDashboardInfo(BaseModel):
    name: str
    type: str
    level: int
    xp: int
    xp_to_next: int
    mood: str
    xp_history: List[int]

    current_emoji: str
    next_emoji: str

    progress_percent: float

    is_maxed: bool


class StreakInfo(BaseModel):
    days: int
    active_today: bool


class Quest(BaseModel):
    id: str
    title: str
    type: str
    progress: int
    target: int
    members_count: Optional[int] = None


class TodayPlanSession(BaseModel):
    """
    A single chip on the dashboard's "Today's Plan" row.

    plan_id / emoji / color_hex let the Flutter side group chips
    by plan and colour them consistently with the plan switcher.
    """
    id: str
    plan_id: Optional[int] = None
    subject: str
    duration_mins: int
    mode: str
    emoji: Optional[str] = None
    color_hex: Optional[str] = None


class DashboardResponse(BaseModel):
    user: UserDashboardInfo
    pet: PetDashboardInfo
    quests: List[Quest]
    today_plan: List[TodayPlanSession]
    streak: StreakInfo
    greeting: str

    goal_type: Optional[str] = None
    study_goal: Optional[str] = None
    target_date: Optional[date] = None


# ==========================================
# STUDY PLAN
# ==========================================

class PlanGoal(BaseModel):
    subject: str
    deadline: Optional[date] = None


class PlanStats(BaseModel):
    days_remaining: int
    daily_target_mins: int
    topics_count: int


class WeekDay(BaseModel):
    date: str
    day_label: str
    has_session: bool
    session_type: Literal["study", "review", "rest"]


class SessionDetail(BaseModel):
    id: str
    date: str
    time: Optional[str] = None
    subject: str
    duration_mins: int

    mode: Literal["flashcard", "feynman", "review"]
    priority: Literal["normal", "high", "weak_area"]

    completed: bool

    @field_validator("mode", mode="before")
    @classmethod
    def _norm_mode(cls, v):
        return normalize_mode(v)

    @field_validator("priority", mode="before")
    @classmethod
    def _norm_priority(cls, v):
        return normalize_priority(v)


class Nudge(BaseModel):
    message: str
    action: Literal["reschedule", "reduce", "add"]
    session_id: Optional[str] = None


class PlanResponse(BaseModel):
    # NEW: so the client can address the plan (approve / adjust / archive)
    plan_id: int

    # NEW: multi-goal metadata
    title: str
    goal_type: str
    emoji: Optional[str] = None
    color_hex: Optional[str] = None
    is_approved: bool = False

    goal: PlanGoal
    stats: PlanStats
    week: List[WeekDay]
    sessions: List[SessionDetail]
    nudge: Optional[Nudge] = None


class PlanGenerateRequest(BaseModel):
    goal: str
    deadline: Optional[date] = None

    # NEW: caller can override the persona for this plan.
    # If omitted, main.py falls back to current_user.goal_type.
    goal_type: Optional[str] = None

    # NEW: optional display metadata for the chip UI
    title: Optional[str] = None
    emoji: Optional[str] = Field(default=None, max_length=10)
    color_hex: Optional[str] = Field(default=None, max_length=9)


class PlanUpdateRequest(BaseModel):
    """Rename / recolour / re-deadline an existing plan."""
    title: Optional[str] = Field(default=None, max_length=80)
    deadline: Optional[date] = None
    emoji: Optional[str] = Field(default=None, max_length=10)
    color_hex: Optional[str] = Field(default=None, max_length=9)
    goal_type: Optional[str] = None


class PlanArchiveRequest(BaseModel):
    archived: bool = True


class SessionUpdateRequest(BaseModel):
    scheduled_time: Optional[str] = None
    duration_mins: Optional[int] = Field(default=None, ge=1)
    skipped: Optional[bool] = None


class ManualSessionCreate(BaseModel):
    date: str
    time: str
    subject: str = Field(..., min_length=1, max_length=100)
    duration_mins: int = Field(..., ge=1)
    mode: str
    priority: str

    @field_validator("mode", mode="before")
    @classmethod
    def _norm_mode(cls, v):
        return normalize_mode(v)

    @field_validator("priority", mode="before")
    @classmethod
    def _norm_priority(cls, v):
        return normalize_priority(v)


class PlanApproveRequest(BaseModel):
    approved: bool


# ==========================================
# PLAN LIST (multi-goal dashboard / switcher)
# ==========================================

class PlanListItem(BaseModel):
    plan_id: int
    title: str
    subject: str
    goal_type: str
    deadline: Optional[date] = None
    emoji: Optional[str] = None
    color_hex: Optional[str] = None
    is_approved: bool
    is_archived: bool
    days_remaining: Optional[int] = None
    session_count: int
    completed_count: int


class PlanListResponse(BaseModel):
    plans: List[PlanListItem]
    total_count: int


# ==========================================
# AI SOLVE
# ==========================================

class SolveRequest(BaseModel):
    question_text: Optional[str] = None
    question_image_base64: Optional[str] = None
    subject: Optional[str] = None


class HighlightTerm(BaseModel):
    term: str
    color: Literal["mint", "peach"]


class SolutionStep(BaseModel):
    step_number: int
    text: str
    highlight_terms: List[HighlightTerm] = Field(default_factory=list)


class CanvasLink(BaseModel):
    node_id: str
    node_label: str
    canvas_id: str


class SolveResponse(BaseModel):
    solution_id: str
    steps: List[SolutionStep]
    canvas_links: List[CanvasLink]
    confidence_score: float


class SolveFeedbackRequest(BaseModel):
    helpful: bool
    flag_reason: Optional[str] = None


# ==========================================
# CANVAS
# ==========================================

class NodeCreate(BaseModel):
    label: str = Field(..., max_length=40)
    x: float
    y: float
    size: Literal["small", "medium", "large"] = "medium"
    is_hero: bool = False
    is_weak: bool = False


class NodeUpdate(BaseModel):
    label: Optional[str] = Field(default=None, max_length=40)
    x: Optional[float] = None
    y: Optional[float] = None
    size: Optional[Literal["small", "medium", "large"]] = None
    is_hero: Optional[bool] = None
    is_weak: Optional[bool] = None


class ConnectionCreate(BaseModel):
    from_node_id: UUID
    to_node_id: UUID
    label: Optional[str] = None


class CanvasCreate(BaseModel):
    name: str
    subject: str
    source_type: Literal["notes", "upload", "manual"] = "manual"
    source_id: Optional[int] = None


class NodeResponse(BaseModel):
    id: UUID
    label: str
    x: float
    y: float
    size: Literal["small", "medium", "large"]
    is_hero: bool
    is_weak: bool
    definition: Optional[str] = None
    card_id: Optional[int] = None


class ConnectionResponse(BaseModel):
    id: UUID
    from_node_id: UUID
    to_node_id: UUID
    label: Optional[str] = None


class CanvasResponse(BaseModel):
    id: UUID
    name: str
    subject: str
    node_count: int
    weak_node_count: int
    thumbnail_url: Optional[str] = None
    source_type: str
    source_id: Optional[int] = None
    last_studied_at: Optional[datetime] = None
    created_at: datetime
    is_public: bool
    nodes: List[NodeResponse] = Field(default_factory=list)
    connections: List[ConnectionResponse] = Field(default_factory=list)


class CanvasStatusResponse(BaseModel):
    status: Literal["ready", "processing", "failed"]
    node_count: int
    nodes: List[NodeResponse] = Field(default_factory=list)


# ==========================================
# COLLECTION ENUMS
# ==========================================

class VisibilityEnum(str, Enum):
    private = "private"
    shared = "shared"
    public = "public"


class ItemTypeEnum(str, Enum):
    note = "note"
    set = "set"
    canvas = "canvas"


# ==========================================
# COLLECTION ITEM SCHEMAS
# ==========================================

class CollectionItem(BaseModel):
    item_id: str
    item_type: ItemTypeEnum
    position: int = Field(default=0, ge=0)


class AccessUser(BaseModel):
    user_id: int
    email: EmailStr
    granted_at: datetime
    granted_by: Optional[int] = None


class PendingRequest(BaseModel):
    request_id: int
    user_id: int
    username: str
    email: EmailStr
    message: Optional[str] = None
    requested_at: datetime


# ==========================================
# COLLECTION REQUESTS
# ==========================================

class CollectionCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=60)
    subject: str = Field(..., min_length=1, max_length=100)
    visibility: VisibilityEnum = VisibilityEnum.private
    description: Optional[str] = Field(default=None, max_length=300)
    cover_emoji: Optional[str] = Field(default=None, max_length=10)
    item_ids: List[str] = Field(default_factory=list)
    item_types: List[ItemTypeEnum] = Field(default_factory=list)


class CollectionUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=60)
    description: Optional[str] = Field(default=None, max_length=300)
    subject: Optional[str] = Field(default=None, min_length=1, max_length=100)
    visibility: Optional[VisibilityEnum] = None
    cover_emoji: Optional[str] = Field(default=None, max_length=10)


class ItemReorder(BaseModel):
    item_id: str
    item_type: ItemTypeEnum
    position: int = Field(..., ge=0)


class ItemReorderRequest(BaseModel):
    positions: List[ItemReorder] = Field(default_factory=list)


class ItemAddRequest(BaseModel):
    item_id: str = Field(..., min_length=1)
    item_type: ItemTypeEnum


class InviteRequest(BaseModel):
    emails: List[EmailStr] = Field(..., min_length=1, max_length=50)


class AccessRequestCreate(BaseModel):
    message: Optional[str] = Field(default=None, max_length=120)


class AccessRequestSubmit(BaseModel):
    message: Optional[str] = Field(default=None, max_length=120)


class RequestStatusUpdate(BaseModel):
    status: Literal["approved", "denied"]


class SaveItemRequest(BaseModel):
    item_id: str = Field(..., min_length=1)
    item_type: ItemTypeEnum


class ReportRequest(BaseModel):
    reason: str = Field(..., min_length=1, max_length=500)


# ==========================================
# COLLECTION RESPONSES
# ==========================================

class ResolvedCollectionItem(BaseModel):
    id: str
    type: ItemTypeEnum
    title: str
    subject: Optional[str] = None
    card_count: Optional[int] = None
    node_count: Optional[int] = None


class CollectionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    collection_id: int
    owner_id: int
    title: str
    description: Optional[str] = None
    subject: str
    cover_emoji: Optional[str] = None
    visibility: VisibilityEnum
    item_count: int
    share_token: str
    save_count: int
    created_at: datetime
    updated_at: datetime


class CollectionDetailResponse(CollectionResponse):
    items: List[ResolvedCollectionItem] = Field(default_factory=list)
    access_list: List[AccessUser] = Field(default_factory=list)
    pending_requests: List[PendingRequest] = Field(default_factory=list)


class CollectionListResponse(BaseModel):
    collections: List[CollectionResponse]
    total_count: int
    has_more: bool
    page: int
    limit: int


# ==========================================
# COMMUNITY
# ==========================================

class GroupCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=50)


class GroupJoin(BaseModel):
    invite_code: str = Field(
        ...,
        min_length=6,
        max_length=6,
        pattern=r"^[A-Za-z0-9]{6}$",
    )


class GroupResponse(BaseModel):
    id: UUID
    name: str
    invite_code: str
    created_at: datetime


class CoopQuestResponse(BaseModel):
    id: str
    title: str
    progress: int
    target: int
    type: str
    membersCount: int


# ==========================================
# GOOGLE AUTH
# ==========================================

class GoogleLoginRequest(BaseModel):
    id_token: str