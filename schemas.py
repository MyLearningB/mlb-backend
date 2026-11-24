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
    # v2 plan modes
    "learn": "learn",
    "study": "learn",
    "read": "learn",
    "practice": "practice",
    "practise": "practice",
    "drill": "practice",
    "mock": "mock",
    "mock test": "mock",
    "practice test": "mock",
    "test": "mock",
    "project": "project",
    "build": "project",
}

PLAN_MODES = (
    "learn", "practice", "review", "flashcard", "feynman", "mock", "project",
)

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
    Unlike OnboardingQuizSubmit, this is not gated by is_first_session.
    Also used by onboarding step 3 ("What's your goal?").
    """
    goal_type: Optional[str] = None
    study_goal: Optional[str] = Field(default=None, max_length=200)
    target_date: Optional[date] = None
    clear_target_date: bool = False
    daily_goal_mins: Optional[int] = Field(default=None, ge=10, le=600)


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

    xp_into_level: int = 0
    level_xp_span: int = 0

    mood: str
    xp_history: List[int]

    current_emoji: str
    next_emoji: str

    progress_percent: float

    # True once the FINAL evolution is reached. Levels keep counting.
    is_maxed: bool

    # NEW — personality + evolution context (see gamification.py)
    mood_message: str = ""
    title: Optional[str] = None
    trait: Optional[str] = None
    trait_desc: Optional[str] = None
    stage: int = 0
    stage_name: Optional[str] = None
    next_stage_level: Optional[int] = None
    evolution_line: List[str] = Field(default_factory=list)


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
    id: str
    plan_id: Optional[int] = None
    # NEW — distinguishes which plan this session belongs to, so a
    # user running "School" + "GRE" plans can tell the chips apart.
    plan_title: Optional[str] = None
    subject: str
    # NEW — specific sub-topic when the plan has one.
    topic: Optional[str] = None
    duration_mins: int
    mode: str
    emoji: Optional[str] = None
    color_hex: Optional[str] = None
    # NEW — lets the chip render a strikethrough + tick.
    completed: bool = False
    # v2 — what to do and when
    time: Optional[str] = None
    task: Optional[str] = None
    skipped: bool = False


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
    daily_goal_mins: Optional[int] = None

    # NEW — server-decided onboarding routing: "quiz" | "pet" | "done"
    onboarding_step: str = "done"
    has_pet: bool = True


# ==========================================
# STUDY PLAN — shared primitives
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
    session_type: Literal["study", "review", "rest", "weak_area"]


class SessionDetail(BaseModel):
    id: str
    date: str
    time: Optional[str] = None
    subject: str
    topic: Optional[str] = None
    # v2 — the concrete thing to do this session
    task: Optional[str] = None
    subject_id: Optional[int] = None
    duration_mins: int

    mode: Literal[
        "learn", "practice", "review", "flashcard", "feynman", "mock", "project"
    ]
    priority: Literal["normal", "high", "weak_area"]

    completed: bool
    skipped: bool = False
    is_manual: bool = False

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


# ==========================================
# PHASE PLAN
# ==========================================

class PlanPhase(BaseModel):
    """A contiguous stretch of the arc (see generate_gemini_phase_plan)."""
    name: str
    start_week: int = Field(..., ge=1)
    end_week: int = Field(..., ge=1)
    focus: List[str] = Field(default_factory=list)
    milestone: str = ""


class PhasePlan(BaseModel):
    phases: List[PlanPhase]
    total_weeks: int = Field(..., ge=1)


# ==========================================
# PLAN RESPONSE
# ==========================================

class PlanSubjectResponse(BaseModel):
    id: int
    name: str
    exam_date: Optional[date] = None
    weight: int
    current_level: str
    topics: List[str] = Field(default_factory=list)
    weak_topics: List[str] = Field(default_factory=list)

    # Denormalised progress so the client doesn't need N+1 calls
    session_count: int = 0
    completed_count: int = 0


class PlanResponse(BaseModel):
    plan_id: int

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

    # Multi-subject metadata
    subjects: List[PlanSubjectResponse] = Field(default_factory=list)
    plan_type: str = "single_goal"
    notes: Optional[str] = None

    # Arc metadata (computed in routers/plan.py)
    total_weeks: Optional[int] = None
    current_week: Optional[int] = None
    phase_name: Optional[str] = None
    phase_milestone: Optional[str] = None

    updated_at: Optional[datetime] = None

    # ---- v2: flexible goal plans ----
    goal_kind: Optional[str] = None          # exam | course | skill | habit
    level: Optional[str] = None
    summary: Optional[str] = None
    schedule: Optional["PlanSchedule"] = None
    roadmap: List["RoadmapPhase"] = Field(default_factory=list)
    current_unit: Optional[str] = None
    end_date: Optional[date] = None
    roadmap_source: Optional[str] = None     # ai | cache | fallback
    ai_plans_left: int = 0                   # AI builds/re-plans left today


class PlanSchedule(BaseModel):
    minutes_per_day: int
    days_of_week: List[int]
    preferred_times: List[str]


class RoadmapPhase(BaseModel):
    name: str
    milestone: str = ""
    weeks: int = 1
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    status: Literal["done", "current", "upcoming"] = "upcoming"
    units: List[str] = Field(default_factory=list)


PlanResponse.model_rebuild()


# ==========================================
# PLAN REQUESTS
# ==========================================

class PlanAvailability(BaseModel):
    minutes_per_day: Optional[int] = Field(default=None, ge=15, le=600)
    study_days_per_week: Optional[int] = Field(default=None, ge=1, le=7)

    # Richer window the AI must respect
    earliest: Optional[str] = Field(default=None, pattern=r"^\d{2}:\d{2}$")
    latest: Optional[str] = Field(default=None, pattern=r"^\d{2}:\d{2}$")
    session_length_pref: Optional[int] = Field(default=None, ge=10, le=180)


class PlanSubjectInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    exam_date: Optional[date] = None
    weight: int = Field(default=3, ge=1, le=5)
    current_level: Literal["beginner", "intermediate", "advanced"] = "intermediate"
    topics: List[str] = Field(default_factory=list, max_length=50)
    weak_topics: List[str] = Field(default_factory=list, max_length=50)


class PlanGenerateRequest(BaseModel):
    """
    v2 shape (preferred):
      { "goal": "GRE in 10 months", "deadline": "2027-09-01",
        "level": "intermediate", "minutes_per_day": 60,
        "days_of_week": [1,2,3,4,5,6], "preferred_times": ["08:15", "19:00"],
        "notes": "I struggle with geometry" }

    Older clients still work:
      * `availability.minutes_per_day / study_days_per_week / earliest`
      * `subjects[]` — names are joined into one goal, topics go to notes.
    """
    goal: Optional[str] = Field(default=None, min_length=1, max_length=200)
    deadline: Optional[date] = None

    level: Optional[Literal["beginner", "intermediate", "advanced"]] = None
    minutes_per_day: Optional[int] = Field(default=None, ge=10, le=600)
    days_of_week: Optional[List[int]] = Field(default=None, max_length=7)
    preferred_times: Optional[List[str]] = Field(default=None, max_length=3)

    goal_type: Optional[str] = None

    title: Optional[str] = Field(default=None, max_length=80)
    emoji: Optional[str] = Field(default=None, max_length=10)
    color_hex: Optional[str] = Field(default=None, max_length=9)

    # Free-form hint the AI reads verbatim
    notes: Optional[str] = Field(default=None, max_length=1000)

    # ---- legacy ----
    availability: Optional[PlanAvailability] = None
    subjects: Optional[List[PlanSubjectInput]] = None
    plan_type: Optional[str] = None


class PlanScheduleUpdate(BaseModel):
    """Change when/how much you study. Free — no AI call."""
    minutes_per_day: Optional[int] = Field(default=None, ge=10, le=600)
    days_of_week: Optional[List[int]] = Field(default=None, min_length=1, max_length=7)
    preferred_times: Optional[List[str]] = Field(default=None, min_length=1, max_length=3)
    deadline: Optional[date] = None
    clear_deadline: bool = False


class PlanReplanRequest(BaseModel):
    """Ask the AI to rebuild the rest of the plan. Uses one AI credit."""
    feedback: Optional[str] = Field(default=None, max_length=500)
    level: Optional[Literal["beginner", "intermediate", "advanced"]] = None


class UpcomingSession(BaseModel):
    id: str
    plan_id: int
    plan_title: str
    emoji: Optional[str] = None
    date: str
    time: str
    duration_mins: int
    topic: Optional[str] = None
    task: Optional[str] = None
    mode: str


class UpcomingResponse(BaseModel):
    sessions: List[UpcomingSession]


class PlanUpdateRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=80)
    deadline: Optional[date] = None
    emoji: Optional[str] = Field(default=None, max_length=10)
    color_hex: Optional[str] = Field(default=None, max_length=9)
    goal_type: Optional[str] = None


class PlanArchiveRequest(BaseModel):
    archived: bool = True


class SessionUpdateRequest(BaseModel):
    scheduled_time: Optional[str] = None
    duration_mins: Optional[int] = Field(default=None, ge=1, le=300)
    skipped: Optional[bool] = None


class ManualSessionCreate(BaseModel):
    date: str
    time: str
    subject: str = Field(..., min_length=1, max_length=100)
    duration_mins: int = Field(..., ge=1, le=300)
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
# PLAN LIST (multi-goal switcher)
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
    label: str = Field(..., min_length=1, max_length=40)
    x: float
    y: float
    size: Literal["small", "medium", "large"] = "medium"
    is_hero: bool = False
    is_weak: bool = False
    definition: Optional[str] = Field(default=None, max_length=500)
    # When set, the server also creates parent -> new node connection.
    parent_id: Optional[UUID] = None


class NodeUpdate(BaseModel):
    label: Optional[str] = Field(default=None, max_length=40)
    x: Optional[float] = None
    y: Optional[float] = None
    size: Optional[Literal["small", "medium", "large"]] = None
    is_hero: Optional[bool] = None
    is_weak: Optional[bool] = None
    # Send "" to clear the definition.
    definition: Optional[str] = Field(default=None, max_length=500)


class ConnectionCreate(BaseModel):
    from_node_id: UUID
    to_node_id: UUID
    label: Optional[str] = Field(default=None, max_length=60)


class CanvasCreate(BaseModel):
    name: str = Field(default="Untitled Canvas", max_length=80)
    subject: str = Field(default="General", max_length=100)
    source_type: Literal["notes", "upload", "manual"] = "manual"
    source_id: Optional[int] = None
    # FIX: the from-notes route reads payload.note_id, but this field
    # did not exist, so every "From notes" request crashed with a 500.
    note_id: Optional[int] = None


class CanvasUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=80)
    subject: Optional[str] = Field(default=None, min_length=1, max_length=100)


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
    status: str = "ready"
    error_message: Optional[str] = None
    nodes: List[NodeResponse] = Field(default_factory=list)
    connections: List[ConnectionResponse] = Field(default_factory=list)


class CanvasStatusResponse(BaseModel):
    status: Literal["ready", "processing", "failed"]
    node_count: int
    error: Optional[str] = None
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