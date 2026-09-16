from pydantic import BaseModel, EmailStr, Field
from typing import Literal, List, Optional
from datetime import date, datetime
from uuid import UUID
from enum import Enum
 
# ==========================================
# AUTH & USER SETTINGS SCHEMAS
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
        description="Letters, numbers, spaces, hyphens, and apostrophes only."
    )

class FirstSessionUpdate(BaseModel):
    is_first_session: bool

class OnboardingQuizSubmit(BaseModel):
    goal_type: str 
    study_goal: Optional[str] = None
    target_date: Optional[date] = None

# ==========================================
# DASHBOARD SCHEMAS (SCREEN 3)
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
    subject: str
    duration_mins: int
    mode: str 

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

    class Config:
        populate_by_name = True

# ==========================================
# STUDY PLAN SCHEMAS (SCREEN 9)
# ==========================================
class PlanGoal(BaseModel):
    subject: str
    deadline: Optional[date] = None # <-- Now Optional

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
    time: str
    subject: str
    duration_mins: int
    mode: Literal["flashcard", "feynman", "review"]
    priority: Literal["normal", "high", "weak_area"]
    completed: bool

class Nudge(BaseModel):
    message: str
    action: Literal["reschedule", "reduce", "add"]
    session_id: Optional[str] = None

class PlanResponse(BaseModel):
    goal: PlanGoal
    stats: PlanStats
    week: List[WeekDay]
    sessions: List[SessionDetail]
    nudge: Optional[Nudge] = None

class PlanGenerateRequest(BaseModel):
    goal: str
    deadline: Optional[date] = None # <-- Now Optional

class SessionUpdateRequest(BaseModel):
    scheduled_time: Optional[str] = None
    duration_mins: Optional[int] = None
    skipped: Optional[bool] = None

class PlanApproveRequest(BaseModel):
    approved: bool

# ==========================================
# AI SOLVE SCHEMAS (SCREEN 10)
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
    highlight_terms: List[HighlightTerm] = []

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
# CANVAS TAB SCHEMAS (SCREEN 5)
# ==========================================
class NodeCreate(BaseModel):
    label: str = Field(..., max_length=40)
    x: float
    y: float
    size: Literal["small", "medium", "large"] = "medium"
    is_hero: bool = False
    is_weak: bool = False

class NodeUpdate(BaseModel):
    label: Optional[str] = Field(None, max_length=40)
    x: Optional[float] = None
    y: Optional[float] = None
    size: Optional[Literal["small", "medium", "large"]] = None
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
    nodes: List[NodeResponse] = []
    connections: List[ConnectionResponse] = []

class CanvasStatusResponse(BaseModel):
    status: Literal["ready", "processing", "failed"]
    node_count: int
    nodes: List[NodeResponse] = []

# ==========================================
# COLLECTIONS SCHEMAS
# ==========================================
class VisibilityEnum(str, Enum):
    private = "private"
    shared = "shared"
    public = "public"

class ItemTypeEnum(str, Enum):
    note = "note"
    set = "set"
    canvas = "canvas"

class CollectionItem(BaseModel):
    item_id: str 
    item_type: ItemTypeEnum
    position: int

class AccessUser(BaseModel):
    user_id: int  
    email: str
    granted_at: datetime
    granted_by: int 

class PendingRequest(BaseModel):
    request_id: int 
    user_id: int    
    username: str
    email: str
    message: Optional[str] = None
    requested_at: datetime

class CollectionCreate(BaseModel):
    title: str = Field(..., max_length=60)
    subject: str
    visibility: VisibilityEnum
    description: Optional[str] = Field(None, max_length=300)
    cover_emoji: Optional[str] = None
    item_ids: List[str] = []
    item_types: List[str] = []

class CollectionUpdate(BaseModel):
    title: Optional[str] = Field(None, max_length=60)
    description: Optional[str] = Field(None, max_length=300)
    subject: Optional[str] = None
    visibility: Optional[VisibilityEnum] = None
    cover_emoji: Optional[str] = None

class ItemReorder(BaseModel):
    item_id: str 
    position: int

class ItemReorderRequest(BaseModel):
    positions: List[ItemReorder]

class AccessRequestCreate(BaseModel):
    message: Optional[str] = Field(None, max_length=120)

class CollectionResponse(BaseModel):
    collection_id: int 
    owner_id: int      
    title: str
    description: Optional[str]
    subject: str
    cover_emoji: Optional[str]
    visibility: VisibilityEnum
    item_count: int
    share_token: str
    save_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class CollectionDetailResponse(CollectionResponse):
    items: List[CollectionItem] = []
    access_list: List[AccessUser] = []
    pending_requests: List[PendingRequest] = []

class NodeUpdate(BaseModel):
    label: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    size: Optional[str] = None
    is_hero: Optional[bool] = None
    is_weak: Optional[bool] = None  
 
class GoogleLoginRequest(BaseModel):
    id_token: str