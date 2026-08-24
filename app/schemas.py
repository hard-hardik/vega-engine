"""
Pydantic v2 schemas for Vera Engine
Covers all 4 contexts, API requests & responses
"""

from datetime import datetime
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# ==============================================================================
# CATEGORY CONTEXT SCHEMAS
# ==============================================================================

class VoiceProfile(BaseModel):
    """Category-specific voice configuration"""
    tone: str
    tone_register: str = ""
    code_mix: str = "hindi_english_natural"
    vocab_allowed: list[str] = Field(default_factory=list)
    vocab_taboo: list[str] = Field(default_factory=list)
    salutation_examples: list[str] = Field(default_factory=list)
    tone_examples: list[str] = Field(default_factory=list)


class OfferTemplate(BaseModel):
    """Canonical offer template for a category"""
    id: str
    title: str
    value: str
    audience: str
    type: str


class PeerStats(BaseModel):
    """Peer benchmarks for the category"""
    scope: str = ""
    avg_rating: float = 0.0
    avg_review_count: int = 0
    avg_views_30d: int = 0
    avg_calls_30d: int = 0
    avg_directions_30d: int = 0
    avg_ctr: float = 0.0
    avg_photos: int = 0
    avg_post_freq_days: int = 0
    retention_6mo_pct: Optional[float] = None
    retention_3mo_pct: Optional[float] = None
    retention_30d_pct: Optional[float] = None
    monthly_churn_pct: Optional[float] = None
    trial_to_paid_pct: Optional[float] = None
    delivery_share_pct: Optional[float] = None
    repeat_customer_pct: Optional[float] = None


class DigestItem(BaseModel):
    """Weekly digest item (research, compliance, trend, etc.)"""
    id: str
    kind: str
    title: str
    source: str
    summary: str = ""
    actionable: str = ""
    trial_n: Optional[int] = None
    patient_segment: Optional[str] = None
    date: Optional[str] = None
    credits: Optional[int] = None
    deadline_iso: Optional[str] = None


class ContentItem(BaseModel):
    """Patient/customer content library item"""
    id: str
    title: str
    channel: str = "whatsapp"
    length_seconds: int = 60
    body: str = ""


class SeasonalBeat(BaseModel):
    """Seasonal pattern for the category"""
    month_range: str
    note: str


class TrendSignal(BaseModel):
    """Search trend signal"""
    query: str
    delta_yoy: float
    segment_age: str = "all"
    skew: str = "balanced"


class CategoryContext(BaseModel):
    """Complete category context"""
    slug: str
    display_name: str = ""
    voice: VoiceProfile
    offer_catalog: list[OfferTemplate] = Field(default_factory=list)
    peer_stats: PeerStats = Field(default_factory=PeerStats)
    digest: list[DigestItem] = Field(default_factory=list)
    patient_content_library: list[ContentItem] = Field(default_factory=list)
    seasonal_beats: list[SeasonalBeat] = Field(default_factory=list)
    trend_signals: list[TrendSignal] = Field(default_factory=list)
    regulatory_authorities: list[str] = Field(default_factory=list)
    professional_journals: list[str] = Field(default_factory=list)


# ==============================================================================
# MERCHANT CONTEXT SCHEMAS
# ==============================================================================

class MerchantIdentity(BaseModel):
    """Merchant identity information"""
    name: str
    city: str = ""
    locality: str = ""
    place_id: str = ""
    verified: bool = False
    languages: list[str] = Field(default_factory=lambda: ["en"])
    owner_first_name: str = ""
    established_year: int = 2020


class Subscription(BaseModel):
    """Merchant subscription status"""
    status: Literal["active", "expired", "trial"] = "active"
    plan: str = "Pro"
    days_remaining: int = 0
    days_since_expiry: Optional[int] = None
    renewed_at: Optional[str] = None


class PerformanceDelta(BaseModel):
    """7-day performance delta"""
    views_pct: float = 0.0
    calls_pct: float = 0.0
    ctr_pct: Optional[float] = None


class PerformanceSnapshot(BaseModel):
    """Merchant performance metrics"""
    window_days: int = 30
    views: int = 0
    calls: int = 0
    directions: int = 0
    ctr: float = 0.0
    leads: int = 0
    delta_7d: PerformanceDelta = Field(default_factory=PerformanceDelta)


class MerchantOffer(BaseModel):
    """Active or expired merchant offer"""
    id: str
    title: str
    status: Literal["active", "paused", "expired"] = "active"
    started: Optional[str] = None
    ended: Optional[str] = None


class ConversationTurn(BaseModel):
    """Single conversation turn"""
    ts: str
    from_: str = Field(alias="from")
    body: str
    engagement: str = ""

    class Config:
        populate_by_name = True


class CustomerAggregate(BaseModel):
    """Customer aggregate statistics"""
    total_unique_ytd: int = 0
    lapsed_180d_plus: Optional[int] = None
    lapsed_90d_plus: Optional[int] = None
    retention_6mo_pct: Optional[float] = None
    retention_3mo_pct: Optional[float] = None
    high_risk_adult_count: Optional[int] = None
    repeat_customer_pct: Optional[float] = None
    total_active_members: Optional[int] = None
    monthly_churn_pct: Optional[float] = None
    trial_to_paid_pct: Optional[float] = None
    chronic_rx_count: Optional[int] = None
    delivery_orders_30d: Optional[int] = None
    dine_in_orders_30d: Optional[int] = None
    delivery_share_pct: Optional[float] = None


class ReviewTheme(BaseModel):
    """Emerged review theme"""
    theme: str
    sentiment: Literal["pos", "neg", "neutral"] = "neutral"
    occurrences_30d: int = 0
    common_quote: str = ""


class MerchantContext(BaseModel):
    """Complete merchant context"""
    merchant_id: str
    category_slug: str
    identity: MerchantIdentity
    subscription: Subscription = Field(default_factory=Subscription)
    performance: PerformanceSnapshot = Field(default_factory=PerformanceSnapshot)
    offers: list[MerchantOffer] = Field(default_factory=list)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    customer_aggregate: CustomerAggregate = Field(default_factory=CustomerAggregate)
    signals: list[str] = Field(default_factory=list)
    review_themes: list[ReviewTheme] = Field(default_factory=list)


# ==============================================================================
# TRIGGER CONTEXT SCHEMAS
# ==============================================================================

class TriggerContext(BaseModel):
    """Trigger context - the event prompting this outreach"""
    id: str
    scope: Literal["merchant", "customer"]
    kind: str
    source: Literal["external", "internal"]
    merchant_id: str
    customer_id: Optional[str] = None
    payload: dict[str, Any] = Field(default_factory=dict)
    urgency: int = Field(ge=1, le=5, default=2)
    suppression_key: str = ""
    expires_at: str = ""


# ==============================================================================
# CUSTOMER CONTEXT SCHEMAS
# ==============================================================================

class CustomerIdentity(BaseModel):
    """Customer identity information"""
    name: str
    phone_redacted: Optional[str] = "<phone>"
    language_pref: str = "en"
    age_band: str = ""
    senior_citizen: bool = False


class CustomerRelationship(BaseModel):
    """Customer-merchant relationship"""
    first_visit: str = ""
    last_visit: str = ""
    visits_total: int = 0
    services_received: list[str] = Field(default_factory=list)
    lifetime_value: int = 0
    favourite_dish: Optional[str] = None
    chronic_conditions: list[str] = Field(default_factory=list)


class CustomerPreferences(BaseModel):
    """Customer preferences"""
    preferred_slots: str = ""
    channel: str = "whatsapp"
    reminder_opt_in: bool = True
    preferred_stylist: Optional[str] = None
    wedding_date: Optional[str] = None
    training_focus: Optional[str] = None
    health_focus: Optional[str] = None
    delivery_address: Optional[str] = None
    office_nearby: Optional[bool] = None
    family_size: Optional[int] = None
    household_size: Optional[int] = None


class CustomerConsent(BaseModel):
    """Customer consent information"""
    opted_in_at: Optional[str] = None
    scope: list[str] = Field(default_factory=list)


class CustomerContext(BaseModel):
    """Complete customer context"""
    customer_id: str
    merchant_id: str
    identity: CustomerIdentity
    relationship: CustomerRelationship = Field(default_factory=CustomerRelationship)
    state: Literal["new", "active", "lapsed_soft", "lapsed_hard", "churned"] = "active"
    preferences: CustomerPreferences = Field(default_factory=CustomerPreferences)
    consent: CustomerConsent = Field(default_factory=CustomerConsent)


# ==============================================================================
# API REQUEST/RESPONSE SCHEMAS
# ==============================================================================

class HealthResponse(BaseModel):
    """GET /v1/healthz response"""
    status: str = "ok"
    uptime_seconds: int = 0
    contexts_loaded: dict[str, int] = Field(default_factory=dict)


class MetadataResponse(BaseModel):
    """GET /v1/metadata response"""
    team_name: str = "Team Vera Elite"
    team_members: list[str] = Field(default_factory=lambda: ["Candidate"])
    model: str = "gpt-4o"
    approach: str = "Deterministic 4-Context Composition Engine with Multi-Turn State Machine"
    contact_email: str = "candidate@example.com"
    version: str = "1.0.0"
    submitted_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class ContextPushRequest(BaseModel):
    """POST /v1/context request"""
    scope: Literal["category", "merchant", "customer", "trigger"]
    context_id: str
    version: int = 1
    payload: dict[str, Any]
    delivered_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class ContextPushResponse(BaseModel):
    """POST /v1/context response"""
    accepted: bool
    ack_id: Optional[str] = None
    stored_at: Optional[str] = None
    reason: Optional[str] = None
    current_version: Optional[int] = None


class TickRequest(BaseModel):
    """POST /v1/tick request"""
    now: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    available_triggers: list[str] = Field(default_factory=list)


class ComposedAction(BaseModel):
    """Single action in tick response"""
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    send_as: Literal["vera", "merchant_on_behalf"] = "vera"
    trigger_id: str
    template_name: str = ""
    template_params: list[str] = Field(default_factory=list)
    body: str
    cta: Literal["binary_yes_no", "open_ended", "slot_selection", "none"] = "open_ended"
    suppression_key: str = ""
    rationale: str = ""


class TickResponse(BaseModel):
    """POST /v1/tick response"""
    actions: list[ComposedAction] = Field(default_factory=list)


class ReplyRequest(BaseModel):
    """POST /v1/reply request"""
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    from_role: Literal["merchant", "customer"] = "merchant"
    message: str
    received_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    turn_number: int = 1


class ReplyResponse(BaseModel):
    """POST /v1/reply response"""
    action: Literal["send", "wait", "end"]
    body: Optional[str] = None
    cta: Optional[str] = None
    wait_seconds: Optional[int] = None
    rationale: str = ""


# ==============================================================================
# INTERNAL STATE SCHEMAS
# ==============================================================================

class ConversationState(BaseModel):
    """Internal conversation state tracking"""
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    turns: list[dict[str, Any]] = Field(default_factory=list)
    auto_reply_count: int = 0
    last_vera_message: str = ""
    intent_committed: bool = False
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    last_activity_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class SuppressedTrigger(BaseModel):
    """Record of a suppressed trigger"""
    suppression_key: str
    suppressed_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    expires_at: Optional[str] = None
