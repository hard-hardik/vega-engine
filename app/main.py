"""
Vera Engine - FastAPI Server with Dual Routing
Production-grade implementation of magicpin's merchant AI assistant
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .engine import (
    ConversationEngine,
    GroundingValidator,
    LLMClient,
    MessageComposer,
    ToneAdapter,
)
from .schemas import (
    ComposedAction,
    ContextPushRequest,
    ContextPushResponse,
    HealthResponse,
    MetadataResponse,
    ReplyRequest,
    ReplyResponse,
    TickRequest,
    TickResponse,
)
from .state import context_store, conversation_store, suppression_store

load_dotenv()

llm_client = LLMClient()
composer = MessageComposer(llm_client)
conversation_engine = ConversationEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    yield

    if llm_client:
        await llm_client.close()


app = FastAPI(
    title="Vera Engine",
    description="magicpin's Merchant AI Assistant - 4-Context Composition Engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# HEALTH & METADATA ENDPOINTS
# ==============================================================================

@app.get("/healthz", response_model=HealthResponse, tags=["System"])
@app.get("/v1/healthz", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Liveness probe endpoint"""
    return HealthResponse(
        status="ok",
        uptime_seconds=context_store.uptime_seconds,
        contexts_loaded=context_store.get_counts(),
    )


@app.get("/metadata", response_model=MetadataResponse, tags=["System"])
@app.get("/v1/metadata", response_model=MetadataResponse, tags=["System"])
async def get_metadata() -> MetadataResponse:
    """Bot metadata and configuration"""
    return MetadataResponse(
        team_name="Team Vera Elite",
        team_members=["Candidate"],
        model="gpt-4o",
        approach="Deterministic 4-Context Composition Engine with Multi-Turn State Machine",
        contact_email="candidate@example.com",
        version="1.0.0",
        submitted_at="2026-04-26T08:00:00Z",
    )


# ==============================================================================
# CONTEXT PUSH ENDPOINT
# ==============================================================================

@app.post("/context", response_model=ContextPushResponse, tags=["Context"])
@app.post("/v1/context", response_model=ContextPushResponse, tags=["Context"])
async def push_context(request: ContextPushRequest) -> ContextPushResponse:
    """Ingest category, merchant, customer, or trigger context"""
    scope = request.scope
    context_id = request.context_id
    version = request.version
    payload = request.payload

    store_methods = {
        "category": context_store.store_category,
        "merchant": context_store.store_merchant,
        "customer": context_store.store_customer,
        "trigger": context_store.store_trigger,
    }

    store_fn = store_methods.get(scope)
    if not store_fn:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"accepted": False, "reason": f"invalid_scope: {scope}"}
        )

    try:
        success, current_version = store_fn(context_id, version, payload)
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"accepted": False, "reason": f"validation_error: {str(e)}"}
        )

    if not success:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "accepted": False,
                "reason": "stale_version",
                "current_version": current_version,
            }
        )

    return ContextPushResponse(
        accepted=True,
        ack_id=f"ack_{context_id}_v{version}",
        stored_at=datetime.utcnow().isoformat() + "Z",
    )


# ==============================================================================
# TICK ENDPOINT
# ==============================================================================

@app.post("/tick", response_model=TickResponse, tags=["Composition"])
@app.post("/v1/tick", response_model=TickResponse, tags=["Composition"])
async def tick(request: TickRequest) -> TickResponse:
    """Evaluate triggers and compose outbound messages"""
    actions: list[ComposedAction] = []
    compose_tasks = []

    for trigger_id in request.available_triggers:
        trigger = context_store.get_trigger(trigger_id)
        if not trigger:
            continue

        if suppression_store.is_suppressed(trigger.suppression_key):
            continue

        merchant = context_store.get_merchant(trigger.merchant_id)
        if not merchant:
            continue

        category = context_store.get_category(merchant.category_slug)
        if not category:
            continue

        customer = None
        if trigger.customer_id:
            customer = context_store.get_customer(trigger.customer_id)

        compose_tasks.append(
            _compose_with_timeout(category, merchant, trigger, customer)
        )

    if compose_tasks:
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*compose_tasks, return_exceptions=True),
                timeout=25.0
            )
            for result in results:
                if isinstance(result, ComposedAction):
                    suppression_store.suppress(
                        result.suppression_key,
                        expires_at=None
                    )
                    actions.append(result)
        except asyncio.TimeoutError:
            pass

    return TickResponse(actions=actions)


async def _compose_with_timeout(
    category,
    merchant,
    trigger,
    customer,
) -> Optional[ComposedAction]:
    """Compose message with individual timeout"""
    try:
        return await asyncio.wait_for(
            composer.compose(category, merchant, trigger, customer),
            timeout=10.0
        )
    except Exception:
        return None


# ==============================================================================
# REPLY ENDPOINT
# ==============================================================================

@app.post("/reply", response_model=ReplyResponse, tags=["Conversation"])
@app.post("/v1/reply", response_model=ReplyResponse, tags=["Conversation"])
async def handle_reply(request: ReplyRequest) -> ReplyResponse:
    """Handle inbound merchant/customer reply"""
    state = conversation_store.get_or_create(
        conversation_id=request.conversation_id,
        merchant_id=request.merchant_id,
        customer_id=request.customer_id,
    )

    conversation_store.add_turn(
        conversation_id=request.conversation_id,
        from_role=request.from_role,
        message=request.message,
        turn_number=request.turn_number,
    )

    merchant = context_store.get_merchant(request.merchant_id)
    if not merchant:
        return ReplyResponse(
            action="wait",
            wait_seconds=3600,
            rationale="Merchant context not found; backing off.",
        )

    category = context_store.get_category(merchant.category_slug)
    if not category:
        return ReplyResponse(
            action="wait",
            wait_seconds=3600,
            rationale="Category context not found; backing off.",
        )

    customer = None
    if request.customer_id:
        customer = context_store.get_customer(request.customer_id)

    response = conversation_engine.handle_reply(
        message=request.message,
        state=state,
        category=category,
        merchant=merchant,
        customer=customer,
    )

    conversation_store.update(state)

    if response.action == "send" and response.body:
        if conversation_store.was_message_sent(request.conversation_id, response.body):
            return ReplyResponse(
                action="wait",
                wait_seconds=7200,
                rationale="Anti-repetition: this exact message was already sent. Backing off.",
            )
        conversation_store.record_sent_message(request.conversation_id, response.body)
        state.last_vera_message = response.body

    return response


# ==============================================================================
# ERROR HANDLERS
# ==============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
