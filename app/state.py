"""
Thread-safe in-memory state management for Vera Engine
Stores contexts, conversations, and suppression records
"""

import threading
from datetime import datetime
from typing import Any, Optional

from .schemas import (
    CategoryContext,
    ConversationState,
    CustomerContext,
    MerchantContext,
    SuppressedTrigger,
    TriggerContext,
)


class ContextStore:
    """Thread-safe in-memory store for all context types"""

    def __init__(self):
        self._lock = threading.RLock()
        self._categories: dict[str, tuple[int, CategoryContext]] = {}
        self._merchants: dict[str, tuple[int, MerchantContext]] = {}
        self._customers: dict[str, tuple[int, CustomerContext]] = {}
        self._triggers: dict[str, tuple[int, TriggerContext]] = {}
        self._start_time = datetime.utcnow()

    @property
    def uptime_seconds(self) -> int:
        return int((datetime.utcnow() - self._start_time).total_seconds())

    def get_counts(self) -> dict[str, int]:
        with self._lock:
            return {
                "category": len(self._categories),
                "merchant": len(self._merchants),
                "customer": len(self._customers),
                "trigger": len(self._triggers),
            }

    def store_category(self, context_id: str, version: int, data: dict) -> tuple[bool, Optional[int]]:
        with self._lock:
            ctx = CategoryContext.model_validate(data)
            key = ctx.slug
            current = self._categories.get(key)
            if current and current[0] >= version:
                return False, current[0]
            self._categories[key] = (version, ctx)
            return True, None

    def store_merchant(self, context_id: str, version: int, data: dict) -> tuple[bool, Optional[int]]:
        with self._lock:
            current = self._merchants.get(context_id)
            if current and current[0] >= version:
                return False, current[0]
            ctx = MerchantContext.model_validate(data)
            self._merchants[context_id] = (version, ctx)
            return True, None

    def store_customer(self, context_id: str, version: int, data: dict) -> tuple[bool, Optional[int]]:
        with self._lock:
            current = self._customers.get(context_id)
            if current and current[0] >= version:
                return False, current[0]
            ctx = CustomerContext.model_validate(data)
            self._customers[context_id] = (version, ctx)
            return True, None

    def store_trigger(self, context_id: str, version: int, data: dict) -> tuple[bool, Optional[int]]:
        with self._lock:
            current = self._triggers.get(context_id)
            if current and current[0] >= version:
                return False, current[0]
            ctx = TriggerContext.model_validate(data)
            self._triggers[context_id] = (version, ctx)
            return True, None

    def get_category(self, slug: str) -> Optional[CategoryContext]:
        with self._lock:
            item = self._categories.get(slug)
            return item[1] if item else None

    def get_merchant(self, merchant_id: str) -> Optional[MerchantContext]:
        with self._lock:
            item = self._merchants.get(merchant_id)
            return item[1] if item else None

    def get_customer(self, customer_id: str) -> Optional[CustomerContext]:
        with self._lock:
            item = self._customers.get(customer_id)
            return item[1] if item else None

    def get_trigger(self, trigger_id: str) -> Optional[TriggerContext]:
        with self._lock:
            item = self._triggers.get(trigger_id)
            return item[1] if item else None

    def list_all_categories(self) -> list[CategoryContext]:
        with self._lock:
            return [item[1] for item in self._categories.values()]

    def list_all_merchants(self) -> list[MerchantContext]:
        with self._lock:
            return [item[1] for item in self._merchants.values()]


class ConversationStore:
    """Thread-safe store for active conversations"""

    def __init__(self):
        self._lock = threading.RLock()
        self._conversations: dict[str, ConversationState] = {}
        self._sent_messages: dict[str, set[str]] = {}

    def get_or_create(
        self,
        conversation_id: str,
        merchant_id: str,
        customer_id: Optional[str] = None,
    ) -> ConversationState:
        with self._lock:
            if conversation_id not in self._conversations:
                self._conversations[conversation_id] = ConversationState(
                    conversation_id=conversation_id,
                    merchant_id=merchant_id,
                    customer_id=customer_id,
                )
            return self._conversations[conversation_id]

    def update(self, state: ConversationState) -> None:
        with self._lock:
            state.last_activity_at = datetime.utcnow().isoformat() + "Z"
            self._conversations[state.conversation_id] = state

    def add_turn(
        self,
        conversation_id: str,
        from_role: str,
        message: str,
        turn_number: int,
    ) -> None:
        with self._lock:
            if conversation_id in self._conversations:
                state = self._conversations[conversation_id]
                state.turns.append({
                    "turn": turn_number,
                    "from": from_role,
                    "message": message,
                    "ts": datetime.utcnow().isoformat() + "Z",
                })
                state.last_activity_at = datetime.utcnow().isoformat() + "Z"

    def record_sent_message(self, conversation_id: str, message_body: str) -> None:
        with self._lock:
            if conversation_id not in self._sent_messages:
                self._sent_messages[conversation_id] = set()
            normalized = message_body.strip().lower()
            self._sent_messages[conversation_id].add(normalized)

    def was_message_sent(self, conversation_id: str, message_body: str) -> bool:
        with self._lock:
            sent = self._sent_messages.get(conversation_id, set())
            normalized = message_body.strip().lower()
            return normalized in sent

    def get(self, conversation_id: str) -> Optional[ConversationState]:
        with self._lock:
            return self._conversations.get(conversation_id)


class SuppressionStore:
    """Thread-safe store for suppressed triggers"""

    def __init__(self):
        self._lock = threading.RLock()
        self._suppressed: dict[str, SuppressedTrigger] = {}

    def suppress(self, suppression_key: str, expires_at: Optional[str] = None) -> None:
        with self._lock:
            self._suppressed[suppression_key] = SuppressedTrigger(
                suppression_key=suppression_key,
                expires_at=expires_at,
            )

    def is_suppressed(self, suppression_key: str) -> bool:
        with self._lock:
            record = self._suppressed.get(suppression_key)
            if not record:
                return False
            if record.expires_at:
                try:
                    expires = datetime.fromisoformat(record.expires_at.replace("Z", "+00:00"))
                    if datetime.utcnow().replace(tzinfo=expires.tzinfo) > expires:
                        del self._suppressed[suppression_key]
                        return False
                except ValueError:
                    pass
            return True


# Global singleton instances
context_store = ContextStore()
conversation_store = ConversationStore()
suppression_store = SuppressionStore()
