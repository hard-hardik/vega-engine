"""
Multi-turn replay engine, auto-reply detector, and intent router
Handles critical conversation scenarios including auto-replies, intent transitions, and hostile messages
"""

import re
from typing import Optional

from ..schemas import (
    CategoryContext,
    ConversationState,
    CustomerContext,
    MerchantContext,
    ReplyResponse,
    TriggerContext,
)


AUTO_REPLY_PATTERNS = [
    r"thank\s*you\s+for\s+(?:contacting|reaching|messaging)",
    r"our\s+team\s+will\s+(?:respond|get\s+back|reply)",
    r"we\s+will\s+(?:respond|get\s+back|reply)\s+(?:shortly|soon)",
    r"this\s+is\s+an?\s+auto(?:mated|matic)?\s*(?:reply|response|message)",
    r"i\s*(?:am|'m)\s+(?:an?\s+)?auto(?:mated)?\s+(?:assistant|bot|response)",
    r"(?:currently|presently)\s+(?:away|unavailable|out\s+of\s+office)",
    r"(?:will|shall)\s+(?:revert|respond)\s+(?:shortly|soon|at\s+the\s+earliest)",
    r"aapki\s+jankari\s+.*\s+pahuncha",
    r"madad\s+ke\s+liye\s+shukriya",
    r"bahut[- ]bahut\s+shukriya.*team\s+tak",
]

COMMITMENT_PATTERNS = [
    r"^(?:ok|okay|yes|yep|yeah)\b",
    r"\b(?:go\s+ahead|let'?s\s+do\s+it|sounds\s+good)\b",
    r"\b(?:please|pls)\s+(?:send|do|proceed|go\s+ahead|start)\b",
    r"(?:^|\s)(?:haan|ha|theek|thik|chalega|chalo)(?:\s|$|,|\.)",
    r"\bwhat'?s\s+next\b",
    r"\bdo\s+it\b",
    r"\blet'?s\s+go\b",
]

HOSTILE_PATTERNS = [
    r"\b(?:stop|spam|unsubscribe|remove\s+me|don'?t\s+(?:message|contact|text))\b",
    r"\b(?:useless|waste|annoying|irritating|harassing)\b",
    r"\b(?:leave\s+me\s+alone|go\s+away|not\s+interested)\b",
    r"\b(?:block(?:ed|ing)?|report(?:ed|ing)?)\b",
    r"\b(?:bakwas|faltu|bekaar|band\s+karo|mat\s+karo)\b",
]

QUALIFYING_PATTERNS = [
    r"\bwould\s+you\s+(?:like|want|prefer)\b",
    r"\bdo\s+you\s+(?:want|need|have)\b",
    r"\bcan\s+(?:you|i)\s+(?:tell|share|know)\b",
    r"\bwhat\s+if\b",
    r"\bhow\s+about\b",
    r"\bare\s+you\s+(?:interested|looking|considering)\b",
]


class ConversationEngine:
    """Handles multi-turn conversation logic and intent routing"""

    def __init__(self):
        self._auto_reply_compiled = [
            re.compile(p, re.IGNORECASE) for p in AUTO_REPLY_PATTERNS
        ]
        self._commitment_compiled = [
            re.compile(p, re.IGNORECASE) for p in COMMITMENT_PATTERNS
        ]
        self._hostile_compiled = [
            re.compile(p, re.IGNORECASE) for p in HOSTILE_PATTERNS
        ]
        self._qualifying_compiled = [
            re.compile(p, re.IGNORECASE) for p in QUALIFYING_PATTERNS
        ]

    def detect_auto_reply(self, message: str) -> bool:
        """Detect if message is an auto-reply/canned response"""
        message_clean = message.strip().lower()

        for pattern in self._auto_reply_compiled:
            if pattern.search(message_clean):
                return True

        return False

    def detect_commitment(self, message: str) -> bool:
        """Detect if merchant has committed/agreed to proceed"""
        message_clean = message.strip().lower()

        for pattern in self._commitment_compiled:
            if pattern.search(message_clean):
                return True

        return False

    def detect_hostile(self, message: str) -> bool:
        """Detect hostile/unsubscribe intent"""
        message_clean = message.strip().lower()

        for pattern in self._hostile_compiled:
            if pattern.search(message_clean):
                return True

        return False

    def detect_repeated_message(
        self,
        message: str,
        state: ConversationState,
    ) -> int:
        """Count how many times this exact message has been sent before"""
        message_normalized = message.strip().lower()
        count = 0

        for turn in state.turns:
            if turn.get("from") in ["merchant", "customer"]:
                prev_msg = turn.get("message", "").strip().lower()
                if prev_msg == message_normalized:
                    count += 1

        return count

    def handle_reply(
        self,
        message: str,
        state: ConversationState,
        category: CategoryContext,
        merchant: MerchantContext,
        customer: Optional[CustomerContext] = None,
    ) -> ReplyResponse:
        """Process incoming reply and determine appropriate response"""

        if self.detect_hostile(message):
            return ReplyResponse(
                action="end",
                rationale="Merchant indicated opt-out or hostility; gracefully exiting conversation."
            )

        is_auto_reply = self.detect_auto_reply(message)
        repeat_count = self.detect_repeated_message(message, state)

        if is_auto_reply or repeat_count >= 1:
            state.auto_reply_count += 1

            if state.auto_reply_count == 1:
                body = self._compose_auto_reply_nudge(merchant, category)
                return ReplyResponse(
                    action="send",
                    body=body,
                    cta="binary_yes_no",
                    rationale="Detected auto-reply (turn 1); sending friendly notice to reach owner directly."
                )
            elif state.auto_reply_count == 2:
                return ReplyResponse(
                    action="wait",
                    wait_seconds=86400,
                    rationale="Detected repeated auto-reply; backing off for 24 hours."
                )
            else:
                return ReplyResponse(
                    action="end",
                    rationale="Multiple auto-replies detected; ending conversation gracefully."
                )

        if self.detect_commitment(message):
            state.intent_committed = True
            body = self._compose_action_response(merchant, category, state)
            return ReplyResponse(
                action="send",
                body=body,
                cta="binary_yes_no",
                rationale="Merchant committed; transitioning immediately to action mode without further qualification."
            )

        if self._is_out_of_scope(message):
            body = self._compose_redirect_response(merchant)
            return ReplyResponse(
                action="send",
                body=body,
                cta="open_ended",
                rationale="Out-of-scope request detected; politely redirecting to core value proposition."
            )

        body = self._compose_continuation(message, state, merchant, category)
        return ReplyResponse(
            action="send",
            body=body,
            cta="open_ended",
            rationale="Continuing conversation with relevant follow-up based on merchant's response."
        )

    def _compose_auto_reply_nudge(
        self,
        merchant: MerchantContext,
        category: CategoryContext,
    ) -> str:
        """Compose message when auto-reply detected"""
        first_name = merchant.identity.owner_first_name or "there"

        if "hi" in merchant.identity.languages:
            return (
                f"Samajh gayi — yeh auto-reply lag raha hai. "
                f"Koi baat nahi, {first_name}! Main directly owner/manager se connect karna chahungi. "
                f"Kya aap available hain ya koi aur dekh sakta hai?"
            )
        else:
            return (
                f"Got it — looks like an auto-reply. No problem! "
                f"I'd like to connect directly with {first_name} or whoever manages day-to-day. "
                f"Is anyone available?"
            )

    def _compose_action_response(
        self,
        merchant: MerchantContext,
        category: CategoryContext,
        state: ConversationState,
    ) -> str:
        """Compose action response when merchant commits"""
        first_name = merchant.identity.owner_first_name or "there"

        if category.slug == "dentists":
            if "hi" in merchant.identity.languages:
                return (
                    f"Perfect, {first_name}! Main abhi start karti hoon. "
                    f"Draft ready hone par aapko bhej dungi — 2 minute mein review kar sakte ho. "
                    f"Confirm karne ke baad post ho jayega."
                )
            return (
                f"Perfect, {first_name}! Starting on this now. "
                f"I'll send you the draft to review — takes 2 minutes to confirm. "
                f"Once you approve, it goes live."
            )

        elif category.slug == "salons":
            return (
                f"Done, {first_name}! I'm drafting this for you now. "
                f"Will share the preview in a moment — just say 'Go' when you're ready to publish."
            )

        elif category.slug == "restaurants":
            return (
                f"On it, {first_name}! Putting this together now. "
                f"I'll share the draft shortly — one tap to publish when you're happy with it."
            )

        elif category.slug == "gyms":
            return (
                f"Let's go, {first_name}! Working on this right now. "
                f"Will send you the draft to review — takes 30 seconds to approve."
            )

        elif category.slug == "pharmacies":
            return (
                f"Noted, {first_name}. Processing this now. "
                f"I'll prepare the details and send them to you for confirmation."
            )

        return (
            f"Got it, {first_name}! Working on this now. "
            f"I'll send you the details to review shortly."
        )

    def _compose_redirect_response(
        self,
        merchant: MerchantContext,
    ) -> str:
        """Compose response for out-of-scope requests"""
        first_name = merchant.identity.owner_first_name or "there"
        return (
            f"Interesting question, {first_name}! That's outside what I can help with directly. "
            f"I focus on helping you with your Google profile, marketing campaigns, and customer engagement. "
            f"Shall we get back to improving your visibility?"
        )

    def _compose_continuation(
        self,
        message: str,
        state: ConversationState,
        merchant: MerchantContext,
        category: CategoryContext,
    ) -> str:
        """Compose continuation response based on conversation context"""
        first_name = merchant.identity.owner_first_name or "there"

        message_lower = message.lower()

        if any(w in message_lower for w in ["more", "details", "tell me", "explain"]):
            return self._provide_details(merchant, category, state)

        if "?" in message:
            return self._answer_question(message, merchant, category)

        return (
            f"Thanks for sharing that, {first_name}. "
            f"Based on what you've mentioned, I can help you take the next step. "
            f"Want me to proceed?"
        )

    def _provide_details(
        self,
        merchant: MerchantContext,
        category: CategoryContext,
        state: ConversationState,
    ) -> str:
        """Provide more details based on conversation context"""
        first_name = merchant.identity.owner_first_name or "there"

        if category.slug == "dentists":
            return (
                f"Of course, {first_name}. The key points: "
                f"this is based on verified data from your peer group in {merchant.identity.locality}. "
                f"Implementation takes about 5 minutes total — I handle the technical parts. "
                f"Want me to walk you through step by step, or should I just get started?"
            )

        return (
            f"Happy to explain more, {first_name}. "
            f"Here's the gist: minimal effort on your end, I handle the details. "
            f"Results typically show within a week. "
            f"Ready to give it a try?"
        )

    def _answer_question(
        self,
        message: str,
        merchant: MerchantContext,
        category: CategoryContext,
    ) -> str:
        """Answer a question from the merchant"""
        first_name = merchant.identity.owner_first_name or "there"

        return (
            f"Good question, {first_name}. "
            f"Based on what I see in your data, I can give you a clear answer once we proceed. "
            f"Shall I show you?"
        )

    def _is_out_of_scope(self, message: str) -> bool:
        """Check if message is requesting something out of scope"""
        out_of_scope_keywords = [
            "gst", "tax", "filing", "invoice", "accounting",
            "loan", "finance", "bank", "insurance",
            "legal", "lawyer", "court",
            "employee", "salary", "payroll",
            "rent", "lease", "landlord",
        ]

        message_lower = message.lower()
        return any(kw in message_lower for kw in out_of_scope_keywords)

    def should_avoid_qualifying(self, state: ConversationState) -> bool:
        """Check if we should avoid asking qualifying questions"""
        return state.intent_committed

    def get_conversation_summary(self, state: ConversationState) -> dict:
        """Get summary of conversation for context"""
        return {
            "turn_count": len(state.turns),
            "auto_reply_count": state.auto_reply_count,
            "intent_committed": state.intent_committed,
            "last_vera_message": state.last_vera_message[:100] if state.last_vera_message else "",
        }
