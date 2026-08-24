"""
Prompt builder and LLM dispatcher for /v1/tick
Composes messages that satisfy judge scoring criteria
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Optional

from ..schemas import (
    CategoryContext,
    ComposedAction,
    CustomerContext,
    MerchantContext,
    TriggerContext,
)
from .grounding import GroundingValidator
from .llm_client import LLMClient
from .tone_adapter import ToneAdapter


SYSTEM_PROMPT = """You are Vera, magicpin's AI assistant for local merchants. You compose WhatsApp messages to help merchants grow their business.

CRITICAL RULES (Violations = instant penalty):
1. NEVER fabricate data. Only use facts from the provided context.
2. NEVER use taboo words for this category.
3. ALWAYS address the merchant by their first name using the correct salutation.
4. ALWAYS state "why now" (the trigger) in the FIRST sentence.
5. ALWAYS end with exactly ONE clear, low-friction CTA.
6. NEVER ask multiple questions in one message.
7. Keep messages concise (under 250 words).
8. Match the category's voice/tone register.
9. For Hindi-speaking merchants, use natural Hindi-English code-mix.
10. Include specific numbers, dates, and source citations when available.

SCORING DIMENSIONS (you must maximize ALL):
- Specificity: Use real numbers, dates, sources from context
- Category Fit: Match the voice (clinical for dentists, warm for salons, etc.)
- Merchant Fit: Personalize with their name, data, language
- Trigger Relevance: Make "why now" crystal clear
- Engagement Compulsion: Use curiosity, loss aversion, social proof

OUTPUT FORMAT:
Respond with ONLY a JSON object:
{
    "body": "The WhatsApp message",
    "cta": "binary_yes_no|open_ended|slot_selection|none",
    "template_name": "template_identifier",
    "rationale": "Brief explanation of composition choices"
}
"""


class MessageComposer:
    """Composes messages using the 4-Context Framework"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self.tone_adapter = ToneAdapter()
        self.grounding = GroundingValidator()

    async def compose(
        self,
        category: CategoryContext,
        merchant: MerchantContext,
        trigger: TriggerContext,
        customer: Optional[CustomerContext] = None,
    ) -> ComposedAction:
        """Compose a message for the given context"""

        if not self.llm.is_configured():
            return self._compose_fallback(category, merchant, trigger, customer)

        prompt = self._build_prompt(category, merchant, trigger, customer)

        try:
            response = await asyncio.wait_for(
                self.llm.complete(prompt, system=SYSTEM_PROMPT),
                timeout=25.0
            )
            action = self._parse_response(response, category, merchant, trigger, customer)
        except (asyncio.TimeoutError, Exception) as e:
            action = self._compose_fallback(category, merchant, trigger, customer)

        is_valid, issues = self.tone_adapter.validate_voice(
            action.body, category, merchant
        )
        if not is_valid:
            action = self._compose_fallback(category, merchant, trigger, customer)

        return action

    def _build_prompt(
        self,
        category: CategoryContext,
        merchant: MerchantContext,
        trigger: TriggerContext,
        customer: Optional[CustomerContext] = None,
    ) -> str:
        """Build the LLM prompt with all context"""
        lines = []

        lines.append("=== COMPOSE MESSAGE FOR THIS CONTEXT ===\n")

        salutation = self.tone_adapter.get_salutation(category, merchant)
        lines.append(f"Salutation to use: {salutation}")

        tone_instructions = self.tone_adapter.get_tone_instructions(category)
        lines.append(f"\nTone Instructions:\n{tone_instructions}")

        grounded_context = self.grounding.build_grounded_context(
            category, merchant, trigger, customer
        )
        lines.append(f"\n{grounded_context}")

        lines.append(f"\n=== TRIGGER DETAILS ===")
        lines.append(f"Trigger Kind: {trigger.kind}")
        lines.append(f"Urgency: {trigger.urgency}/5")
        lines.append(f"Source: {trigger.source}")

        lines.append(self._get_trigger_specific_instructions(trigger, category, merchant))

        if customer:
            lines.append(f"\n=== CUSTOMER-FACING MESSAGE ===")
            lines.append(f"Send As: merchant_on_behalf (from {merchant.identity.name})")
            lines.append(f"Customer Name: {customer.identity.name}")
            lines.append(f"Language Preference: {customer.identity.language_pref}")
            lines.append(f"Preferred Slots: {customer.preferences.preferred_slots}")
        else:
            lines.append(f"\n=== MERCHANT-FACING MESSAGE ===")
            lines.append("Send As: vera")

        return "\n".join(lines)

    def _get_trigger_specific_instructions(
        self,
        trigger: TriggerContext,
        category: CategoryContext,
        merchant: MerchantContext,
    ) -> str:
        """Get trigger-specific composition instructions"""
        kind = trigger.kind
        payload = trigger.payload

        if kind == "research_digest":
            item_id = payload.get("top_item_id")
            digest_item = None
            for item in category.digest:
                if item.id == item_id:
                    digest_item = item
                    break

            if digest_item:
                return f"""
RESEARCH DIGEST MESSAGE:
- Lead with the specific finding: "{digest_item.title}"
- Include source citation: "{digest_item.source}"
- If trial_n available: mention the sample size ({digest_item.trial_n if digest_item.trial_n else 'N/A'})
- Connect to merchant's patient segment if applicable
- CTA: Offer to pull the abstract or draft patient-ed content
"""
            return "\nResearch digest - cite the source and offer value."

        elif kind == "perf_dip":
            metric = payload.get("metric", "performance")
            delta = payload.get("delta_pct", 0)
            return f"""
PERFORMANCE DIP ALERT:
- {metric} dropped {abs(delta):.0%} in 7 days
- Frame with curiosity, not alarm: "Noticed something..."
- Offer diagnostic insight or quick fix
- CTA: Binary yes/no for help
"""

        elif kind == "perf_spike":
            metric = payload.get("metric", "performance")
            delta = payload.get("delta_pct", 0)
            return f"""
PERFORMANCE SPIKE - GOOD NEWS:
- {metric} up {delta:.0%}
- Celebrate briefly, suggest capitalizing
- CTA: Offer to boost momentum
"""

        elif kind == "recall_due":
            return f"""
RECALL REMINDER (customer-facing):
- State the specific recall type and timing
- Reference their preferred slots: {payload.get('available_slots', [])}
- Include the service and price
- CTA: Slot selection (Reply 1 for X, 2 for Y)
"""

        elif kind == "renewal_due":
            days = payload.get("days_remaining", 0)
            return f"""
SUBSCRIPTION RENEWAL:
- {days} days remaining
- Emphasize continuity of service
- Don't be pushy; be helpful
- CTA: Binary confirmation or call back
"""

        elif kind == "festival_upcoming":
            festival = payload.get("festival", "")
            days_until = payload.get("days_until", 0)
            return f"""
FESTIVAL OPPORTUNITY:
- {festival} in {days_until} days
- Suggest timely campaign or offer
- Connect to category-relevant opportunity
- CTA: Open-ended suggestion
"""

        elif kind == "review_theme_emerged":
            theme = payload.get("theme", "")
            return f"""
REVIEW THEME ALERT:
- Theme: "{theme}" emerging in recent reviews
- Don't be negative; be constructive
- Offer to help address it
- CTA: Ask if they want suggestions
"""

        elif kind == "ipl_match_today":
            match = payload.get("match", "")
            return f"""
IPL MATCH DAY:
- {match} tonight
- Suggest match-night promotion
- Keep it brief and actionable
- CTA: Quick yes/no for running the promo
"""

        elif kind == "competitor_opened":
            competitor = payload.get("competitor_name", "")
            distance = payload.get("distance_km", 0)
            return f"""
COMPETITIVE ALERT:
- New competitor "{competitor}" opened {distance}km away
- Frame constructively: opportunity to differentiate
- Don't be alarmist
- CTA: Offer competitive audit or response strategy
"""

        elif kind in ["chronic_refill_due", "supply_alert"]:
            return """
PHARMACY/REFILL MESSAGE:
- Be precise about molecules/batches
- Reference patient safety
- Keep professional tone
- CTA: Confirm action or delivery
"""

        elif kind == "curious_ask_due":
            return """
CURIOUS ASK (engagement):
- Ask a genuine question about their business
- Show curiosity, not data
- Build relationship
- CTA: Open-ended question
"""

        elif kind == "regulation_change":
            return f"""
COMPLIANCE/REGULATION ALERT:
- State the specific regulation change
- Include deadline if available: {payload.get('deadline_iso', 'TBD')}
- Offer to help audit compliance
- CTA: Binary yes/no for audit help
"""

        return "\nCompose an appropriate message based on the trigger kind."

    def _parse_response(
        self,
        response: str,
        category: CategoryContext,
        merchant: MerchantContext,
        trigger: TriggerContext,
        customer: Optional[CustomerContext] = None,
    ) -> ComposedAction:
        """Parse LLM response into ComposedAction"""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                body = data.get("body", "")
                cta = data.get("cta", "open_ended")
                if cta not in ["binary_yes_no", "open_ended", "slot_selection", "none"]:
                    cta = "open_ended"

                return ComposedAction(
                    conversation_id=self._generate_conversation_id(merchant, trigger, customer),
                    merchant_id=merchant.merchant_id,
                    customer_id=customer.customer_id if customer else None,
                    send_as="merchant_on_behalf" if customer else "vera",
                    trigger_id=trigger.id,
                    template_name=data.get("template_name", f"vera_{trigger.kind}_v1"),
                    template_params=self._extract_template_params(body),
                    body=body,
                    cta=cta,
                    suppression_key=trigger.suppression_key,
                    rationale=data.get("rationale", "LLM-composed message"),
                )
        except (json.JSONDecodeError, KeyError):
            pass

        return self._compose_fallback(category, merchant, trigger, customer)

    def _compose_fallback(
        self,
        category: CategoryContext,
        merchant: MerchantContext,
        trigger: TriggerContext,
        customer: Optional[CustomerContext] = None,
    ) -> ComposedAction:
        """Compose deterministic fallback message without LLM"""

        salutation = self.tone_adapter.get_salutation(category, merchant)
        first_name = merchant.identity.owner_first_name or "there"

        body = self._get_fallback_body(
            category, merchant, trigger, customer, salutation, first_name
        )

        cta = self._get_fallback_cta(trigger)

        return ComposedAction(
            conversation_id=self._generate_conversation_id(merchant, trigger, customer),
            merchant_id=merchant.merchant_id,
            customer_id=customer.customer_id if customer else None,
            send_as="merchant_on_behalf" if customer else "vera",
            trigger_id=trigger.id,
            template_name=f"vera_{trigger.kind}_v1",
            template_params=self._extract_template_params(body),
            body=body,
            cta=cta,
            suppression_key=trigger.suppression_key,
            rationale=f"Deterministic fallback for {trigger.kind} trigger with grounded facts from context.",
        )

    def _get_fallback_body(
        self,
        category: CategoryContext,
        merchant: MerchantContext,
        trigger: TriggerContext,
        customer: Optional[CustomerContext],
        salutation: str,
        first_name: str,
    ) -> str:
        """Generate fallback message body based on trigger kind"""
        kind = trigger.kind
        payload = trigger.payload
        use_hindi = "hi" in merchant.identity.languages

        if kind == "research_digest":
            item_id = payload.get("top_item_id")
            for item in category.digest:
                if item.id == item_id:
                    if use_hindi:
                        return (
                            f"{salutation}, {item.source} ki latest issue aa gayi hai. "
                            f"Ek relevant finding: {item.title}. "
                            f"{'Trial mein ' + str(item.trial_n) + ' patients the — ' if item.trial_n else ''}"
                            f"Kya main abstract pull karoon aur patient-ed draft bhejoon?"
                        )
                    return (
                        f"{salutation}, {item.source} just landed. "
                        f"One item relevant to your practice: {item.title}. "
                        f"{'Based on a ' + str(item.trial_n) + '-patient trial. ' if item.trial_n else ''}"
                        f"Want me to pull the abstract + draft patient-ed content you can share?"
                    )

            return f"{salutation}, new research digest available for {category.slug}. Want me to share the highlights?"

        elif kind == "perf_dip":
            metric = payload.get("metric", "performance")
            delta = payload.get("delta_pct", -0.2)
            if use_hindi:
                return (
                    f"{salutation}, aapki {metric} pichhle hafte se {abs(delta):.0%} neeche aayi hai. "
                    f"Main ek quick check kar sakti hoon kya issue hai. "
                    f"Kya aap chahenge main dekh loon?"
                )
            return (
                f"{salutation}, noticed your {metric} dropped {abs(delta):.0%} vs last week. "
                f"I can run a quick diagnostic to spot the issue. "
                f"Want me to take a look?"
            )

        elif kind == "perf_spike":
            metric = payload.get("metric", "views")
            delta = payload.get("delta_pct", 0.15)
            return (
                f"{salutation}, great news — your {metric} are up {delta:.0%} this week! "
                f"Want me to help you capitalize on this momentum?"
            )

        elif kind == "recall_due" and customer:
            service = payload.get("service_due", "recall")
            slots = payload.get("available_slots", [])
            slot_text = ""
            if slots:
                slot_labels = [s.get("label", "") for s in slots[:2]]
                slot_text = f"Slots available: {' ya '.join(slot_labels) if use_hindi else ' or '.join(slot_labels)}. "

            if use_hindi:
                return (
                    f"Hi {customer.identity.name}, {merchant.identity.name} ki taraf se. "
                    f"Aapki {service.replace('_', ' ')} due hai. {slot_text}"
                    f"Reply 1 for first slot, 2 for second, ya apna time batayein."
                )
            return (
                f"Hi {customer.identity.name}, this is {merchant.identity.name}. "
                f"Your {service.replace('_', ' ')} is due. {slot_text}"
                f"Reply 1 for first slot, 2 for second, or tell us a time that works."
            )

        elif kind == "renewal_due":
            days = payload.get("days_remaining", 30)
            if use_hindi:
                return (
                    f"{salutation}, aapka subscription {days} din mein expire ho raha hai. "
                    f"Profile maintenance continue rahe iske liye renew kar lein? "
                    f"Main details bhej sakti hoon."
                )
            return (
                f"{salutation}, your subscription expires in {days} days. "
                f"Renew to keep your profile maintenance running? "
                f"I can share the details."
            )

        elif kind == "festival_upcoming":
            festival = payload.get("festival", "upcoming festival")
            days = payload.get("days_until", 30)
            return (
                f"{salutation}, {festival} is {days} days away. "
                f"Want me to draft a campaign for your customers?"
            )

        elif kind == "ipl_match_today":
            match = payload.get("match", "IPL match")
            return (
                f"{salutation}, {match} tonight! "
                f"Want me to push a match-night combo offer to drive footfall?"
            )

        elif kind == "review_theme_emerged":
            theme = payload.get("theme", "service quality")
            count = payload.get("occurrences_30d", 3)
            return (
                f"{salutation}, noticed '{theme}' came up {count} times in recent reviews. "
                f"Want me to share some tips on how to address it?"
            )

        elif kind == "competitor_opened":
            competitor = payload.get("competitor_name", "a new competitor")
            distance = payload.get("distance_km", 1.0)
            return (
                f"{salutation}, {competitor} just opened {distance}km from you. "
                f"Want me to run a competitive analysis and suggest differentiation?"
            )

        elif kind == "regulation_change":
            return (
                f"{salutation}, new compliance update you should know about. "
                f"I can help you audit your setup before the deadline. Want details?"
            )

        elif kind == "supply_alert":
            molecule = payload.get("molecule", "medication")
            return (
                f"{salutation}, heads up: voluntary recall on certain {molecule} batches. "
                f"Want me to filter your customer list for affected prescriptions?"
            )

        elif kind == "chronic_refill_due" and customer:
            molecules = payload.get("molecule_list", [])
            mol_str = ", ".join(molecules[:3]) if molecules else "your regular medications"
            return (
                f"Hi {customer.identity.name}, {merchant.identity.name} se. "
                f"Aapki {mol_str} refill due hai. "
                f"Delivery schedule karein?"
            )

        elif kind == "curious_ask_due":
            return (
                f"{salutation}, quick question — what's been your most-requested "
                f"{'service' if category.slug not in ['restaurants'] else 'dish'} this week? "
                f"Helps me spot what to promote."
            )

        elif kind == "dormant_with_vera":
            days = payload.get("days_since_last_merchant_message", 14)
            return (
                f"{salutation}, it's been {days} days since we chatted. "
                f"Just checking in — anything I can help with for your profile or marketing?"
            )

        elif kind == "milestone_reached":
            metric = payload.get("metric", "")
            value = payload.get("value_now", 0)
            return (
                f"{salutation}, congrats — you're about to hit {value} {metric}! "
                f"Want me to help celebrate and share this milestone?"
            )

        return f"{salutation}, I have an update for you. Let me know if you'd like to hear more."

    def _get_fallback_cta(self, trigger: TriggerContext) -> str:
        """Determine appropriate CTA type for trigger"""
        kind = trigger.kind

        binary_triggers = [
            "perf_dip", "perf_spike", "renewal_due", "ipl_match_today",
            "competitor_opened", "review_theme_emerged", "supply_alert",
            "regulation_change", "milestone_reached",
        ]
        if kind in binary_triggers:
            return "binary_yes_no"

        if kind == "recall_due" and trigger.payload.get("available_slots"):
            return "slot_selection"

        return "open_ended"

    def _generate_conversation_id(
        self,
        merchant: MerchantContext,
        trigger: TriggerContext,
        customer: Optional[CustomerContext] = None,
    ) -> str:
        """Generate unique conversation ID"""
        parts = ["conv"]
        if customer:
            parts.append(f"c_{customer.customer_id[:15]}")
        else:
            parts.append(f"m_{merchant.merchant_id[:15]}")
        parts.append(trigger.kind[:15])
        parts.append(datetime.utcnow().strftime("%Y%m%d"))
        return "_".join(parts)

    def _extract_template_params(self, body: str) -> list[str]:
        """Extract potential template parameters from message body"""
        params = []
        sentences = body.split(".")
        if sentences:
            params.append(sentences[0].strip()[:50])
            if len(sentences) > 1:
                params.append(sentences[-1].strip()[:50])
        return params[:3]
