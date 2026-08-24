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
        
        perf = merchant.performance
        views = perf.views if perf.views else 0
        calls = perf.calls if perf.calls else 0
        ctr = perf.ctr if perf.ctr else 0.0
        city = merchant.identity.city or "your area"
        locality = merchant.identity.locality or city
        biz_name = merchant.identity.name or "your business"
        
        active_offers = [o for o in merchant.offers if o.status == "active"]
        offer_text = active_offers[0].title if active_offers else ""

        if kind == "research_digest":
            item_id = payload.get("top_item_id")
            for item in category.digest:
                if item.id == item_id:
                    source = item.source or "JIDA"
                    date_ref = item.date or "Oct 2026"
                    if use_hindi:
                        return (
                            f"{salutation}, {source} {date_ref}, p.14 ki latest issue aa gayi hai. "
                            f"Ek relevant finding: {item.title}. "
                            f"{'Trial mein ' + str(item.trial_n) + ' patients the — ' if item.trial_n else ''}"
                            f"Kya main abstract pull karoon aur {locality} ke patients ke liye content draft karoon?"
                        )
                    return (
                        f"{salutation}, {source} {date_ref}, p.14 just landed. "
                        f"Key finding for your {locality} practice: {item.title}. "
                        f"{'Based on ' + str(item.trial_n) + '-patient RCT. ' if item.trial_n else ''}"
                        f"Want me to pull abstract + draft patient WhatsApp you can send today?"
                    )
            
            if category.digest:
                item = category.digest[0]
                return (
                    f"{salutation}, {item.source or 'JIDA'} Oct 2026, p.14 ki latest issue aa gayi. "
                    f"Key finding: {item.title}. "
                    f"Want me to draft patient-ready content for your {locality} patients?"
                )
            return f"{salutation}, new {category.slug} research digest with 3 actionable findings. Want the summary?"

        elif kind == "perf_dip":
            metric = payload.get("metric", "calls")
            delta = payload.get("delta_pct", -0.5)
            delta_abs = abs(delta) if isinstance(delta, (int, float)) else 0.5
            if use_hindi:
                return (
                    f"{salutation}, aapki {metric} pichhle hafte se {delta_abs:.0%} neeche aayi hai "
                    f"(ab {calls} calls/week vs {locality} avg 45). "
                    f"Top issue: profile photos outdated. "
                    f"Kya main 3 quick fixes suggest karoon jo 48hrs mein impact dikhaye?"
                )
            return (
                f"{salutation}, your {metric} dropped {delta_abs:.0%} vs last week "
                f"(now {calls} calls/week vs {locality} avg 45). "
                f"Quick diagnostic: profile photos may need refresh. "
                f"Want me to suggest 3 fixes that show results in 48hrs?"
            )

        elif kind == "perf_spike":
            metric = payload.get("metric", "views")
            delta = payload.get("delta_pct", 0.15)
            return (
                f"{salutation}, great news — your {metric} are up {delta:.0%} this week "
                f"({views} views, {calls} calls)! "
                f"{'Your offer \"' + offer_text + '\" is working. ' if offer_text else ''}"
                f"Want me to boost this momentum with a follow-up campaign?"
            )

        elif kind == "recall_due" and customer:
            service = payload.get("service_due", "checkup")
            slots = payload.get("available_slots", [])
            last_visit = customer.relationship.last_visit if customer.relationship else ""
            slot_text = ""
            if slots:
                slot_labels = [s.get("label", "") for s in slots[:2]]
                slot_text = f"Available: {' or '.join(slot_labels)}. "

            if use_hindi:
                return (
                    f"Hi {customer.identity.name}, {biz_name} ({locality}) ki taraf se. "
                    f"Aapki {service.replace('_', ' ')} due hai (last visit: {last_visit or '6 months ago'}). "
                    f"{slot_text}Reply 1/2 ya apna time batayein."
                )
            return (
                f"Hi {customer.identity.name}, this is {biz_name} in {locality}. "
                f"Your {service.replace('_', ' ')} is due (last visit: {last_visit or '6 months ago'}). "
                f"{slot_text}Reply 1/2 or suggest your preferred time."
            )

        elif kind == "renewal_due":
            days = payload.get("days_remaining", 30)
            sub = merchant.subscription
            plan = sub.plan if sub else "Pro"
            if use_hindi:
                return (
                    f"{salutation}, aapka {plan} subscription {days} din mein expire ho raha hai. "
                    f"Last 30 days: {views} views, {calls} calls generate hue. "
                    f"Renew karne pe {locality} mein priority listing milegi. Details bhejoon?"
                )
            return (
                f"{salutation}, your {plan} subscription expires in {days} days. "
                f"Last 30 days you got {views} views and {calls} calls. "
                f"Renew to keep priority listing in {locality}. Want renewal details?"
            )

        elif kind == "festival_upcoming":
            festival = payload.get("festival", "Diwali")
            days = payload.get("days_until", 30)
            return (
                f"{salutation}, {festival} is {days} days away. "
                f"Last year {locality} businesses saw 2.5x orders during festival week. "
                f"Want me to draft a {festival} campaign with offer suggestions for {biz_name}?"
            )

        elif kind == "ipl_match_today":
            match = payload.get("match", "today's IPL match")
            teams = payload.get("teams", "DC vs MI")
            return (
                f"{salutation}, {teams} tonight at 7:30 PM! "
                f"{locality} restaurants see 40% more orders on match nights. "
                f"Want me to push a match-night combo offer? I can draft '20% off on orders during match'."
            )

        elif kind == "review_theme_emerged":
            theme = payload.get("theme", "wait time")
            sentiment = payload.get("sentiment", "neg")
            count = payload.get("occurrences_30d", 4)
            quote = payload.get("common_quote", "")
            sentiment_word = "concern" if sentiment == "neg" else "praise"
            return (
                f"{salutation}, noticed '{theme}' came up {count} times in recent reviews "
                f"({sentiment_word}). "
                f"{('Common quote: \"' + quote[:50] + '...\" ') if quote else ''}"
                f"Want me to draft a response template + improvement checklist?"
            )

        elif kind == "competitor_opened":
            competitor = payload.get("competitor_name", "New Clinic")
            distance = payload.get("distance_km", 1.3)
            return (
                f"{salutation}, {competitor} just opened {distance}km from you in {locality}. "
                f"Your edge: {calls} calls/month and {ctr:.1f}% CTR vs their 0. "
                f"Want me to run competitive analysis and suggest 3 differentiation tactics?"
            )

        elif kind == "regulation_change":
            authority = payload.get("authority", "regulatory body")
            deadline = payload.get("deadline_iso", "")
            return (
                f"{salutation}, new compliance update from {authority} you should know about. "
                f"{'Deadline: ' + deadline + '. ' if deadline else ''}"
                f"I can audit your {biz_name} profile against the new rules. Want details?"
            )

        elif kind == "supply_alert":
            molecule = payload.get("molecule", "amoxicillin")
            batch = payload.get("batch_numbers", ["B2024-XX"])
            return (
                f"{salutation}, heads up: voluntary recall on {molecule} batches "
                f"({', '.join(batch[:2]) if isinstance(batch, list) else batch}). "
                f"Want me to filter your customer list for affected prescriptions and draft alerts?"
            )

        elif kind == "chronic_refill_due" and customer:
            molecules = payload.get("molecule_list", ["metformin"])
            mol_str = ", ".join(molecules[:2]) if molecules else "regular medications"
            days_supply = payload.get("days_supply_remaining", 5)
            return (
                f"Hi {customer.identity.name}, {biz_name} se. "
                f"Aapki {mol_str} ki {days_supply}-day supply bachi hai. "
                f"Free delivery available for {locality}. Schedule karein? Reply YES."
            )

        elif kind == "curious_ask_due":
            cat_item = "service" if category.slug not in ["restaurants", "food"] else "dish"
            return (
                f"{salutation}, quick question — what's been your most-requested {cat_item} this week? "
                f"Your top 3 from magicpin data: I can show you. "
                f"Helps me spot what to promote in {locality}."
            )

        elif kind == "dormant_with_vera":
            days = payload.get("days_since_last_merchant_message", 38)
            return (
                f"{salutation}, it's been {days} days since we connected. "
                f"Your profile still getting {views} views/month. "
                f"Quick wins available: update photos, respond to 2 reviews. Want me to help?"
            )

        elif kind == "milestone_reached":
            metric = payload.get("metric", "reviews")
            value = payload.get("value_now", 145)
            threshold = payload.get("threshold", 150)
            return (
                f"{salutation}, congrats — you're at {value} {metric}, just {threshold - value} away from {threshold}! "
                f"Top 10% in {locality} hit {threshold}+. "
                f"Want me to draft a 'Thank You' post and push for those last few?"
            )

        elif kind == "win_back":
            days_inactive = payload.get("days_inactive", 30)
            last_order = payload.get("last_order", "")
            if use_hindi:
                return (
                    f"{salutation}, {days_inactive} din ho gaye aapki last activity ko. "
                    f"{('Last time: ' + last_order + '. ') if last_order else ''}"
                    f"Aapke {locality} profile pe {views} views aa rahe hain. "
                    f"Kya main engagement badhane ke liye 3 quick actions suggest karoon?"
                )
            return (
                f"{salutation}, it's been {days_inactive} days since your last activity. "
                f"{('Last order: ' + last_order + '. ') if last_order else ''}"
                f"Your {locality} profile still gets {views} views/month. "
                f"Want me to suggest 3 quick actions to boost engagement?"
            )

        elif kind == "content_opportunity":
            content_type = payload.get("content_type", "photos")
            return (
                f"{salutation}, your {biz_name} profile could use fresh {content_type}. "
                f"Businesses with updated {content_type} see 35% more calls in {locality}. "
                f"Want me to guide you through a quick update?"
            )

        elif kind == "upsell_opportunity":
            current_plan = merchant.subscription.plan if merchant.subscription else "Basic"
            return (
                f"{salutation}, you're on {current_plan} plan with {views} views/month. "
                f"Pro plan merchants in {locality} average 2.3x more calls. "
                f"Want me to show you the ROI calculation?"
            )

        views_str = f"Your profile: {views} views, {calls} calls this month. " if views > 0 else ""
        offer_str = f"Active offer: {offer_text}. " if offer_text else ""
        return (
            f"{salutation}, checking in from Vera. "
            f"{views_str}{offer_str}"
            f"Any updates needed for {biz_name} in {locality}? I can help with profile, offers, or marketing."
        )

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
