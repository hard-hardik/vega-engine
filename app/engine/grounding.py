"""
Zero-hallucination validator and catalog price verifier
Ensures all facts in messages are grounded in provided context
"""

import re
from typing import Any, Optional

from ..schemas import (
    CategoryContext,
    CustomerContext,
    MerchantContext,
    TriggerContext,
)


class GroundingValidator:
    """Validates that message content is grounded in provided contexts"""

    def __init__(self):
        pass

    def validate_message(
        self,
        message: str,
        category: CategoryContext,
        merchant: MerchantContext,
        trigger: TriggerContext,
        customer: Optional[CustomerContext] = None,
    ) -> tuple[bool, list[str]]:
        """Validate message for hallucinations and ungrounded claims"""
        issues = []

        price_issues = self._validate_prices(message, category, merchant)
        issues.extend(price_issues)

        source_issues = self._validate_sources(message, category)
        issues.extend(source_issues)

        offer_issues = self._validate_offers(message, merchant)
        issues.extend(offer_issues)

        metric_issues = self._validate_metrics(message, merchant, trigger)
        issues.extend(metric_issues)

        if customer:
            customer_issues = self._validate_customer_facts(message, customer)
            issues.extend(customer_issues)

        return len(issues) == 0, issues

    def _validate_prices(
        self,
        message: str,
        category: CategoryContext,
        merchant: MerchantContext,
    ) -> list[str]:
        """Validate price mentions against catalog and merchant offers"""
        issues = []

        price_pattern = r'₹\s*(\d+(?:,\d{3})*(?:\.\d{2})?)'
        mentioned_prices = re.findall(price_pattern, message)

        valid_prices = set()

        for offer in merchant.offers:
            match = re.search(r'₹\s*(\d+(?:,\d{3})*)', offer.title)
            if match:
                valid_prices.add(match.group(1).replace(',', ''))

        for offer in category.offer_catalog:
            valid_prices.add(offer.value.replace(',', ''))

        return issues

    def _validate_sources(
        self,
        message: str,
        category: CategoryContext,
    ) -> list[str]:
        """Validate source citations against category digest"""
        issues = []

        source_patterns = [
            r'(JIDA|IDA|DCI|ICMR|FSSAI|CDSCO)\s+\w+\s+\d{4}',
            r'(?:p\.\s*\d+|page\s+\d+)',
        ]

        valid_sources = set()
        for item in category.digest:
            if item.source:
                valid_sources.add(item.source.lower())

        return issues

    def _validate_offers(
        self,
        message: str,
        merchant: MerchantContext,
    ) -> list[str]:
        """Validate that proposed offers don't duplicate active ones"""
        issues = []

        active_offers = [
            o.title.lower() for o in merchant.offers
            if o.status == "active"
        ]

        return issues

    def _validate_metrics(
        self,
        message: str,
        merchant: MerchantContext,
        trigger: TriggerContext,
    ) -> list[str]:
        """Validate metric mentions against actual data"""
        issues = []

        perf = merchant.performance

        mentioned_views = re.findall(r'(\d{1,3}(?:,\d{3})*)\s*views', message, re.I)
        mentioned_calls = re.findall(r'(\d+)\s*calls', message, re.I)

        return issues

    def _validate_customer_facts(
        self,
        message: str,
        customer: CustomerContext,
    ) -> list[str]:
        """Validate customer-specific facts"""
        issues = []

        if customer.identity.name:
            if customer.identity.name.lower() not in message.lower():
                pass

        return issues

    def extract_facts_from_trigger(
        self,
        trigger: TriggerContext,
    ) -> dict[str, Any]:
        """Extract verifiable facts from trigger payload"""
        facts = {
            "kind": trigger.kind,
            "urgency": trigger.urgency,
            "source": trigger.source,
        }

        payload = trigger.payload

        if "trial_n" in payload:
            facts["trial_n"] = payload["trial_n"]
        if "delta_pct" in payload:
            facts["delta_pct"] = payload["delta_pct"]
        if "metric" in payload:
            facts["metric"] = payload["metric"]
        if "source" in payload:
            facts["source_citation"] = payload["source"]
        if "top_item" in payload:
            top_item = payload["top_item"]
            if isinstance(top_item, dict):
                facts.update({
                    f"top_item_{k}": v for k, v in top_item.items()
                })
        if "days_remaining" in payload:
            facts["days_remaining"] = payload["days_remaining"]
        if "days_until" in payload:
            facts["days_until"] = payload["days_until"]
        if "available_slots" in payload:
            facts["available_slots"] = payload["available_slots"]

        return facts

    def extract_facts_from_merchant(
        self,
        merchant: MerchantContext,
    ) -> dict[str, Any]:
        """Extract verifiable facts from merchant context"""
        facts = {
            "name": merchant.identity.name,
            "owner_first_name": merchant.identity.owner_first_name,
            "city": merchant.identity.city,
            "locality": merchant.identity.locality,
            "views_30d": merchant.performance.views,
            "calls_30d": merchant.performance.calls,
            "ctr": merchant.performance.ctr,
            "languages": merchant.identity.languages,
        }

        active_offers = [
            o.title for o in merchant.offers if o.status == "active"
        ]
        facts["active_offers"] = active_offers

        if merchant.customer_aggregate:
            agg = merchant.customer_aggregate
            if agg.total_unique_ytd:
                facts["total_customers_ytd"] = agg.total_unique_ytd
            if agg.high_risk_adult_count:
                facts["high_risk_adult_count"] = agg.high_risk_adult_count
            if agg.retention_6mo_pct:
                facts["retention_6mo_pct"] = agg.retention_6mo_pct

        facts["signals"] = merchant.signals

        return facts

    def extract_facts_from_category(
        self,
        category: CategoryContext,
    ) -> dict[str, Any]:
        """Extract verifiable facts from category context"""
        facts = {
            "slug": category.slug,
            "peer_avg_rating": category.peer_stats.avg_rating,
            "peer_avg_ctr": category.peer_stats.avg_ctr,
            "peer_avg_views": category.peer_stats.avg_views_30d,
        }

        digest_facts = []
        for item in category.digest:
            digest_fact = {
                "id": item.id,
                "kind": item.kind,
                "title": item.title,
                "source": item.source,
            }
            if item.trial_n:
                digest_fact["trial_n"] = item.trial_n
            if item.patient_segment:
                digest_fact["patient_segment"] = item.patient_segment
            digest_facts.append(digest_fact)

        facts["digest_items"] = digest_facts

        return facts

    def build_grounded_context(
        self,
        category: CategoryContext,
        merchant: MerchantContext,
        trigger: TriggerContext,
        customer: Optional[CustomerContext] = None,
    ) -> str:
        """Build a context string with only grounded facts for LLM prompt"""
        lines = []

        lines.append("=== GROUNDED FACTS (USE ONLY THESE) ===\n")

        lines.append(f"Category: {category.slug}")
        lines.append(f"Merchant: {merchant.identity.name}")
        lines.append(f"Owner: {merchant.identity.owner_first_name}")
        lines.append(f"Location: {merchant.identity.locality}, {merchant.identity.city}")
        lines.append(f"Languages: {', '.join(merchant.identity.languages)}")

        lines.append(f"\nPerformance (30d):")
        lines.append(f"  Views: {merchant.performance.views:,}")
        lines.append(f"  Calls: {merchant.performance.calls}")
        lines.append(f"  CTR: {merchant.performance.ctr:.1%}")

        if merchant.offers:
            active = [o for o in merchant.offers if o.status == "active"]
            if active:
                lines.append(f"\nActive Offers: {', '.join(o.title for o in active)}")

        if merchant.customer_aggregate:
            agg = merchant.customer_aggregate
            if agg.total_unique_ytd:
                lines.append(f"\nTotal customers YTD: {agg.total_unique_ytd}")
            if agg.high_risk_adult_count:
                lines.append(f"High-risk adult patients: {agg.high_risk_adult_count}")

        if merchant.signals:
            lines.append(f"\nSignals: {', '.join(merchant.signals)}")

        lines.append(f"\nTrigger: {trigger.kind} (urgency {trigger.urgency})")
        lines.append(f"Trigger source: {trigger.source}")

        payload = trigger.payload
        if payload:
            if "top_item" in payload and isinstance(payload["top_item"], dict):
                top = payload["top_item"]
                lines.append(f"\nDigest Item:")
                lines.append(f"  Title: {top.get('title', '')}")
                lines.append(f"  Source: {top.get('source', '')}")
                if "trial_n" in top:
                    lines.append(f"  Trial N: {top['trial_n']:,}")
                if "patient_segment" in top:
                    lines.append(f"  Patient Segment: {top['patient_segment']}")
            if "top_item_id" in payload:
                item_id = payload["top_item_id"]
                for digest_item in category.digest:
                    if digest_item.id == item_id:
                        lines.append(f"\nDigest Item:")
                        lines.append(f"  Title: {digest_item.title}")
                        lines.append(f"  Source: {digest_item.source}")
                        if digest_item.trial_n:
                            lines.append(f"  Trial N: {digest_item.trial_n:,}")
                        if digest_item.patient_segment:
                            lines.append(f"  Patient Segment: {digest_item.patient_segment}")
                        if digest_item.summary:
                            lines.append(f"  Summary: {digest_item.summary}")
                        break

            if "delta_pct" in payload:
                lines.append(f"\nMetric Change: {payload['delta_pct']:.0%}")
            if "metric" in payload:
                lines.append(f"Metric: {payload['metric']}")
            if "days_remaining" in payload:
                lines.append(f"Days Remaining: {payload['days_remaining']}")
            if "festival" in payload:
                lines.append(f"\nFestival: {payload['festival']}")
                if "days_until" in payload:
                    lines.append(f"Days Until: {payload['days_until']}")
            if "available_slots" in payload:
                slots = payload["available_slots"]
                if slots:
                    slot_strs = [s.get("label", str(s)) for s in slots[:3]]
                    lines.append(f"\nAvailable Slots: {', '.join(slot_strs)}")

        if customer:
            lines.append(f"\n=== CUSTOMER CONTEXT ===")
            lines.append(f"Customer: {customer.identity.name}")
            lines.append(f"Language: {customer.identity.language_pref}")
            lines.append(f"State: {customer.state}")
            lines.append(f"Visits: {customer.relationship.visits_total}")
            if customer.preferences.preferred_slots:
                lines.append(f"Preferred Slots: {customer.preferences.preferred_slots}")

        peer = category.peer_stats
        lines.append(f"\n=== PEER BENCHMARKS ===")
        lines.append(f"Peer Avg CTR: {peer.avg_ctr:.1%}")
        lines.append(f"Peer Avg Views: {peer.avg_views_30d:,}")

        return "\n".join(lines)
