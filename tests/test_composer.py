"""
Deterministic unit tests for message composer with seed data
"""

import pytest

from app.engine.composer import MessageComposer
from app.engine.tone_adapter import ToneAdapter
from app.engine.grounding import GroundingValidator
from app.engine.conversation import ConversationEngine
from app.schemas import (
    CategoryContext,
    ConversationState,
    CustomerContext,
    MerchantContext,
    TriggerContext,
    VoiceProfile,
    MerchantIdentity,
    CustomerIdentity,
    PeerStats,
    DigestItem,
)


@pytest.fixture
def dentist_category():
    """Sample dentist category context"""
    return CategoryContext(
        slug="dentists",
        display_name="Dentists",
        voice=VoiceProfile(
            tone="peer_clinical",
            tone_register="respectful_collegial",
            vocab_allowed=["fluoride varnish", "scaling", "caries", "RCT"],
            vocab_taboo=["guaranteed", "100% safe", "miracle", "best in city"],
            salutation_examples=["Dr. {first_name}"],
        ),
        peer_stats=PeerStats(
            avg_rating=4.4,
            avg_ctr=0.030,
            avg_views_30d=1820,
        ),
        digest=[
            DigestItem(
                id="d_2026W17_jida_fluoride",
                kind="research",
                title="3-month fluoride varnish recall outperforms 6-month",
                source="JIDA Oct 2026, p.14",
                trial_n=2100,
                patient_segment="high_risk_adults",
                summary="Multi-center trial shows 38% lower caries recurrence.",
            )
        ],
    )


@pytest.fixture
def salon_category():
    """Sample salon category context"""
    return CategoryContext(
        slug="salons",
        display_name="Salons",
        voice=VoiceProfile(
            tone="warm_practical",
            tone_register="approachable_expert",
            vocab_allowed=["balayage", "keratin", "hair spa"],
            vocab_taboo=["guaranteed glow", "permanent results", "miracle"],
            salutation_examples=["Hi {first_name}"],
        ),
        peer_stats=PeerStats(avg_ctr=0.040),
    )


@pytest.fixture
def merchant_drmeera():
    """Sample dentist merchant"""
    return MerchantContext(
        merchant_id="m_001_drmeera_dentist_delhi",
        category_slug="dentists",
        identity=MerchantIdentity(
            name="Dr. Meera's Dental Clinic",
            city="Delhi",
            locality="Lajpat Nagar",
            owner_first_name="Meera",
            languages=["en", "hi"],
        ),
    )


@pytest.fixture
def merchant_studio11():
    """Sample salon merchant"""
    return MerchantContext(
        merchant_id="m_003_studio11_salon_hyderabad",
        category_slug="salons",
        identity=MerchantIdentity(
            name="Studio11 Family Salon",
            city="Hyderabad",
            locality="Kapra",
            owner_first_name="Lakshmi",
            languages=["en", "hi", "te"],
        ),
    )


@pytest.fixture
def research_digest_trigger():
    """Sample research digest trigger"""
    return TriggerContext(
        id="trg_001_research_digest_dentists",
        scope="merchant",
        kind="research_digest",
        source="external",
        merchant_id="m_001_drmeera_dentist_delhi",
        payload={"top_item_id": "d_2026W17_jida_fluoride"},
        urgency=2,
        suppression_key="research:dentists:2026-W17",
        expires_at="2026-05-03T00:00:00Z",
    )


@pytest.fixture
def recall_trigger():
    """Sample customer recall trigger"""
    return TriggerContext(
        id="trg_003_recall_due_priya",
        scope="customer",
        kind="recall_due",
        source="internal",
        merchant_id="m_001_drmeera_dentist_delhi",
        customer_id="c_001_priya",
        payload={
            "service_due": "6_month_cleaning",
            "available_slots": [
                {"iso": "2026-11-05T18:00:00+05:30", "label": "Wed 5 Nov, 6pm"},
                {"iso": "2026-11-06T17:00:00+05:30", "label": "Thu 6 Nov, 5pm"},
            ],
        },
        urgency=3,
        suppression_key="recall:c_001:6mo",
        expires_at="2026-11-30T00:00:00Z",
    )


@pytest.fixture
def customer_priya():
    """Sample customer context"""
    return CustomerContext(
        customer_id="c_001_priya",
        merchant_id="m_001_drmeera_dentist_delhi",
        identity=CustomerIdentity(
            name="Priya",
            language_pref="hi-en mix",
        ),
        state="lapsed_soft",
    )


class TestToneAdapter:
    """Test tone adapter functionality"""

    def test_get_salutation_dentist(self, dentist_category, merchant_drmeera):
        adapter = ToneAdapter()
        salutation = adapter.get_salutation(dentist_category, merchant_drmeera)
        assert "Meera" in salutation
        assert "Dr." in salutation

    def test_get_salutation_salon(self, salon_category, merchant_studio11):
        adapter = ToneAdapter()
        salutation = adapter.get_salutation(salon_category, merchant_studio11)
        assert "Lakshmi" in salutation
        assert "Hi" in salutation

    def test_check_taboos_dentist(self, dentist_category):
        adapter = ToneAdapter()

        violations = adapter.check_taboos(
            "Get guaranteed results with our treatment!",
            dentist_category
        )
        assert len(violations) > 0
        assert "guaranteed" in [v.lower() for v in violations]

    def test_check_taboos_clean_message(self, dentist_category):
        adapter = ToneAdapter()
        violations = adapter.check_taboos(
            "Consider a 3-month fluoride recall for high-risk patients.",
            dentist_category
        )
        assert len(violations) == 0

    def test_validate_voice_clean(self, dentist_category, merchant_drmeera):
        adapter = ToneAdapter()
        is_valid, issues = adapter.validate_voice(
            "Dr. Meera, JIDA's Oct issue has relevant findings for your practice.",
            dentist_category,
            merchant_drmeera
        )
        assert is_valid
        assert len(issues) == 0

    def test_validate_voice_with_taboo(self, dentist_category, merchant_drmeera):
        adapter = ToneAdapter()
        is_valid, issues = adapter.validate_voice(
            "Dr. Meera, our miracle treatment will guarantee results!",
            dentist_category,
            merchant_drmeera
        )
        assert not is_valid
        assert any("taboo" in issue.lower() for issue in issues)


class TestGroundingValidator:
    """Test grounding validation"""

    def test_extract_facts_from_trigger(self, research_digest_trigger):
        validator = GroundingValidator()
        facts = validator.extract_facts_from_trigger(research_digest_trigger)
        assert facts["kind"] == "research_digest"
        assert facts["urgency"] == 2

    def test_extract_facts_from_merchant(self, merchant_drmeera):
        validator = GroundingValidator()
        facts = validator.extract_facts_from_merchant(merchant_drmeera)
        assert facts["name"] == "Dr. Meera's Dental Clinic"
        assert facts["owner_first_name"] == "Meera"
        assert "hi" in facts["languages"]

    def test_build_grounded_context(
        self,
        dentist_category,
        merchant_drmeera,
        research_digest_trigger,
    ):
        validator = GroundingValidator()
        context = validator.build_grounded_context(
            dentist_category,
            merchant_drmeera,
            research_digest_trigger,
        )
        assert "Meera" in context
        assert "dentists" in context.lower()


class TestConversationEngine:
    """Test conversation handling"""

    def test_detect_auto_reply(self):
        engine = ConversationEngine()

        assert engine.detect_auto_reply(
            "Thank you for contacting us! Our team will respond shortly."
        )
        assert engine.detect_auto_reply(
            "I am an automated assistant and cannot help with this."
        )
        assert not engine.detect_auto_reply(
            "Yes, I'd like to proceed with that."
        )

    def test_detect_commitment(self):
        engine = ConversationEngine()

        assert engine.detect_commitment("Ok lets do it")
        assert engine.detect_commitment("Yes please proceed")
        assert engine.detect_commitment("Sure, go ahead")
        assert engine.detect_commitment("Haan chalega")
        assert not engine.detect_commitment("I need to think about this more")

    def test_detect_hostile(self):
        engine = ConversationEngine()

        assert engine.detect_hostile("Stop messaging me")
        assert engine.detect_hostile("This is spam, unsubscribe me")
        assert engine.detect_hostile("Leave me alone")
        assert not engine.detect_hostile("Tell me more about this")

    def test_detect_repeated_message(self):
        engine = ConversationEngine()
        state = ConversationState(
            conversation_id="test",
            merchant_id="m_001",
            turns=[
                {"from": "merchant", "message": "Thank you for contacting us!"},
                {"from": "vera", "message": "Hi there!"},
                {"from": "merchant", "message": "Thank you for contacting us!"},
            ],
        )
        count = engine.detect_repeated_message("Thank you for contacting us!", state)
        assert count == 2


class TestMessageComposer:
    """Test message composition"""

    @pytest.mark.asyncio
    async def test_compose_fallback_research_digest(
        self,
        dentist_category,
        merchant_drmeera,
        research_digest_trigger,
    ):
        composer = MessageComposer(llm_client=None)
        action = await composer.compose(
            dentist_category,
            merchant_drmeera,
            research_digest_trigger,
        )

        assert action.merchant_id == merchant_drmeera.merchant_id
        assert action.trigger_id == research_digest_trigger.id
        assert action.send_as == "vera"
        assert len(action.body) > 0

        body_lower = action.body.lower()
        assert "meera" in body_lower
        assert "jida" in body_lower.replace("jida", "jida") or "research" in body_lower

    @pytest.mark.asyncio
    async def test_compose_fallback_recall(
        self,
        dentist_category,
        merchant_drmeera,
        recall_trigger,
        customer_priya,
    ):
        composer = MessageComposer(llm_client=None)
        action = await composer.compose(
            dentist_category,
            merchant_drmeera,
            recall_trigger,
            customer_priya,
        )

        assert action.send_as == "merchant_on_behalf"
        assert action.customer_id == customer_priya.customer_id
        assert "priya" in action.body.lower()

    @pytest.mark.asyncio
    async def test_compose_generates_valid_cta(
        self,
        dentist_category,
        merchant_drmeera,
        research_digest_trigger,
    ):
        composer = MessageComposer(llm_client=None)
        action = await composer.compose(
            dentist_category,
            merchant_drmeera,
            research_digest_trigger,
        )
        assert action.cta in ["binary_yes_no", "open_ended", "slot_selection", "none"]

    @pytest.mark.asyncio
    async def test_compose_includes_suppression_key(
        self,
        dentist_category,
        merchant_drmeera,
        research_digest_trigger,
    ):
        composer = MessageComposer(llm_client=None)
        action = await composer.compose(
            dentist_category,
            merchant_drmeera,
            research_digest_trigger,
        )
        assert action.suppression_key == research_digest_trigger.suppression_key


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_compose_with_minimal_context(self):
        """Test composition with minimal context data"""
        composer = MessageComposer(llm_client=None)

        category = CategoryContext(
            slug="dentists",
            voice=VoiceProfile(
                tone="peer_clinical",
                tone_register="collegial",
            ),
        )
        merchant = MerchantContext(
            merchant_id="m_minimal",
            category_slug="dentists",
            identity=MerchantIdentity(name="Test Clinic"),
        )
        trigger = TriggerContext(
            id="trg_minimal",
            scope="merchant",
            kind="curious_ask_due",
            source="internal",
            merchant_id="m_minimal",
            payload={},
            urgency=1,
            suppression_key="test",
            expires_at="2026-12-31T00:00:00Z",
        )

        action = await composer.compose(category, merchant, trigger)
        assert action is not None
        assert len(action.body) > 0

    def test_conversation_handles_empty_state(self):
        """Test conversation engine with fresh state"""
        engine = ConversationEngine()
        state = ConversationState(
            conversation_id="new",
            merchant_id="m_test",
        )

        category = CategoryContext(
            slug="dentists",
            voice=VoiceProfile(tone="peer_clinical", tone_register="collegial"),
        )
        merchant = MerchantContext(
            merchant_id="m_test",
            category_slug="dentists",
            identity=MerchantIdentity(name="Test", owner_first_name="Test"),
        )

        response = engine.handle_reply(
            message="Yes, I'm interested",
            state=state,
            category=category,
            merchant=merchant,
        )
        assert response.action in ["send", "wait", "end"]
