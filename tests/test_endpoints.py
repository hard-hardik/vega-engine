"""
Contract validation tests against judge HTTP calls
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.state import context_store, conversation_store, suppression_store


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_stores():
    """Reset stores before each test"""
    context_store._categories.clear()
    context_store._merchants.clear()
    context_store._customers.clear()
    context_store._triggers.clear()
    conversation_store._conversations.clear()
    conversation_store._sent_messages.clear()
    suppression_store._suppressed.clear()
    yield


class TestHealthEndpoint:
    """Test /v1/healthz endpoint"""

    def test_healthz_returns_ok(self, client):
        response = client.get("/v1/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "uptime_seconds" in data
        assert "contexts_loaded" in data

    def test_healthz_root_path(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200

    def test_healthz_context_counts(self, client):
        response = client.get("/v1/healthz")
        data = response.json()
        counts = data["contexts_loaded"]
        assert "category" in counts
        assert "merchant" in counts
        assert "customer" in counts
        assert "trigger" in counts


class TestMetadataEndpoint:
    """Test /v1/metadata endpoint"""

    def test_metadata_returns_required_fields(self, client):
        response = client.get("/v1/metadata")
        assert response.status_code == 200
        data = response.json()
        assert "team_name" in data
        assert "team_members" in data
        assert "model" in data
        assert "approach" in data
        assert "version" in data

    def test_metadata_root_path(self, client):
        response = client.get("/metadata")
        assert response.status_code == 200


class TestContextPushEndpoint:
    """Test /v1/context endpoint"""

    def test_push_category_context(self, client):
        payload = {
            "scope": "category",
            "context_id": "dentists",
            "version": 1,
            "payload": {
                "slug": "dentists",
                "display_name": "Dentists",
                "voice": {
                    "tone": "peer_clinical",
                    "tone_register": "respectful_collegial",
                },
            },
            "delivered_at": "2026-04-26T09:45:00Z",
        }
        response = client.post("/v1/context", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["accepted"] is True
        assert "ack_id" in data
        assert "stored_at" in data

    def test_push_merchant_context(self, client):
        payload = {
            "scope": "merchant",
            "context_id": "m_001_drmeera",
            "version": 1,
            "payload": {
                "merchant_id": "m_001_drmeera",
                "category_slug": "dentists",
                "identity": {
                    "name": "Dr. Meera's Dental Clinic",
                    "city": "Delhi",
                    "locality": "Lajpat Nagar",
                    "owner_first_name": "Meera",
                    "languages": ["en", "hi"],
                },
            },
        }
        response = client.post("/v1/context", json=payload)
        assert response.status_code == 200
        assert response.json()["accepted"] is True

    def test_push_stale_version_rejected(self, client):
        payload = {
            "scope": "category",
            "context_id": "dentists",
            "version": 2,
            "payload": {
                "slug": "dentists",
                "voice": {"tone": "peer_clinical", "tone_register": "collegial"},
            },
        }
        client.post("/v1/context", json=payload)

        payload["version"] = 1
        response = client.post("/v1/context", json=payload)
        assert response.status_code == 409
        data = response.json()
        assert data["accepted"] is False
        assert data["reason"] == "stale_version"
        assert data["current_version"] == 2

    def test_push_trigger_context(self, client):
        payload = {
            "scope": "trigger",
            "context_id": "trg_001",
            "version": 1,
            "payload": {
                "id": "trg_001",
                "scope": "merchant",
                "kind": "research_digest",
                "source": "external",
                "merchant_id": "m_001",
                "payload": {},
                "urgency": 2,
                "suppression_key": "research:2026-W17",
                "expires_at": "2026-05-03T00:00:00Z",
            },
        }
        response = client.post("/v1/context", json=payload)
        assert response.status_code == 200


class TestTickEndpoint:
    """Test /v1/tick endpoint"""

    def test_tick_empty_triggers(self, client):
        payload = {
            "now": "2026-04-26T10:35:00Z",
            "available_triggers": [],
        }
        response = client.post("/v1/tick", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["actions"] == []

    def test_tick_missing_trigger(self, client):
        payload = {
            "now": "2026-04-26T10:35:00Z",
            "available_triggers": ["trg_nonexistent"],
        }
        response = client.post("/v1/tick", json=payload)
        assert response.status_code == 200
        assert response.json()["actions"] == []

    def test_tick_with_valid_context(self, client):
        client.post("/v1/context", json={
            "scope": "category",
            "context_id": "dentists",
            "version": 1,
            "payload": {
                "slug": "dentists",
                "voice": {"tone": "peer_clinical", "tone_register": "collegial"},
                "peer_stats": {},
                "digest": [{
                    "id": "d_2026W17_jida_fluoride",
                    "kind": "research",
                    "title": "3-month fluoride recall cuts caries 38%",
                    "source": "JIDA Oct 2026, p.14",
                    "trial_n": 2100,
                }],
            },
        })

        client.post("/v1/context", json={
            "scope": "merchant",
            "context_id": "m_001",
            "version": 1,
            "payload": {
                "merchant_id": "m_001",
                "category_slug": "dentists",
                "identity": {
                    "name": "Dr. Meera's Dental",
                    "owner_first_name": "Meera",
                    "languages": ["en", "hi"],
                },
            },
        })

        client.post("/v1/context", json={
            "scope": "trigger",
            "context_id": "trg_001",
            "version": 1,
            "payload": {
                "id": "trg_001",
                "scope": "merchant",
                "kind": "research_digest",
                "source": "external",
                "merchant_id": "m_001",
                "payload": {"top_item_id": "d_2026W17_jida_fluoride"},
                "urgency": 2,
                "suppression_key": "research:dentists:2026-W17",
                "expires_at": "2026-05-03T00:00:00Z",
            },
        })

        response = client.post("/v1/tick", json={
            "now": "2026-04-26T10:35:00Z",
            "available_triggers": ["trg_001"],
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["actions"]) == 1

        action = data["actions"][0]
        assert action["merchant_id"] == "m_001"
        assert action["trigger_id"] == "trg_001"
        assert "body" in action
        assert len(action["body"]) > 0


class TestReplyEndpoint:
    """Test /v1/reply endpoint"""

    def test_reply_auto_reply_detection(self, client):
        client.post("/v1/context", json={
            "scope": "merchant",
            "context_id": "m_001",
            "version": 1,
            "payload": {
                "merchant_id": "m_001",
                "category_slug": "dentists",
                "identity": {"name": "Test", "owner_first_name": "Test"},
            },
        })
        client.post("/v1/context", json={
            "scope": "category",
            "context_id": "dentists",
            "version": 1,
            "payload": {
                "slug": "dentists",
                "voice": {"tone": "peer_clinical", "tone_register": "collegial"},
            },
        })

        response = client.post("/v1/reply", json={
            "conversation_id": "conv_test",
            "merchant_id": "m_001",
            "from_role": "merchant",
            "message": "Thank you for contacting us! Our team will respond shortly.",
            "turn_number": 1,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "send"

    def test_reply_hostile_detection(self, client):
        client.post("/v1/context", json={
            "scope": "merchant",
            "context_id": "m_001",
            "version": 1,
            "payload": {
                "merchant_id": "m_001",
                "category_slug": "dentists",
                "identity": {"name": "Test", "owner_first_name": "Test"},
            },
        })
        client.post("/v1/context", json={
            "scope": "category",
            "context_id": "dentists",
            "version": 1,
            "payload": {
                "slug": "dentists",
                "voice": {"tone": "peer_clinical", "tone_register": "collegial"},
            },
        })

        response = client.post("/v1/reply", json={
            "conversation_id": "conv_test2",
            "merchant_id": "m_001",
            "from_role": "merchant",
            "message": "Stop messaging me. This is spam.",
            "turn_number": 1,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "end"

    def test_reply_commitment_detection(self, client):
        client.post("/v1/context", json={
            "scope": "merchant",
            "context_id": "m_001",
            "version": 1,
            "payload": {
                "merchant_id": "m_001",
                "category_slug": "dentists",
                "identity": {"name": "Test", "owner_first_name": "Test"},
            },
        })
        client.post("/v1/context", json={
            "scope": "category",
            "context_id": "dentists",
            "version": 1,
            "payload": {
                "slug": "dentists",
                "voice": {"tone": "peer_clinical", "tone_register": "collegial"},
            },
        })

        response = client.post("/v1/reply", json={
            "conversation_id": "conv_test3",
            "merchant_id": "m_001",
            "from_role": "merchant",
            "message": "Ok lets do it. Whats next?",
            "turn_number": 2,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "send"
        body = data.get("body", "").lower()
        qualifying_words = ["would you", "do you", "can you tell"]
        assert not any(w in body for w in qualifying_words)
