# Vera Engine

Production-grade FastAPI implementation of magicpin's Merchant AI Assistant ("Vera").

## Architecture

Vera uses the **4-Context Framework** to compose personalized WhatsApp messages:

1. **CategoryContext** - Industry knowledge (voice, offers, benchmarks, digest)
2. **MerchantContext** - Business state (identity, performance, offers, history)
3. **TriggerContext** - Event prompting outreach (research digest, performance dip, recall)
4. **CustomerContext** - Optional customer data for merchant-on-behalf messages

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:

```env
# LLM Configuration (choose one provider)
LLM_PROVIDER=openai
OPENAI_API_KEY=your-api-key-here

# Alternative providers
# LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY=your-key
# LLM_PROVIDER=groq
# GROQ_API_KEY=your-key

# Optional: Override model
# LLM_MODEL=gpt-4o
```

### 3. Expand Dataset

```bash
cd dataset
python generate_dataset.py --out ../expanded
```

### 4. Start the Engine

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

### 5. Run Judge Simulator

Open `judge_simulator.py`, configure your LLM provider and API key, then:

```bash
python judge_simulator.py
```

## API Endpoints

All endpoints available under both `/` and `/v1/` prefixes.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/healthz` | GET | Liveness probe |
| `/v1/metadata` | GET | Bot configuration |
| `/v1/context` | POST | Ingest context (category/merchant/customer/trigger) |
| `/v1/tick` | POST | Evaluate triggers, compose messages |
| `/v1/reply` | POST | Handle inbound merchant/customer replies |

## Scoring Dimensions

The engine optimizes for all 5 judge scoring dimensions:

| Dimension | Strategy |
|-----------|----------|
| **Specificity** | Uses exact numbers, dates, source citations from context |
| **Category Fit** | Applies category-specific voice rules and taboo checking |
| **Merchant Fit** | Personalizes with owner name, language, active offers |
| **Trigger Relevance** | States "why now" in the first sentence |
| **Engagement Compulsion** | Single clear CTA with loss aversion/curiosity hooks |

## Key Features

### Zero-Hallucination Guarantee

- All facts validated against provided context
- Prices cross-checked against merchant offers and category catalog
- Source citations verified against digest items

### Multi-Turn Conversation Handling

- **Auto-reply detection**: Recognizes canned responses, backs off appropriately
- **Intent transition**: Stops qualifying when merchant commits
- **Hostile handling**: Graceful exit on opt-out signals
- **Anti-repetition**: Never sends the same message twice

### Category Voice Rules

| Category | Tone | Taboos |
|----------|------|--------|
| Dentists | Peer-clinical | "guaranteed", "miracle", "best in city" |
| Salons | Warm-practical | "guaranteed glow", "permanent results" |
| Restaurants | Fellow-operator | "best food in city", "miracle marketing" |
| Gyms | Coach-energetic | "guaranteed weight loss", "shred in 7 days" |
| Pharmacies | Trustworthy-precise | "miracle cure", "100% safe" |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_endpoints.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

## Docker

```bash
# Build
docker build -t vera-engine .

# Run
docker run -p 8080:8080 -e OPENAI_API_KEY=your-key vera-engine
```

## Project Structure

```
vera-engine/
├── app/
│   ├── main.py          # FastAPI server with dual routing
│   ├── schemas.py       # Pydantic v2 models
│   ├── state.py         # Thread-safe stores
│   └── engine/
│       ├── composer.py      # Message composition
│       ├── conversation.py  # Multi-turn handling
│       ├── tone_adapter.py  # Voice/taboo rules
│       ├── grounding.py     # Hallucination prevention
│       └── llm_client.py    # LLM connector
├── tests/
│   ├── test_endpoints.py    # API contract tests
│   └── test_composer.py     # Unit tests
├── requirements.txt
├── Dockerfile
└── README.md
```

## Approach

1. **Deterministic Fallback**: Every trigger kind has a handcrafted template that works without LLM
2. **LLM Enhancement**: When configured, LLM improves fluency while respecting grounding constraints
3. **Validation Layer**: All outputs checked for taboos, repetition, and grounding before sending
4. **State Machine**: Conversation state tracks auto-replies, commitments, and sent messages

## What Would Help Most

- More diverse trigger examples with expected outputs
- Real merchant conversation logs for fine-tuning intent detection
- A/B test data on which CTAs drive highest response rates

---

Built for the magicpin AI Challenge 2026.
