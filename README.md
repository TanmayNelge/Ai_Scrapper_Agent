# AI-Powered Web Scraper v2.0
## LangGraph Architecture with Dual-LLM Pipeline

### What changed from v1

| Change | Why |
|--------|-----|
| **LangGraph state machine** replaces monolithic `engine.py` | Each step is a graph node — testable, retryable, observable |
| **Headful Playwright** (not headless) | Better anti-bot bypass on e-commerce sites |
| **Separate binary LLM** for YES/NO triage | Keeps main GPU free for extraction; faster decisions |
| **Table context fix** | Detects when table headers scroll out of view, injects them as context so LLM always knows column names |
| **Viewport-aware text extraction** | Only sends visible text to LLM (not full page every scroll) |
| **Watchdog** kills stalled operations | Auto-retries with backoff; abandons dead branches |
| **Priority queue** replaces DFS stack | URLs scored by LLM confidence; best pages crawled first |
| **Validator** between extraction and dedup | Catches junk (mostly-null items) before they hit the database |
| **Data pages get link extraction too** | Fixes the pagination bug where product listing links were missed |
| **Async seed generator** | No longer blocks the event loop |
| **Event bus with replay** | WebSocket reconnection catches up on missed events |

---

### Requirements

- **Python 3.11+**
- **Node.js** (for Playwright browser install)
- **LM Studio** running Qwen models
- **GPU**: NVIDIA 3050/4050 (4GB+ VRAM)


### FlowChart

![dd6ccc43-930b-470e-b8e3-862fa426dc3d](https://github.com/user-attachments/assets/ef07ff56-5ffc-49c9-8eb7-d393abf5e902)



### Setup

```bash
# 1. Install dependencies
cd ai_scraper
pip install -r requirements.txt

# 2. Install Playwright browsers
playwright install chromium

# 3. Start LM Studio with your models
#    Port 1234: Main model (Qwen 3.5 4B) — extraction + link scoring
#    Port 1235: Binary model (Qwen 0.5B or Phi-3-mini) — YES/NO only
#
#    OR use --same-llm flag to share one model on one port
```

### Running

```bash
# Default: headful browser, dual LLM (ports 1234 + 1235)
python run.py

# Single LLM mode (if you only have one model loaded)
python run.py --same-llm

# Headless + custom ports
python run.py --headless --llm-port 1234 --binary-port 1234 --same-llm

# Custom server port
python run.py --port 8080
```

**API docs**: http://localhost:8000/docs

### API Usage

```bash
# Create a project
curl -X POST http://localhost:8000/api/projects \
  -H "Content-Type: application/json" \
  -d '{
    "name": "GPU Prices",
    "query": "NVIDIA RTX 4090 GPU prices",
    "mode": "TEXT_ONLY",
    "schema_def": {
      "product_name": "string",
      "price": "number",
      "retailer": "string",
      "in_stock": "boolean"
    },
    "max_depth": 3
  }'

# Start crawling (project ID from response above)
curl -X POST http://localhost:8000/api/projects/1/start

# Check live stats
curl http://localhost:8000/api/projects/1/stats

# Get results
curl http://localhost:8000/api/projects/1/results

# Export CSV
curl http://localhost:8000/api/projects/1/export/csv -o results.csv

# Stop crawling
curl -X POST http://localhost:8000/api/projects/1/stop
```

### WebSocket Live Stream

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/1');

// Optional: replay missed events after reconnection
ws.onopen = () => {
  ws.send(JSON.stringify({ last_event_id: 0 }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // data.type: "log" | "data" | "status" | "error" | "progress" | "heartbeat"
  // data.payload: the event content
  // data.event_id: sequential ID for replay
  console.log(`[${data.type}] ${data.payload}`);
};
```

---

### Project Structure

```
ai_scraper/
├── run.py                          # Entry point
├── requirements.txt
└── backend/
    ├── main.py                     # FastAPI app + WebSocket
    ├── config.py                   # All settings in one place
    │
    ├── core/                       # The brain
    │   ├── graph.py                # LangGraph state machine definition
    │   ├── nodes.py                # Graph nodes (pick_url, navigate, parse, etc.)
    │   ├── state.py                # TypedDict state flowing through graph
    │   ├── orchestrator.py         # Top-level controller
    │   └── database.py             # SQLAlchemy models + SQLite-WAL
    │
    ├── browser/                    # The eyes & hands
    │   ├── browser_manager.py      # Headful Playwright + table context fix
    │   └── dom_parser.py           # Link extraction (without Readability)
    │
    ├── llm/                        # The brain's language
    │   ├── llm_gateway.py          # Main LLM: extraction + link scoring
    │   ├── binary_llm.py           # Binary LLM: YES/NO triage only
    │   └── prompt_registry.py      # All prompts versioned in one file
    │
    ├── pipeline/                   # Data processing
    │   ├── classifier.py           # Phase 2: page triage
    │   ├── extractor.py            # Phase 3: text + image extraction
    │   ├── validator.py            # Catches junk before dedup
    │   └── deduplicator.py         # SimHash + RapidFuzz (fixed)
    │
    ├── seeds/                      # URL discovery
    │   └── duckduckgo.py           # Async DDG seed generator
    │
    └── utils/                      # Shared utilities
        ├── event_bus.py            # Event streaming + replay
        ├── watchdog.py             # Stall detection + retry
        ├── url_frontier.py         # Priority queue + bloom filter
        ├── image_processor.py      # WhatsApp-style compression
        └── schema_builder.py       # Runtime Pydantic models
```

### How the Table Context Fix Works

When you scroll through a data table, the column headers disappear above the viewport.
The LLM then sees rows of data but doesn't know what each column represents.

**Before (broken)**:
```
LLM sees: "RTX 4090  |  $1,599  |  Yes  |  NVIDIA"
LLM thinks: What are these columns? Price? Name? Stock? 
```

**After (fixed)**:
```
=== TABLE COLUMN HEADERS (scrolled above, still apply) ===
Columns: Product Name | Price | In Stock | Manufacturer
=== DATA IN CURRENT VIEWPORT ===

RTX 4090  |  $1,599  |  Yes  |  NVIDIA
```

The browser's JavaScript TreeWalker detects when a `<thead>` or first `<tr>` 
has scrolled above the viewport while the table body is still visible, 
and injects the headers as context prefix.

### LM Studio Setup for Victus 3050/4050

**Option A: Dual model (recommended)**
1. Load Qwen-2.5-3B-Instruct on port 1234 (main extraction)
2. Load Qwen-2.5-0.5B-Instruct on port 1235 (binary triage)
3. Run: `python run.py`

**Option B: Single model**
1. Load Qwen-2.5-3B-Instruct on port 1234
2. Run: `python run.py --same-llm`

Both options work. Dual model is faster because triage calls 
don't compete with extraction calls for GPU time.
