# Acme E-Commerce AI Support Agent

An async FastAPI + LangGraph support agent for **Acme**, a fictional online store that sells books and consumer electronics. It answers three kinds of questions:

| Question type | Source | Tool |
|---|---|---|
| Products: description, price, stock, specs, recommendations | SQLite `products` table | `product_lookup` |
| Orders: status, items, totals, tracking, arrival date | SQLite `orders` + `order_items` tables | `track_order` |
| Policies: shipping, returns, refunds, warranty, Acme Plus, payments | Pinecone index built from a PDF | `knowledge_base_search` |

It hands off to a human when the customer is angry, disputes a charge, reports a defective item, or asks for a spec that isn't in the catalog.

## Architecture

```
POST /api/v1/chat ──► LangGraph (checkpointer, thread_id = session_id)

   START ─► router (LLM + bound tools)
              │
              ├─ tool calls ──────────────► tools ──► router   (loop)
              ├─ escalate_to_human call ──► human_escalation ─► END
              ├─ "I don't have that exact specification…" ─► human_escalation ─► END
              └─ final answer ────────────► END

data/checkpoints.db (SQLite)          Pinecone index "acme-knowledge-base"
├── checkpoints, writes   ← memory     └── chunks of knowledge_base/*.pdf
├── products              ← product_lookup
└── orders, order_items   ← track_order
```

| Module | Responsibility |
|---|---|
| `app/config.py` | `pydantic-settings` config (keys, provider, timeouts, Pinecone, SQLite paths) |
| `app/llm/factory.py` | **The only place an LLM or embedding client is built.** `get_chat_model()` / `get_embedding_model()` |
| `app/agent/state.py` | `AgentState`: `messages`, `needs_human_escalation`, `current_order_id`, `escalation_reason` |
| `app/agent/prompts.py` | System prompt, tool-selection rules, guardrails, the exact unverified-spec sentence |
| `app/agent/nodes.py` | `router_node`, `tool_executor_node`, `human_escalation_node`, `route_after_router` |
| `app/agent/graph.py` | `StateGraph` wiring and compilation |
| `app/agent/checkpointer.py` | Conversation memory backend chosen by `CHECKPOINTER` (`memory` or `sqlite`) |
| `app/db/database.py` | SQLite connection and schema for `products`, `orders`, `order_items` |
| `app/db/repository.py` | Time-boxed queries: `search_products()`, `get_order()` |
| `app/db/seed_data.py` | Demo catalog (8 electronics, 6 books) and 6 orders |
| `app/vectorstore.py` | Pinecone client, `ensure_index()` (create if missing, check dimension) |
| `app/tools/product_lookup.py` | `product_lookup` tool |
| `app/tools/order_lookup.py` | `track_order` tool (returns no customer name or email) |
| `app/tools/vector_search.py` | `knowledge_base_search` tool |
| `app/exceptions.py` | Typed errors mapped to HTTP 500 / 503 / 504 |
| `knowledge_base/acme_knowledge_base.pdf` | Demo policy document (7 pages, 15 sections) |
| `scripts/seed_db.py` | Creates the store tables and loads demo data (safe to re-run) |
| `scripts/ingest_pdf.py` | PDF → chunks → embeddings → Pinecone (creates the index if missing) |
| `scripts/build_kb_pdf.py` | Regenerates the demo PDF (needs `reportlab`) |

### Pluggable LLM layer

Nodes call `get_chat_model()` and never import a provider SDK. Switching providers only takes a `.env` change:

```env
LLM_PROVIDER=together      # default: nvidia/NVIDIA-Nemotron-3.5-Super-VL
LLM_PROVIDER=openai        # needs langchain-openai + OPENAI_API_KEY
LLM_PROVIDER=anthropic     # needs langchain-anthropic + ANTHROPIC_API_KEY
LLM_PROVIDER=self_hosted   # any OpenAI-compatible server: LLM_BASE_URL + LLM_MODEL
```

To add another provider, write a builder and call `register_chat_provider("name", builder)`. The factory checks at build time that the model supports `.bind_tools()`, so a misconfigured model fails at startup instead of during a conversation.

### Conversation memory

Each `session_id` is a LangGraph thread. The `CHECKPOINTER` setting chooses where threads are stored:

| `CHECKPOINTER` | Storage | Survives restart | Works across multiple servers |
|---|---|---|---|
| `memory` (default) | Process RAM (`MemorySaver`) | No | No |
| `sqlite` | `CHECKPOINT_SQLITE_PATH` (default `data/checkpoints.db`) via `AsyncSqliteSaver` | Yes | No (single file) |

The backend is opened when the app starts and closed when it shuts down. An unknown value stops the app at startup. To add MongoDB or Postgres, write an async context manager that yields a `BaseCheckpointSaver` and register it in `_BACKENDS` in `app/agent/checkpointer.py`.

### Embeddings (local BGE)

Embeddings run **locally** with a BGE model from your Hugging Face cache (`EMBEDDING_PROVIDER=huggingface`), so no API key is needed. The model is loaded once at startup, on Apple MPS, CUDA or CPU (whichever is available), and by default it never connects to the network.

| `EMBEDDING_MODEL` | `EMBEDDING_DIMENSION` | Notes |
|---|---|---|
| `BAAI/bge-small-en-v1.5` (default) | 384 | Fast, about 40 ms per query on Apple MPS |
| `BAAI/bge-large-en-v1.5` | 1024 | Better recall on large document sets, about 60 ms per query |

- Vectors are normalized and the index uses cosine similarity. Queries get BGE's retrieval instruction (`EMBEDDING_QUERY_INSTRUCTION`); documents don't.
- At startup the app checks that the model's real output size matches `EMBEDDING_DIMENSION`, and stops with an error if it doesn't.
- A Pinecone index has a fixed dimension. If you switch models, set a new `PINECONE_INDEX_NAME` and run the ingestion script again.
- To use a model that isn't cached yet, set `EMBEDDING_LOCAL_FILES_ONLY=false` for one run to download it.

### Store database (products and orders)

Products and orders live in the **same SQLite file as the checkpointer** (`data/checkpoints.db`), in their own tables. Set `STORE_SQLITE_PATH` to keep them in a separate file. The file uses WAL mode so the checkpointer can write while the tools read. Tables are created at startup. Load data with `python -m scripts.seed_db`.

- Prices are stored as integer cents. Product specs are a JSON `attributes` column, so books and electronics can have different fields.
- `product_lookup` ranks matches by SKU, then name, then author or brand, then description and specs. It also takes optional `category` and `max_price` filters.
- `track_order` accepts `ORD-10021`, `ord 10021` or `10021`, and never returns the customer's name or email to the model.
- Every query has a `STORE_QUERY_TIMEOUT_SECONDS` limit. If the database fails, the tool returns `STORE_UNAVAILABLE` and the agent escalates.

Demo data: order IDs `ORD-10021` to `ORD-10026` (Shipped, Processing, Delivered, Cancelled, Out for delivery, pre-order). Three products are missing a spec on purpose, to demo the escalation:

| Product | Missing spec |
|---|---|
| Nimbus Buds Pro (`ACME-EL-1004`) | Water-resistance rating |
| Acme Volt 65W Charger (`ACME-EL-1007`) | Weight |
| The Quiet Algorithm (`ACME-BK-2003`) | Page count |

### Knowledge base (Pinecone)

`knowledge_base/acme_knowledge_base.pdf` is a fictional help center modeled on a large online retailer. It covers the company overview, Acme Plus membership, shipping speeds and costs, order changes, returns and refunds, damaged items, warranty and Acme Protect, price adjustments, the marketplace guarantee, payments, books, electronics, account security, contacting support, and FAQs.

The Pinecone index (`PINECONE_INDEX_NAME`, default `acme-knowledge-base`) is **created automatically** as a serverless cosine index with `EMBEDDING_DIMENSION` dimensions (384 for the default model). This happens both when you run the ingestion script and when the app starts. If the index already exists, its dimension is checked against the embedding model and a mismatch fails with a clear error.

> **Use a standard index, not an integrated-embedding one.** If you create an index in the Pinecone console with a hosted embedding model (such as `llama-text-embed-v2`, with fields like `chunk_text`), Pinecone embeds the text itself and only accepts writes through its documents API. It rejects this app's BGE vectors with `400 ... writes must go through the documents API`. `ensure_index()` detects this and stops with a clear message. The fix is to set `PINECONE_INDEX_NAME` to a name that doesn't exist yet, and the app creates a standard index for you.

## Setup

```bash
cd Section_07_AI_Agents/capstone/ecommerce_agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in TOGETHER_API_KEY (chat) and PINECONE_API_KEY

python -m scripts.seed_db                                            # products + orders -> SQLite
python -m scripts.ingest_pdf knowledge_base/acme_knowledge_base.pdf  # PDF -> Pinecone

uvicorn app.main:app --reload
```

> If `pinecone-client` is already installed, run `pip uninstall pinecone-client` before installing `pinecone`. The two packages both provide the `pinecone` module.

### Ingesting a PDF

```bash
python -m scripts.ingest_pdf <path/to/file.pdf> [--chunk-size 1000] [--chunk-overlap 150] [--dry-run]
```

1. Each page's text is extracted with `pypdf`. Lines repeated on most pages (running headers and footers) are removed.
2. The text is split with `RecursiveCharacterTextSplitter`, using a chunk size of 1000 characters and an overlap of 150 by default.
3. Chunks are embedded in batches with the local BGE model.
4. The index is created if missing. Chunks from an earlier run of the same file are removed, and the new chunks are upserted with IDs like `acme-knowledge-base-p003-c001`. The metadata holds the text, source file name, page and chunk number.

`--dry-run` shows the chunks without calling Pinecone or the embedding API. Old chunks are deleted only after embedding succeeds, so a failed run leaves the existing knowledge base in place. You can ingest several PDFs into the same index, because chunk IDs are prefixed by file name. Scanned PDFs need OCR first.

## Try it

```bash
chat() { curl -s localhost:8000/api/v1/chat -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$1\",\"message\":\"$2\"}"; echo; }

chat demo-1 "What's the battery life and price of the Paperlight e-reader?"   # product_lookup
chat demo-1 "Any Python books under \$50?"                                     # product_lookup + max_price
chat demo-1 "Are the Nimbus Buds Pro waterproof?"                              # missing spec -> escalation
chat demo-2 "Where is my order 10025?"                                         # track_order
chat demo-2 "Can I return it if I open the tablet?"                            # knowledge_base_search
chat demo-3 "How much is Acme Plus and what does it include?"                  # knowledge_base_search
chat demo-4 "My earbuds arrived broken and I want a refund now!"               # escalate_to_human
```

Response:

```json
{"session_id": "demo-2", "message": "...", "needs_human_escalation": false, "current_order_id": "ORD-10025"}
```

## Logging

Logs go to the console (stdout). Each line is tagged with the request ID and session ID, so you can follow one chat turn from the customer's message to the final answer, even when several requests run at the same time:

```
[req=c1dfdea3 sid=demo-1] app.main: [chat] user message: 'Where is my order 10025, and is return shipping free ...?'
[req=c1dfdea3 sid=demo-1] app.agent.nodes: [router] LLM call #1: 2 messages (system + 1 of 1 in history), active order=none
[req=c1dfdea3 sid=demo-1] app.agent.nodes: [router] LLM call #1 replied in 2991 ms (tokens in=1648 out=204) -> tool calls: track_order(...), knowledge_base_search(...)
[req=c1dfdea3 sid=demo-1] app.agent.nodes: [route] router -> tools (2 tool call(s) requested)
[req=c1dfdea3 sid=demo-1] app.agent.nodes: [tools] -> track_order(order_id='10025')
[req=c1dfdea3 sid=demo-1] app.tools.order_lookup: [track_order] ORD-10025 found: status=Out for delivery, 2 item(s)
[req=c1dfdea3 sid=demo-1] app.tools.vector_search: [kb] embed 322 ms, Pinecone 1337 ms: 4 matches [p6=0.66, ...], 4 kept (min_score=0.50)
[req=c1dfdea3 sid=demo-1] app.agent.nodes: [tools] active order set to ORD-10025
[req=c1dfdea3 sid=demo-1] app.agent.nodes: [router] LLM call #2 replied in 1651 ms (...) -> answer: 'Your order ORD-10025 is out for delivery ...'
[req=c1dfdea3 sid=demo-1] app.agent.nodes: [route] router -> END (final answer)
[req=c1dfdea3 sid=demo-1] app.main: [chat] done in 6322 ms: llm_calls=2 tools=['track_order', 'knowledge_base_search'] escalated=False order=ORD-10025
```

| Tag | What it shows |
|---|---|
| `[chat]` | The customer message, then a summary of the turn (time, LLM calls, tools, escalation) and the final answer |
| `[router]` | Each LLM call: prompt size, time taken, tokens, and its decision (which tool calls, or the answer) |
| `[route]` | Which node runs next, and why |
| `[tools]` | Each tool call with its arguments, each result (ok or failed), and changes to the active order |
| `[product_lookup]` / `[track_order]` / `[kb]` | Inside the tools: matched SKUs, order found or not, and retrieval timings, pages and scores |
| `[escalation]` | Hand-off reason and summary (logged as WARNING) |

| Setting | Default | Effect |
|---|---|---|
| `LOG_LEVEL` | `INFO` | `DEBUG` for more detail, `WARNING` for escalations and errors only |
| `LOG_MESSAGE_CONTENT` | `true` | `false` replaces message, answer and tool text with its length (`<87 chars>`) so logs hold no customer data |
| `LOG_PREVIEW_CHARS` | `200` | Maximum characters of text shown per log line |

Every response includes an `X-Request-ID` header. Search the logs for it to find that request. If a proxy sends an `X-Request-ID`, that value is used instead.

## Error handling

| Failure | Behavior |
|---|---|
| LLM call exceeds `LLM_TIMEOUT_SECONDS` | `504 llm_timeout` + `Retry-After` |
| LLM provider error (5xx, auth, network) | `503 llm_unavailable` + `Retry-After`. The SDK retries `LLM_MAX_RETRIES` times first |
| Whole request exceeds `REQUEST_TIMEOUT_SECONDS` | `504 llm_timeout` |
| SQLite query fails or is slow | The tool returns `STORE_UNAVAILABLE`, the model apologizes and escalates. The API does not fail |
| Pinecone or embeddings down or slow | The tool returns `KB_UNAVAILABLE`, the model apologizes and escalates. The API does not fail |
| Pinecone index check fails at startup | Logged. The app still starts and the knowledge base tool reports `KB_UNAVAILABLE` |
| Unexpected tool exception or bad tool args | `ToolNode` converts it to a `TOOL_ERROR` tool message and the model does not guess |
| Agent loops past `GRAPH_RECURSION_LIMIT` | `200` with a human hand-off and `needs_human_escalation: true` |
| Bad request body | `422` (Pydantic validation) |
| Misconfiguration (unknown provider, missing key) | `500 llm_misconfigured` |

Clients only ever see generic `detail` messages. Full error details go to the logs.

If a turn is interrupted (timeout or loop limit), it can leave an unanswered tool call in the session history. The router repairs this before the next LLM call, so the session keeps working.

## Production notes

- Use `CHECKPOINTER=sqlite` for a single server. For several replicas, add a shared backend such as Postgres (`langgraph-checkpoint-postgres`) or MongoDB (`langgraph-checkpoint-mongodb`) in `app/agent/checkpointer.py`, and move the store tables to the same database.
- `track_order` gives order details to anyone who knows the order ID. Before going live, verify the customer, for example by matching `customer_email` (already stored) or using an authenticated user ID.
- `human_escalation_node` marks where to create a helpdesk ticket or push the session to a live-agent queue.
- Concurrent requests for the same `session_id` are not serialized. Add a per-session lock or queue if your client can send them.



The model calls the escalate_to_human tool for any of these. The tool is never actually run; the router just uses it as a signal.

┌───────────────────────┬──────────────────────────────────────────────────────────────────────────┬─────────────────────┐
│        Trigger        │                             Example message                              │  escalation_reason  │
├───────────────────────┼──────────────────────────────────────────────────────────────────────────┼─────────────────────┤
│ Damaged or defective  │ "My tablet arrived cracked"                                              │ defective_item      │
│ item                  │                                                                          │                     │
├───────────────────────┼──────────────────────────────────────────────────────────────────────────┼─────────────────────┤
│ Charge or refund      │ "I was charged twice, I'm filing a chargeback"                           │ transaction_dispute │
│ dispute               │                                                                          │                     │
├───────────────────────┼──────────────────────────────────────────────────────────────────────────┼─────────────────────┤
│ Extreme anger         │ "This is the worst service ever, I'm done with you people!"              │ extreme_anger       │
├───────────────────────┼──────────────────────────────────────────────────────────────────────────┼─────────────────────┤
│ Asks for a person     │ "Let me talk to a real human"                                            │ requested_human     │
├───────────────────────┼──────────────────────────────────────────────────────────────────────────┼─────────────────────┤
│ A backend is down     │ Any question where SQLite returns STORE_UNAVAILABLE or Pinecone returns  │ system_issue        │
│                       │ KB_UNAVAILABLE                                                           │                     │
└───────────────────────┴──────────────────────────────────────────────────────────────────────────┴─────────────────────┘

The reply is the standard hand-off message: "I'm connecting you with a member of our support team now…"

2. A product spec is missing from the catalog

The product exists, but the attribute the customer asked about isn't in its data. The model must reply with the fixed sentence "I don't have that exact specification verified in our catalog. Let me connect you with a team member." The router spots that sentence and escalates. Three demo products are set up for this:

┌──────────────────────────────────────────────────┬─────────────────────────┐
│                       Ask                        │      Missing spec       │
├──────────────────────────────────────────────────┼─────────────────────────┤
│ "Are the Nimbus Buds Pro waterproof?"            │ Water-resistance rating │
├──────────────────────────────────────────────────┼─────────────────────────┤
│ "How much does the Acme Volt 65W charger weigh?" │ Weight                  │
├──────────────────────────────────────────────────┼─────────────────────────┤
│ "How many pages is The Quiet Algorithm?"         │ Page count              │
└──────────────────────────────────────────────────┴─────────────────────────┘
