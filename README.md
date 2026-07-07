# JobFit-AI

JobFit-AI is an end-to-end job-application system: it scrapes LinkedIn job postings, extracts structured requirements with a local LLM, matches jobs to a resume with semantic vector search, and then hands the shortlist to a **LangGraph agent** that decides which jobs are worth pursuing and prepares tailored application materials for each one.

The project is built in two layers with deliberately different designs:

- **A deterministic data pipeline** (scrape → extract → index) — batch jobs where control flow should be explicit and repeatable, so no LLM decides anything.
- **An agentic layer** (LangGraph) — where judgment is actually needed: *which* of the matched jobs to pursue, and what tailored materials to produce. The LLM makes those calls through typed tools; everything it does is written back into typed graph state.

## Architecture

### Offline ingestion pipeline (run periodically)

```
LinkedIn (Voyager API)
      │  app/ingestion/fetch_jobs.py
      ▼
  SQLite (scraped_jobs)
      │
      ├── app/ingestion/extract_metadata_batch.py
      │      └─ Ollama (qwen2.5:7b) → validated JobMetadata → job_metadata table
      │
      └── app/ingestion/build_index.py
             └─ sentence-transformers (all-MiniLM-L6-v2) → FAISS index
                (vector position stored back on each scraped_jobs row)
```

### Agent graph (run per resume)

```
        START
          │
          ▼
     ┌─────────┐   error   ┌─────┐
     │find_jobs├──────────►│ END │
     └────┬────┘           └─────┘
          │ matches
          ▼
 ┌──────────────────┐
 │ extract_metadata │   (fills gaps only — persisted metadata rides along)
 └────────┬─────────┘
          │
          ▼
     ┌─────────┐  tool calls   ┌───────┐
     │  agent  ├──────────────►│ tools │   tailor_resume
     │ (Groq)  │◄──────────────┤       │   write_cover_letter
     └────┬────┘               └───────┘   mark_applied
          │ no tool calls
          ▼
         END
```

- `find_jobs` / `extract_metadata` are **deterministic nodes**: embed the resume, search FAISS, apply compatibility filters, enrich with structured requirements. A conditional edge routes to END if matching fails (`state["error"]`).
- The **agent node** sees the enriched shortlist and loops with a `ToolNode` until it stops emitting tool calls. Tools receive graph state via `InjectedState` and write results back through `Command` updates — the LLM only ever supplies a `job_id`.
- `mark_applied` only updates tracking state; nothing is ever submitted externally without a human in the loop.

## Project structure

```
app/
├── config.py                  # paths + model names (env-overridable), single source of truth
├── schemas/
│   └── job_metadata.py        # Pydantic schema the extraction LLM must satisfy
├── db/
│   └── repository.py          # all SQLite access; schema defined once in init_db()
├── embeddings/
│   ├── encoder.py             # lazy-loaded sentence-transformers model
│   └── index_store.py         # FAISS index load/save
├── ingestion/                 # offline batch jobs
│   ├── fetch_jobs.py          # LinkedIn scraping (ids + full details)
│   ├── metadata_extractor.py  # Ollama extraction w/ validation-feedback retries
│   ├── extract_metadata_batch.py
│   ├── build_index.py         # embed new jobs into FAISS
│   └── helpers.py             # auth, text cleaning, derived fields
├── matching/
│   └── matcher.py             # query-time: embed resume → search → filter → rank
├── agent/                     # LangGraph agentic layer
│   ├── state.py               # domain schemas + AgentState (TypedDict w/ reducers)
│   ├── prompts.py             # all prompt text in one place
│   ├── tools.py               # pipeline functions + LLM-callable action tools
│   ├── nodes.py               # thin state-marshalling adapters over tools
│   ├── llm.py                 # Groq chat-model factory (dependency-injected)
│   ├── graph.py               # graph wiring only
│   └── run.py                 # CLI entrypoint
└── llm/prompts/               # extraction prompt template
tests/                         # pure-logic + graph-wiring tests (no keys/network needed)
```

## Setup

Prerequisites: Python 3.10+, Chrome + ChromeDriver (scraping), [Ollama](https://ollama.com/) with `qwen2.5:7b` pulled (metadata extraction), and a [Groq](https://groq.com/) API key (agent).

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # add -r requirements-dev.txt for tests
```

Configuration is environment-driven (see `app/config.py`):

| Variable | Purpose | Default |
|---|---|---|
| `GROQ_API_KEY` | Agent reasoning model (required for the agent) | — |
| `GROQ_MODEL` | Override the Groq model | `llama-3.3-70b-versatile` |
| `OLLAMA_MODEL` | Local extraction model | `qwen2.5:7b` |
| `EMBEDDING_MODEL` | sentence-transformers model | `all-MiniLM-L6-v2` |
| `LINKEDIN_EMAIL` / `LINKEDIN_PASSWORD` | Scraper login | falls back to gitignored `logins.csv` |

## Running the pipeline

Each stage is an idempotent batch job — re-running only processes what is new or previously failed:

```bash
# 1. Discover job ids and fetch full details into SQLite
python -m app.ingestion.fetch_jobs

# 2. Extract structured requirements for unprocessed jobs (Ollama)
python -m app.ingestion.extract_metadata_batch

# 3. Embed new job descriptions into the FAISS index
python -m app.ingestion.build_index
```

## Running the agent

```bash
export GROQ_API_KEY=gsk_...
python -m app.agent.run path/to/resume.txt --top-k 5 --experience-years 3
```

The agent prints a summary of which jobs it prepared (and why) and writes tailored resumes and cover letters to `outputs/<job_id>/`.

## Design decisions

- **Deterministic spine, agentic head.** Matching and enrichment are plain graph nodes with explicit conditional edges; the LLM loop only starts once there is a concrete, typed shortlist to reason over. Agents should decide things that need judgment — not run ETL.
- **Typed state everywhere.** `AgentState` is a TypedDict whose fields are Pydantic domain models (`CandidateProfile`, `JobMatch`, `TailoredArtifacts`, `ApplicationRecord`). Only `messages` uses an appending reducer; every other field is overwritten wholesale by the node that owns it.
- **Tools write state via `Command`, not prose.** Action tools receive `InjectedState`, look up the job by id, and return a `Command` that updates typed fields (`artifacts`, `applications`) plus a `ToolMessage`. Results are never smuggled through free-text messages.
- **Dependency-injected model.** `build_graph(model=...)` accepts any chat model, so the whole graph is testable with a fake — no API key needed (see `tests/test_agent_graph.py`).
- **Schema-validated extraction with error feedback.** The extraction prompt embeds the Pydantic JSON schema; invalid responses are retried with the specific validation error appended, which fixes most malformed outputs within a retry or two.
- **Metadata is extracted once, then reused.** Batch extraction persists requirements to SQLite; at query time the matcher rehydrates them, and the agent's enrichment node only calls the LLM for jobs with gaps.
- **Failures are data, not crashes.** Node failures land in `state["error"]` and route through a conditional edge; per-job failures in batch jobs are collected and reported, leaving the job eligible for retry on the next run.

## Testing

```bash
python -m pytest tests/
```

The suite covers the metadata compatibility filters, metadata rehydration from DB rows, prompt rendering, and full graph wiring (happy path, error routing, enrichment skipping) using an injected fake model — no network, database, or API keys required.

## Extraction accuracy evaluation

To validate the LLM-based metadata extraction layer, 50 job postings were manually labeled and compared against the extracted output across the three fields used for hard filtering.

| Field                   | Correct | Wrong | Accuracy |
|-------------------------|---------|-------|----------|
| `min_experience_years`  | 44 / 50 | 6     | 88.0%    |
| `only_english_required` | 48 / 50 | 2     | 96.0%    |
| `higher_education_req`  | 48 / 50 | 2     | 96.0%    |
| **Overall**             | **140 / 150** | **10** | **93.3%** |

`only_english_required` and `higher_education_req` are reliable enough for hard filtering. `min_experience_years` at 88% reflects genuine ambiguity in how experience is stated ("senior level", "3–5 years", "some experience preferred") — which is why unknown values are never treated as disqualifying.

## Limitations & future work

- **Human-in-the-loop approval** via LangGraph `interrupt` before `mark_applied`, plus a checkpointer for resumable multi-turn sessions.
- **Two-tower retrieval**: separate resume/job encoders trained on application feedback, replacing the single off-the-shelf embedding model.
- **Re-ranking** with explainable signals (skill overlap, recency, seniority match) on top of cosine similarity.
- **Resume parsing** to derive `experience_years` and skills from the resume instead of CLI flags.
- Scraping depends on LinkedIn's internal API and cookies; it is best-effort and for personal use.
