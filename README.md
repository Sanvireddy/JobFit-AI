# JobFit-AI

JobFit-AI is an end-to-end job-application system: it scrapes LinkedIn job postings, extracts structured requirements with a local LLM, matches jobs to a resume with semantic vector search, and then hands the shortlist to a **LangGraph multi-agent team** — a *screener* agent that investigates each job and decides which are worth pursuing, and a *preparer* agent that writes tailored application materials for the selected ones, connected by a typed handoff.

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
     ┌────────┐
     │ intake │   (--interactive only: interrupt() asks for skills,
     └───┬────┘    locations, relocation, visa — feeds the filter)
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
    ┌──────────┐  tool calls  ┌────────────────┐   analyze_fit,
    │ screener ├─────────────►│ screener_tools │   get_job_description,
    │  (Groq)  │◄─────────────┤                │   check_job_active,
    └────┬─────┘              └────────────────┘   record_screening_decision
          │ no tool calls
          ▼
     ┌─────────┐   typed handoff: ScreeningDecision per job;
     │ handoff │   screener summary stashed, message channel
     └────┬────┘   CLEARED — the preparer starts a fresh context
          │
          ├── nothing pursued ─────────────────────────┐
          ▼                                            │
    ┌──────────┐  tool calls  ┌────────────────┐       │
    │ preparer ├─────────────►│ preparer_tools │       │   research_company,
    │  (Groq)  │◄─────────────┤                │       │   tailor_resume,
    └────┬─────┘              └────────────────┘       │   write_cover_letter
          │ no tool calls                              │   (draft → LLM review
          ▼                                            │      → revise)
 ┌──────────────┐◄─────────────────────────────────────┘
 │ human_review │   interrupt(): approve/skip each prepared application
 └──────┬───────┘
         ▼
        END
```

- `find_jobs` / `extract_metadata` are **deterministic nodes**: embed the resume, search FAISS, apply compatibility filters, enrich with structured requirements. A conditional edge routes to END if matching fails (`state["error"]`).
- **Two specialized agents, not one generalist.** The **screener** investigates each shortlisted job (a deterministic `analyze_fit` report, the full description on demand, an optional LinkedIn liveness check) and records a `ScreeningDecision` (pursue/skip + reason) per job. The **preparer** writes materials for the pursued jobs only. Each agent is bound with **only its own tool roster**, so the capability split is enforced at the model level — the screener cannot generate documents, the preparer cannot re-screen.
- **Typed handoff, isolated contexts.** The agents never share a transcript. The `handoff` node stashes the screener's summary, wipes the message channel (`RemoveMessage(REMOVE_ALL_MESSAGES)`), and the preparer seeds a fresh conversation rendered from `state["screening"]` — all inter-agent communication flows through typed Pydantic state, never prose-in-context. Skipped jobs are recorded with the screener's reason and bypass the preparer entirely.
- The **generation tools run an evaluator-optimizer loop**: every draft is judged by a structured-output LLM reviewer for truthfulness against the original resume and regenerated with the critique when rejected (the same retry-with-feedback idea as the extraction layer, applied to open-ended generation).
- **Applying is not a tool.** Prepared materials stop at the `human_review` interrupt; a person approves or skips each job, and only then are statuses set and artifacts saved. Neither agent can mark anything applied.

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

# Ask for preferences (must-have skills, locations, visa) before matching:
python -m app.agent.run path/to/resume.txt --interactive
```

The run pauses twice for input when there is something to decide: at intake (with `--interactive`) and at the approval gate whenever materials were prepared. The agent prints a summary of which jobs it prepared or skipped (and why), then writes tailored resumes and cover letters for the **approved** jobs to `outputs/<job_id>/` along with a per-job status report.

## Design decisions

- **Deterministic spine, agentic head.** Matching and enrichment are plain graph nodes with explicit conditional edges; the LLM loop only starts once there is a concrete, typed shortlist to reason over. Agents should decide things that need judgment — not run ETL. Even inside the loop, `analyze_fit` is deterministic: it reports facts, and the model supplies the judgment.
- **Multi-agent where the split earns its keep.** Screening and preparing are different jobs with different tools, different prompts, and different failure modes — so they are different agents with a typed handoff, not one agent with a longer prompt. The split buys enforced capability boundaries (roster-level, not prompt-level), small isolated contexts (the preparer never pays tokens for the screener's investigation transcript), and independently testable roles.
- **Typed state everywhere, merged by reducers.** `AgentState` is a TypedDict whose fields are Pydantic domain models (`CandidateProfile`, `JobMatch`, `TailoredArtifacts`, `ApplicationRecord`). `messages` appends; `artifacts` and `applications` merge per job (field-wise for artifacts), so the parallel tool calls the model routinely emits in one turn can never clobber each other. `matches` is overwritten wholesale by the node that owns it.
- **Tools write state via `Command`, not prose.** Action tools receive `InjectedState`, look up the job by id, and return a `Command` with per-job *deltas* to typed fields (`artifacts`, `applications`) plus a `ToolMessage`. Results are never smuggled through free-text messages.
- **Generation is reviewed, not trusted.** Tailored resumes and cover letters pass an LLM-as-judge truthfulness check (structured output) and are revised with the critique when rejected; the tool result tells the agent whether the final draft passed.
- **Humans gate side effects.** LangGraph `interrupt()` + a checkpointer pause the run for profile intake and for per-job approval of prepared applications. "Applied" is a status only a human decision can set.
- **Dependency-injected model.** `build_graph(model=...)` accepts any chat model, so the whole graph is testable with a fake — no API key needed (see `tests/test_agent_graph.py`).
- **Schema-validated extraction with error feedback.** The extraction prompt embeds the Pydantic JSON schema; invalid responses are retried with the specific validation error appended, which fixes most malformed outputs within a retry or two.
- **Metadata is extracted once, then reused.** Batch extraction persists requirements to SQLite; at query time the matcher rehydrates them, and the agent's enrichment node only calls the LLM for jobs with gaps.
- **Failures are data, not crashes.** Node failures land in `state["error"]` and route through a conditional edge; per-job failures in batch jobs are collected and reported, leaving the job eligible for retry on the next run.

## Testing

```bash
python -m pytest tests/
```

The suite covers the metadata compatibility filters (including skills / location / visa preferences), metadata rehydration from DB rows, prompt rendering, the fit-report tool, the critique-and-revise loop (revision on rejection, giving up after max rounds), reducer merging under parallel tool calls, and full graph wiring — happy path, error routing, enrichment skipping, the two-agent screener → handoff → preparer flow (including an assertion that the preparer starts from a fresh 2-message context, proving isolation), the skip-all path that bypasses the preparer, and both interrupts (intake and approval) resumed via `Command(resume=...)` — using injected fake/scripted models. No network, database, or API keys required.

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

- **Persistent sessions**: swap the in-memory checkpointer for `SqliteSaver` so an interrupted run (or a crash mid-loop) can resume across processes, and persist `applications` so past decisions inform future runs.
- **Two-tower retrieval**: separate resume/job encoders trained on application feedback, replacing the single off-the-shelf embedding model.
- **Re-ranking** with explainable signals (skill overlap, recency, seniority match) on top of cosine similarity.
- **Resume parsing** to derive `experience_years` and skills from the resume instead of CLI flags.
- Scraping depends on LinkedIn's internal API and cookies; it is best-effort and for personal use.
