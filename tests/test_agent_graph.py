"""Graph wiring tests with a fake model — no API keys, DB, or network needed."""

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

import app.agent.nodes as nodes
import app.agent.tools as tools_mod
from app.agent.graph import build_graph, initial_state
from app.agent.prompts import format_shortlist
from app.agent.state import CandidateProfile, JobMatch, merge_artifacts, TailoredArtifacts
from app.agent.tools import TOOLS, DraftReview, _find_match, analyze_fit
from app.schemas.job_metadata import ExperienceRequirement, JobMetadata


class FakeModel:
    """Minimal chat model: replies with a canned message, records calls."""

    def __init__(self, reply="Prepared 1 job."):
        self.reply = reply
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.reply)


class FakeToolCallModel:
    """Emits parallel tool calls on the first turn, then a plain reply."""

    def __init__(self, tool_calls):
        self.tool_calls = tool_calls
        self.turns = 0

    def invoke(self, messages):
        self.turns += 1
        if self.turns == 1:
            return AIMessage(content="", tool_calls=self.tool_calls)
        return AIMessage(content="Prepared job 42.")


def fake_match(job_id="42", metadata=None):
    return JobMatch(
        job_id=job_id,
        title="ML Engineer",
        company="Acme",
        similarity_score=0.87,
        description="Build ML systems with Python.",
        passed_filters=True,
        metadata=metadata,
    )


def approve_all_reviews(monkeypatch):
    monkeypatch.setattr(tools_mod, "_groq_generate", lambda sys, usr: "DRAFT")
    monkeypatch.setattr(
        tools_mod, "_review_draft", lambda *a, **kw: DraftReview(approved=True)
    )


def test_happy_path_reaches_agent_and_ends(monkeypatch):
    monkeypatch.setattr(nodes, "find_jobs", lambda *a, **kw: [fake_match()])
    model = FakeModel()

    graph = build_graph(model=model, tools=TOOLS)
    final_state = graph.invoke(initial_state("my resume", top_k=3))

    assert final_state["error"] is None
    assert [m.job_id for m in final_state["matches"]] == ["42"]
    # Model was seeded with system prompt + shortlist and replied once.
    assert len(model.calls) == 1
    assert final_state["messages"][-1].content == "Prepared 1 job."


def test_matching_failure_routes_to_end_without_calling_model(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("index missing")

    monkeypatch.setattr(nodes, "find_jobs", boom)
    model = FakeModel()

    graph = build_graph(model=model, tools=TOOLS)
    final_state = graph.invoke(initial_state("my resume"))

    assert "index missing" in final_state["error"]
    assert final_state["matches"] == []
    assert model.calls == []


def test_metadata_node_skips_matches_that_already_have_metadata(monkeypatch):
    match = fake_match()
    monkeypatch.setattr(nodes, "find_jobs", lambda *a, **kw: [match])

    def no_extraction(*args, **kwargs):
        raise AssertionError("extract_job_metadata should not run")

    # Match has no metadata but extraction failing should not abort the run.
    monkeypatch.setattr(nodes, "extract_job_metadata", no_extraction)
    graph = build_graph(model=FakeModel(), tools=TOOLS)
    final_state = graph.invoke(initial_state("my resume"))
    assert final_state["matches"][0].metadata is None  # failure swallowed


def test_parallel_tool_calls_merge_and_human_review_approves(monkeypatch):
    """Resume + cover letter in ONE model turn, then the approval interrupt."""
    monkeypatch.setattr(nodes, "find_jobs", lambda *a, **kw: [fake_match()])
    approve_all_reviews(monkeypatch)

    model = FakeToolCallModel([
        {"name": "tailor_resume", "args": {"job_id": "42"}, "id": "c1"},
        {"name": "write_cover_letter", "args": {"job_id": "42"}, "id": "c2"},
    ])
    graph = build_graph(model=model, tools=TOOLS, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t1"}}

    state = graph.invoke(initial_state("my resume"), config)

    # Both parallel deltas survived the merge reducers.
    artifact = state["artifacts"]["42"]
    assert artifact.tailored_resume == "DRAFT"
    assert artifact.cover_letter == "DRAFT"
    assert state["applications"]["42"].status == "tailored"

    # Graph paused at the human-review gate.
    payload = state["__interrupt__"][0].value
    assert payload["type"] == "application_approval"
    assert payload["jobs"][0]["has_resume"] and payload["jobs"][0]["has_cover_letter"]

    final = graph.invoke(Command(resume={"42": "y"}), config)
    assert final["applications"]["42"].status == "applied"


def test_human_review_skips_unapproved_jobs(monkeypatch):
    monkeypatch.setattr(nodes, "find_jobs", lambda *a, **kw: [fake_match()])
    approve_all_reviews(monkeypatch)

    model = FakeToolCallModel(
        [{"name": "tailor_resume", "args": {"job_id": "42"}, "id": "c1"}]
    )
    graph = build_graph(model=model, tools=TOOLS, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t2"}}

    graph.invoke(initial_state("my resume"), config)
    # A decision dict that doesn't mention job 42: absence means skip.
    # (Never resume with a truly empty dict — langgraph reads {} as an empty
    # interrupt-id map and resumes nothing.)
    final = graph.invoke(Command(resume={"unrelated": "y"}), config)
    assert final["applications"]["42"].status == "skipped"


def test_no_artifacts_means_no_interrupt(monkeypatch):
    monkeypatch.setattr(nodes, "find_jobs", lambda *a, **kw: [fake_match()])
    graph = build_graph(model=FakeModel(), tools=TOOLS, checkpointer=MemorySaver())
    state = graph.invoke(
        initial_state("my resume"), {"configurable": {"thread_id": "t3"}}
    )
    assert "__interrupt__" not in state


def test_interactive_intake_fills_profile_and_feeds_matcher(monkeypatch):
    seen = {}

    def capture_find(candidate, top_k=5):
        seen["candidate"] = candidate
        return []

    monkeypatch.setattr(nodes, "find_jobs", capture_find)
    graph = build_graph(model=FakeModel(), tools=TOOLS, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t4"}}

    state = graph.invoke(initial_state("my resume", interactive=True), config)
    assert state["__interrupt__"][0].value["type"] == "profile_intake"

    graph.invoke(Command(resume={
        "must_have_skills": "python, sql",
        "preferred_locations": "Germany",
        "open_to_relocation": "y",
        "requires_visa_sponsorship": "",
    }), config)

    candidate = seen["candidate"]
    assert candidate.must_have_skills == ["python", "sql"]
    assert candidate.preferred_locations == ["Germany"]
    assert candidate.open_to_relocation is True
    assert candidate.requires_visa_sponsorship is False


def test_generate_reviewed_revises_rejected_drafts(monkeypatch):
    drafts = iter(["bad draft", "good draft"])
    reviews = iter([
        DraftReview(approved=False, issues=["invented a PhD"]),
        DraftReview(approved=True),
    ])
    prompts_seen = []

    def fake_generate(system_prompt, user_prompt):
        prompts_seen.append(user_prompt)
        return next(drafts)

    monkeypatch.setattr(tools_mod, "_groq_generate", fake_generate)
    monkeypatch.setattr(
        tools_mod, "_review_draft", lambda *a, **kw: next(reviews)
    )

    draft, review = tools_mod._generate_reviewed(
        "tailored resume", "sys", "job desc", "resume"
    )
    assert draft == "good draft"
    assert review.approved
    # The revision prompt carried the reviewer's critique back to the model.
    assert "invented a PhD" in prompts_seen[1]


def test_generate_reviewed_gives_up_after_max_rounds(monkeypatch):
    monkeypatch.setattr(tools_mod, "_groq_generate", lambda s, u: "still bad")
    monkeypatch.setattr(
        tools_mod,
        "_review_draft",
        lambda *a, **kw: DraftReview(approved=False, issues=["untruthful"]),
    )
    draft, review = tools_mod._generate_reviewed("cover letter", "sys", "jd", "r")
    assert draft == "still bad"
    assert not review.approved  # surfaced to the agent, not hidden


def test_analyze_fit_reports_facts():
    metadata = JobMetadata(
        experience_requirement=ExperienceRequirement(
            min_years_experience=5,
            experience_requirement_evidence="5+ years required",
        )
    )
    state = {
        "matches": [fake_match(metadata=metadata)],
        "candidate": CandidateProfile(
            resume_text="r",
            experience_years=3,
            must_have_skills=["Python", "Kubernetes"],
        ),
    }
    report = analyze_fit.func(job_id="42", state=state)
    assert "short by 2" in report
    assert "must-have skills mentioned: Python" in report
    assert "NOT mentioned: Kubernetes" in report

    assert "No shortlisted job" in analyze_fit.func(job_id="zzz", state=state)


def test_merge_artifacts_combines_fields_for_same_job():
    left = {"42": TailoredArtifacts(job_id="42", tailored_resume="R")}
    right = {"42": TailoredArtifacts(job_id="42", cover_letter="C")}
    merged = merge_artifacts(left, right)
    assert merged["42"].tailored_resume == "R"
    assert merged["42"].cover_letter == "C"


def test_format_shortlist_renders_ids_and_scores():
    text = format_shortlist([fake_match()])
    assert "[42]" in text
    assert "ML Engineer at Acme" in text
    assert "0.87" in text


def test_format_shortlist_empty():
    assert "No matching jobs" in format_shortlist([])


def test_find_match_helper():
    state = {"matches": [fake_match("a"), fake_match("b")]}
    assert _find_match(state, "b").job_id == "b"
    assert _find_match(state, "zzz") is None
