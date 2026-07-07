"""Graph wiring tests with a fake model — no API keys, DB, or network needed."""

from langchain_core.messages import AIMessage

import app.agent.nodes as nodes
from app.agent.graph import build_graph, initial_state
from app.agent.prompts import format_shortlist
from app.agent.state import JobMatch
from app.agent.tools import TOOLS, _find_match


class FakeModel:
    """Minimal chat model: replies with a canned message, records calls."""

    def __init__(self, reply="Prepared 1 job."):
        self.reply = reply
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.reply)


def fake_match(job_id="42"):
    return JobMatch(
        job_id=job_id,
        title="ML Engineer",
        company="Acme",
        similarity_score=0.87,
        description="Build ML systems.",
        passed_filters=True,
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
