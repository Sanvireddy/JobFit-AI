"""LangGraph wiring for the JobFit-AI agent.

This assembles the deterministic spine of the pipeline:

    START -> find_jobs_node -> extract_metadata_node -> END

with a conditional edge after ``find_jobs_node`` that short-circuits to END when
a node has recorded a problem in ``state["error"]`` (instead of raising). The
graph is built in a factory (``build_graph``) so importing this module stays
cheap and side-effect free, matching the rest of ``app.agent``. The Groq-powered
``agent_node`` + ``ToolNode`` loop attaches after this spine in a later step.
"""

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import extract_metadata_node, find_jobs_node
from app.agent.state import AgentState, CandidateProfile


def _route_after_find_jobs(state: AgentState) -> str:
    """Continue to metadata enrichment, or bail to END if matching failed."""
    return "end" if state.get("error") else "continue"


def build_graph():
    """Assemble and compile the deterministic matching spine."""
    builder = StateGraph(AgentState)

    builder.add_node("find_jobs", find_jobs_node)
    builder.add_node("extract_metadata", extract_metadata_node)

    builder.add_edge(START, "find_jobs")
    builder.add_conditional_edges(
        "find_jobs",
        _route_after_find_jobs,
        {"continue": "extract_metadata", "end": END},
    )
    builder.add_edge("extract_metadata", END)

    return builder.compile()


def initial_state(
    resume_text: str,
    top_k: int = 5,
    experience_years: int = 3,
) -> AgentState:
    """Build a fresh AgentState to invoke the graph with.

    Convenience so callers can do ``build_graph().invoke(initial_state(resume))``
    without hand-assembling every state key.
    """
    return {
        "messages": [],
        "candidate": CandidateProfile(
            resume_text=resume_text,
            experience_years=experience_years,
        ),
        "top_k": top_k,
        "matches": [],
        "artifacts": {},
        "applications": {},
        "error": None,
    }
