"""LangGraph wiring for the JobFit-AI agent.

Assembles the full pipeline: a deterministic spine followed by a Groq-powered
agent loop.

    START -> find_jobs -> extract_metadata -> agent  <-> tools
                 |                               |
                 └── (on error) ──> END          └── (no tool calls) ──> END

- ``find_jobs`` / ``extract_metadata`` are deterministic nodes (embed+filter,
  then LLM metadata enrichment). A conditional edge after ``find_jobs`` bails to
  END if matching recorded ``state["error"]``.
- ``agent`` invokes the Groq model over the conversation, having been seeded with
  the shortlisted matches. It either emits tool calls (routed to ``tools`` by the
  prebuilt ``tools_condition``, which loop back to ``agent``) or finishes,
  routing to END.

The model is *injected* into ``build_graph`` so the graph can be built and tested
without a GROQ_API_KEY (pass a fake model); in production it is lazily created
via ``get_agent_model()``.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.llm import get_agent_model
from app.agent.nodes import extract_metadata_node, find_jobs_node
from app.agent.state import AgentState, CandidateProfile, JobMatch
from app.agent.tools import TOOLS

AGENT_SYSTEM_PROMPT = (
    "You are a job-application assistant. You are given a shortlist of jobs that "
    "were already matched to the candidate's resume and enriched with structured "
    "requirements. Decide which jobs are genuinely worth pursuing (strong fit and "
    "the candidate looks eligible). For each job you recommend, call tailor_resume "
    "and write_cover_letter (passing its job_id) to prepare the application "
    "materials. Do NOT call mark_applied unless the user has explicitly confirmed "
    "they applied. When done, briefly summarize which jobs you prepared and why. "
    "Reference job_ids from the shortlist and stay strictly truthful — never "
    "invent experience or job details."
)


def _matches_context(matches: list) -> str:
    """Render the shortlisted matches into a compact text block for the model."""
    if not matches:
        return "No matching jobs were found for this resume."

    lines = ["Shortlisted jobs:"]
    for i, m in enumerate(matches, 1):
        bits = [f"{i}. [{m.job_id}] {m.title} at {m.company}"]
        bits.append(f"score {m.similarity_score:.2f}")
        if m.location:
            bits.append(m.location)
        meta = m.metadata
        if meta is not None:
            if meta.experience_requirement and meta.experience_requirement.min_years_experience is not None:
                bits.append(f"{meta.experience_requirement.min_years_experience}y exp")
            if meta.relocation_requirement:
                bits.append(meta.relocation_requirement.work_mode)
        lines.append(" — ".join(bits))
    return "\n".join(lines)


def make_agent_node(model):
    """Build the agent node, closing over an injected chat model.

    On first entry (empty ``messages``) it seeds the system prompt and a
    human message summarizing the matches, then persists those alongside the
    model's reply (``messages`` appends via its reducer). On later loop
    iterations the conversation already carries history, so it just replies.
    """

    def agent_node(state: AgentState) -> dict:
        history = list(state.get("messages") or [])
        if history:
            response = model.invoke(history)
            return {"messages": [response]}

        seed = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=_matches_context(state.get("matches") or [])),
        ]
        response = model.invoke(seed)
        return {"messages": seed + [response]}

    return agent_node


def _route_after_find_jobs(state: AgentState) -> str:
    """Continue to metadata enrichment, or bail to END if matching failed."""
    return "end" if state.get("error") else "continue"


def build_graph(model=None, tools=None):
    """Assemble and compile the full agent graph.

    Args:
        model: A chat model bound with the agent tools. When None, one is built
            lazily via ``get_agent_model()`` (requires GROQ_API_KEY). Inject a
            fake model to build/test the graph without a key.
        tools: Tool list for the ToolNode. Defaults to ``TOOLS``.
    """
    if model is None:
        model = get_agent_model()
    if tools is None:
        tools = TOOLS

    builder = StateGraph(AgentState)

    builder.add_node("find_jobs", find_jobs_node)
    builder.add_node("extract_metadata", extract_metadata_node)
    builder.add_node("agent", make_agent_node(model))
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "find_jobs")
    builder.add_conditional_edges(
        "find_jobs",
        _route_after_find_jobs,
        {"continue": "extract_metadata", "end": END},
    )
    builder.add_edge("extract_metadata", "agent")
    # tools_condition routes to "tools" when the last message has tool calls,
    # otherwise to END.
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder.compile()


def initial_state(
    resume_text: str,
    top_k: int = 5,
    experience_years: int = 3,
) -> AgentState:
    """Build a fresh AgentState to invoke the graph with."""
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
