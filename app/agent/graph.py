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
from app.agent.prompts import AGENT_SYSTEM_PROMPT, format_shortlist
from app.agent.state import AgentState, CandidateProfile
from app.agent.tools import TOOLS


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
            HumanMessage(content=format_shortlist(state.get("matches") or [])),
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
