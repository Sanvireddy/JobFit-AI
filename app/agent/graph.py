"""LangGraph wiring for the JobFit-AI agent.

Assembles the full pipeline: a deterministic spine, a Groq-powered agent loop,
and two human-in-the-loop pauses.

    START -> intake -> find_jobs -> extract_metadata -> agent <-> tools
               |           |                              |
      (interactive only:   └── (on error) ──> END         └── (no tool calls)
       interrupt for                                          ──> human_review
       preferences)                                                  |
                                                    (interrupt: approve/skip
                                                     each prepared job) -> END

- ``intake`` interrupts (only with ``interactive=True``) to collect the
  candidate's preferences, which feed the compatibility filter.
- ``find_jobs`` / ``extract_metadata`` are deterministic nodes (embed+filter,
  then LLM metadata enrichment). A conditional edge after ``find_jobs`` bails to
  END if matching recorded ``state["error"]``.
- ``agent`` invokes the Groq model over the conversation, having been seeded with
  the shortlisted matches. It either emits tool calls (routed to ``tools`` by the
  prebuilt ``tools_condition``, which loop back to ``agent``) or finishes,
  routing to ``human_review``.
- ``human_review`` interrupts so a person approves or skips every prepared
  application; the model can never mark anything applied itself. It is a no-op
  (and does not interrupt) when nothing was prepared.

Interrupts require compiling with a checkpointer — pass one via
``build_graph(checkpointer=...)`` (the CLI uses ``MemorySaver``). Tests build
without one; both interrupting nodes are no-ops on their default paths.

The model is *injected* into ``build_graph`` so the graph can be built and tested
without a GROQ_API_KEY (pass a fake model); in production it is lazily created
via ``get_agent_model()``.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.llm import get_agent_model
from app.agent.nodes import (
    extract_metadata_node,
    find_jobs_node,
    human_review_node,
    intake_node,
)
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


def build_graph(model=None, tools=None, checkpointer=None):
    """Assemble and compile the full agent graph.

    Args:
        model: A chat model bound with the agent tools. When None, one is built
            lazily via ``get_agent_model()`` (requires GROQ_API_KEY). Inject a
            fake model to build/test the graph without a key.
        tools: Tool list for the ToolNode. Defaults to ``TOOLS``.
        checkpointer: Optional LangGraph checkpointer. Required for the
            interactive intake and human-review interrupts to pause/resume;
            without one the graph still runs, but only the non-interrupting
            paths are reachable.
    """
    if model is None:
        model = get_agent_model()
    if tools is None:
        tools = TOOLS

    builder = StateGraph(AgentState)

    builder.add_node("intake", intake_node)
    builder.add_node("find_jobs", find_jobs_node)
    builder.add_node("extract_metadata", extract_metadata_node)
    builder.add_node("agent", make_agent_node(model))
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("human_review", human_review_node)

    builder.add_edge(START, "intake")
    builder.add_edge("intake", "find_jobs")
    builder.add_conditional_edges(
        "find_jobs",
        _route_after_find_jobs,
        {"continue": "extract_metadata", "end": END},
    )
    builder.add_edge("extract_metadata", "agent")
    # tools_condition routes to "tools" when the last message has tool calls;
    # its END outcome is remapped to the human-review gate.
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: "human_review"},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("human_review", END)

    return builder.compile(checkpointer=checkpointer)


def initial_state(
    resume_text: str,
    top_k: int = 5,
    experience_years: int = 3,
    interactive: bool = False,
) -> AgentState:
    """Build a fresh AgentState to invoke the graph with."""
    return {
        "messages": [],
        "candidate": CandidateProfile(
            resume_text=resume_text,
            experience_years=experience_years,
        ),
        "top_k": top_k,
        "interactive": interactive,
        "matches": [],
        "artifacts": {},
        "applications": {},
        "error": None,
    }
