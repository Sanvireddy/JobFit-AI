"""LangGraph wiring for the JobFit-AI agent team.

Assembles the full pipeline: a deterministic spine, a two-agent screener →
preparer team with a typed handoff, and two human-in-the-loop pauses.

    START -> intake -> find_jobs -> extract_metadata -> screener <-> screener_tools
               |           |                               |
      (interactive only:   └── (on error) ──> END          └── (no tool calls)
       interrupt for                                           ──> handoff
       preferences)                                                 |
                              ┌──────────────────────────────────────┤
                              │ (no jobs pursued)                    │ (jobs pursued)
                              ▼                                      ▼
                        human_review <── (no tool calls) ── preparer <-> preparer_tools
                              |
                 (interrupt: approve/skip each prepared job) -> END

Multi-agent design, and why it is shaped this way:

- **Two specialized agents, not one generalist.** The *screener* investigates
  (fit reports, full descriptions, liveness checks) and records a
  ``ScreeningDecision`` per job; the *preparer* writes materials for the
  pursued jobs only. Each agent is bound with ONLY its own tool roster, so the
  capability split is enforced at the model level — the screener cannot
  generate documents and the preparer cannot re-screen.
- **Typed handoff, isolated contexts.** The agents never share a transcript.
  The ``handoff`` node stashes the screener's closing summary, wipes the
  message channel (``RemoveMessage(REMOVE_ALL_MESSAGES)``), and routes; the
  preparer then seeds a fresh conversation from ``state["screening"]`` — the
  Pydantic handoff — rendered by ``format_handoff``. All inter-agent
  communication flows through typed state, never prose-in-context.
- ``find_jobs`` / ``extract_metadata`` are deterministic nodes (embed+filter,
  then LLM metadata enrichment). A conditional edge after ``find_jobs`` bails
  to END if matching recorded ``state["error"]``.
- ``human_review`` interrupts so a person approves or skips every prepared
  application; neither agent can mark anything applied. It is a no-op (and
  does not interrupt) when nothing was prepared.

Interrupts require compiling with a checkpointer — pass one via
``build_graph(checkpointer=...)`` (the CLI uses ``MemorySaver``). Tests build
without one; both interrupting nodes are no-ops on their default paths.

Models are *injected* into ``build_graph`` so the graph can be built and tested
without a GROQ_API_KEY (pass a fake model, used for both agents); in production
each agent gets the Groq model bound with its own roster.
"""

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.llm import get_agent_model
from app.agent.nodes import (
    extract_metadata_node,
    find_jobs_node,
    human_review_node,
    intake_node,
)
from app.agent.prompts import (
    PREPARER_SYSTEM_PROMPT,
    SCREENER_SYSTEM_PROMPT,
    format_handoff,
    format_shortlist,
)
from app.agent.state import AgentState, CandidateProfile
from app.agent.tools import PREPARER_TOOLS, SCREENER_TOOLS


def make_agent_node(model, system_prompt, render_seed):
    """Build one agent node, closing over its model, role prompt, and seed.

    On first entry (empty ``messages``) it seeds the conversation with the
    role's system prompt and a human message rendered from state, then
    persists those alongside the model's reply (``messages`` appends via its
    reducer). On later loop iterations the conversation already carries
    history, so it just replies. Because the handoff node clears ``messages``,
    the same seeding logic serves both agents.
    """

    def agent_node(state: AgentState) -> dict:
        history = list(state.get("messages") or [])
        if history:
            response = model.invoke(history)
            return {"messages": [response]}

        seed = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=render_seed(state)),
        ]
        response = model.invoke(seed)
        return {"messages": seed + [response]}

    return agent_node


def handoff_node(state: AgentState) -> dict:
    """Close out the screener's turn and isolate the preparer's context.

    Stashes the screener's final summary (its last non-empty AI message) into
    ``screener_summary``, then clears the message channel so the preparer
    starts a fresh conversation seeded only from the typed handoff
    (``state["screening"]``). Routing happens on the conditional edge after
    this node.
    """
    summary = next(
        (
            message.content
            for message in reversed(state.get("messages") or [])
            if isinstance(message, AIMessage) and message.content
        ),
        None,
    )
    return {
        "screener_summary": summary,
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
    }


def _route_after_find_jobs(state: AgentState) -> str:
    """Continue to metadata enrichment, or bail to END if matching failed."""
    return "end" if state.get("error") else "continue"


def _route_after_handoff(state: AgentState) -> str:
    """Run the preparer only if the screener chose to pursue anything."""
    screening = state.get("screening") or {}
    if any(decision.pursue for decision in screening.values()):
        return "preparer"
    return "human_review"


def build_graph(model=None, tools=None, checkpointer=None):
    """Assemble and compile the two-agent graph.

    Args:
        model: A chat model used for BOTH agents (inject a fake to build/test
            the graph without a key). When None, each agent gets the Groq
            model bound with its own tool roster via ``get_agent_model()``.
        tools: Tool list override for BOTH ToolNodes (tests pass the union
            roster). Defaults to each agent's own roster.
        checkpointer: Optional LangGraph checkpointer. Required for the
            interactive intake and human-review interrupts to pause/resume;
            without one the graph still runs, but only the non-interrupting
            paths are reachable.
    """
    if model is None:
        screener_model = get_agent_model(tools=SCREENER_TOOLS)
        preparer_model = get_agent_model(tools=PREPARER_TOOLS)
    else:
        screener_model = preparer_model = model

    screener_toolset = tools if tools is not None else SCREENER_TOOLS
    preparer_toolset = tools if tools is not None else PREPARER_TOOLS

    builder = StateGraph(AgentState)

    builder.add_node("intake", intake_node)
    builder.add_node("find_jobs", find_jobs_node)
    builder.add_node("extract_metadata", extract_metadata_node)
    builder.add_node(
        "screener",
        make_agent_node(
            screener_model,
            SCREENER_SYSTEM_PROMPT,
            lambda state: format_shortlist(state.get("matches") or []),
        ),
    )
    builder.add_node("screener_tools", ToolNode(screener_toolset))
    builder.add_node("handoff", handoff_node)
    builder.add_node(
        "preparer",
        make_agent_node(
            preparer_model,
            PREPARER_SYSTEM_PROMPT,
            lambda state: format_handoff(
                state.get("matches") or [], state.get("screening") or {}
            ),
        ),
    )
    builder.add_node("preparer_tools", ToolNode(preparer_toolset))
    builder.add_node("human_review", human_review_node)

    builder.add_edge(START, "intake")
    builder.add_edge("intake", "find_jobs")
    builder.add_conditional_edges(
        "find_jobs",
        _route_after_find_jobs,
        {"continue": "extract_metadata", "end": END},
    )
    builder.add_edge("extract_metadata", "screener")
    # tools_condition routes to "tools" when the last message has tool calls;
    # its END outcome is remapped to each agent's next stage.
    builder.add_conditional_edges(
        "screener",
        tools_condition,
        {"tools": "screener_tools", END: "handoff"},
    )
    builder.add_edge("screener_tools", "screener")
    builder.add_conditional_edges(
        "handoff",
        _route_after_handoff,
        {"preparer": "preparer", "human_review": "human_review"},
    )
    builder.add_conditional_edges(
        "preparer",
        tools_condition,
        {"tools": "preparer_tools", END: "human_review"},
    )
    builder.add_edge("preparer_tools", "preparer")
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
        "screening": {},
        "artifacts": {},
        "applications": {},
        "error": None,
        "screener_summary": None,
    }
