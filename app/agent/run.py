"""CLI entrypoint for the JobFit-AI agent.

Reads a resume, runs the full graph (optional intake -> match -> enrich ->
agent loop -> human review), prints the agent's summary, and saves artifacts
for approved jobs to ``outputs/<job_id>/``.

The graph pauses at ``interrupt()`` points (profile intake with
``--interactive``, and application approval whenever materials were prepared);
this CLI answers each pause on stdin and resumes the same thread until the run
completes.

Run with:  python -m app.agent.run path/to/resume.txt --top-k 5
"""

import argparse
import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.agent.graph import build_graph, initial_state
from app.config import OUTPUT_DIR

logger = logging.getLogger(__name__)


def save_artifacts(artifacts: dict, skipped_job_ids: set) -> None:
    for job_id, artifact in artifacts.items():
        if job_id in skipped_job_ids:
            logger.info("Skipping artifacts for job %s (not approved)", job_id)
            continue
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        if artifact.tailored_resume:
            (job_dir / "tailored_resume.md").write_text(artifact.tailored_resume)
        if artifact.cover_letter:
            (job_dir / "cover_letter.md").write_text(artifact.cover_letter)
        logger.info("Saved artifacts for job %s to %s", job_id, job_dir)


def _answer_interrupt(payload: dict):
    """Collect the human's answer for one graph interrupt on stdin."""
    if payload.get("type") == "profile_intake":
        print("\n=== Candidate intake ===")
        return {
            key: input(f"{question} ")
            for key, question in payload.get("questions", {}).items()
        }

    if payload.get("type") == "application_approval":
        print("\n=== Review prepared applications ===")
        decisions = {}
        for job in payload.get("jobs", []):
            parts = []
            if job.get("has_resume"):
                parts.append("resume")
            if job.get("has_cover_letter"):
                parts.append("cover letter")
            prepared = " + ".join(parts) or "no documents"
            answer = input(
                f"Approve {job['title']} at {job['company']} "
                f"[{job['job_id']}] ({prepared})? [y/N] "
            )
            decisions[job["job_id"]] = answer.strip() or "skip"
        return decisions

    # Unknown payload: resume with a scalar rather than stalling the run.
    # (Resuming with an empty dict would NOT resume: langgraph reads {} as an
    # empty interrupt-id map.)
    logger.warning("Unrecognized interrupt payload: %s", payload)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the JobFit-AI agent.")
    parser.add_argument("resume", help="Path to the resume as a plain-text file")
    parser.add_argument("--top-k", type=int, default=5, help="Jobs to shortlist")
    parser.add_argument(
        "--experience-years",
        type=int,
        default=3,
        help="Candidate's years of experience (used by the compatibility filter)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Ask for preferences (skills, locations, visa) before matching",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with open(args.resume, encoding="utf-8") as file:
        resume_text = file.read()

    # MemorySaver is enough for a single-process CLI run: it lets the graph
    # pause at interrupt() and resume on the same thread_id.
    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "cli"}}

    state = graph.invoke(
        initial_state(
            resume_text,
            top_k=args.top_k,
            experience_years=args.experience_years,
            interactive=args.interactive,
        ),
        config,
    )
    while state.get("__interrupt__"):
        answer = _answer_interrupt(state["__interrupt__"][0].value)
        state = graph.invoke(Command(resume=answer), config)

    if state.get("error"):
        raise SystemExit(f"Agent run failed: {state['error']}")

    messages = state.get("messages") or []
    if messages:
        print("\n=== Agent summary ===\n")
        print(messages[-1].content)

    applications = state.get("applications") or {}
    skipped = {
        job_id
        for job_id, record in applications.items()
        if record.status in ("skipped", "expired")
    }
    if applications:
        print("\n=== Application status ===\n")
        for job_id, record in sorted(applications.items()):
            note = f" — {record.notes}" if record.notes else ""
            print(f"  {job_id}: {record.status}{note}")

    save_artifacts(state.get("artifacts") or {}, skipped)


if __name__ == "__main__":
    main()
