"""CLI entrypoint for the JobFit-AI agent.

Reads a resume, runs the full graph (match -> enrich -> agent loop), prints
the agent's summary, and saves any tailored artifacts to ``outputs/<job_id>/``.

Run with:  python -m app.agent.run path/to/resume.txt --top-k 5
"""

import argparse
import logging

from app.agent.graph import build_graph, initial_state
from app.config import OUTPUT_DIR

logger = logging.getLogger(__name__)


def save_artifacts(artifacts: dict) -> None:
    for job_id, artifact in artifacts.items():
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        if artifact.tailored_resume:
            (job_dir / "tailored_resume.md").write_text(artifact.tailored_resume)
        if artifact.cover_letter:
            (job_dir / "cover_letter.md").write_text(artifact.cover_letter)
        logger.info("Saved artifacts for job %s to %s", job_id, job_dir)


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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with open(args.resume, encoding="utf-8") as file:
        resume_text = file.read()

    graph = build_graph()
    final_state = graph.invoke(
        initial_state(
            resume_text,
            top_k=args.top_k,
            experience_years=args.experience_years,
        )
    )

    if final_state.get("error"):
        raise SystemExit(f"Agent run failed: {final_state['error']}")

    messages = final_state.get("messages") or []
    if messages:
        print("\n=== Agent summary ===\n")
        print(messages[-1].content)

    save_artifacts(final_state.get("artifacts") or {})


if __name__ == "__main__":
    main()
