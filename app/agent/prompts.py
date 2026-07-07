"""All prompt text and prompt-rendering helpers for the agent.

Keeping prompts in one module (rather than inlined next to graph wiring and
tool logic) makes them easy to review, diff, and iterate on without touching
control flow.
"""

from typing import List

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

TAILOR_RESUME_SYSTEM_PROMPT = (
    "You tailor resumes to a specific job. Rewrite the resume to emphasize "
    "experience relevant to the job. Stay strictly truthful; never invent "
    "experience the candidate does not have."
)

COVER_LETTER_SYSTEM_PROMPT = (
    "You write concise, specific cover letters grounded in the candidate's "
    "real experience. Do not invent facts."
)


def format_shortlist(matches: List) -> str:
    """Render the shortlisted matches into a compact text block for the model."""
    if not matches:
        return "No matching jobs were found for this resume."

    lines = ["Shortlisted jobs:"]
    for i, match in enumerate(matches, 1):
        bits = [f"{i}. [{match.job_id}] {match.title} at {match.company}"]
        bits.append(f"score {match.similarity_score:.2f}")
        if match.location:
            bits.append(match.location)
        meta = match.metadata
        if meta is not None:
            experience = meta.experience_requirement
            if experience and experience.min_years_experience is not None:
                bits.append(f"{experience.min_years_experience}y exp")
            if meta.relocation_requirement:
                bits.append(meta.relocation_requirement.work_mode)
        lines.append(" — ".join(bits))
    return "\n".join(lines)


def format_job_and_resume(job_description: str, resume_text: str) -> str:
    """User-prompt body shared by the tailoring tools."""
    return f"JOB DESCRIPTION:\n{job_description}\n\nRESUME:\n{resume_text}"
