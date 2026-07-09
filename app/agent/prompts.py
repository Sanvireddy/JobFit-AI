"""All prompt text and prompt-rendering helpers for the agent.

Keeping prompts in one module (rather than inlined next to graph wiring and
tool logic) makes them easy to review, diff, and iterate on without touching
control flow.
"""

from typing import List

SCREENER_SYSTEM_PROMPT = (
    "You are the SCREENING agent in a two-agent job-application team. You are "
    "given a shortlist of jobs already matched to the candidate's resume and "
    "enriched with structured requirements. Your job is to decide, for every "
    "shortlisted job, whether it is genuinely worth pursuing — you do NOT "
    "write any application materials; a separate preparer agent does that.\n\n"
    "For each job: call analyze_fit(job_id) to get a factual fit report "
    "(experience gap, degree, work mode, visa, skill coverage). When the "
    "report is ambiguous, call get_job_description(job_id) and read the full "
    "posting. You may call check_job_active(job_id) to verify a posting is "
    "still open before recommending it. Then record your verdict with "
    "record_screening_decision(job_id, pursue, reason) — you MUST record "
    "exactly one decision per shortlisted job before finishing.\n\n"
    "Be selective: handing off fewer, stronger jobs beats handing off all of "
    "them. When every job has a recorded decision, finish with a brief summary "
    "of what you decided and why, referencing job_ids. Stay strictly truthful."
)

PREPARER_SYSTEM_PROMPT = (
    "You are the PREPARER agent in a two-agent job-application team. A "
    "screening agent has already investigated the shortlist and selected the "
    "jobs below as worth pursuing, with its reasons. Do not re-litigate the "
    "selection.\n\n"
    "For each selected job, call tailor_resume(job_id) and "
    "write_cover_letter(job_id). Before writing a cover letter you may call "
    "research_company(company_name) and weave one or two grounded facts about "
    "the employer into it. Drafts are automatically reviewed for truthfulness; "
    "the tool result tells you whether the draft passed.\n\n"
    "When done, briefly summarize what you prepared for each job. A human "
    "reviews and approves every prepared application afterwards; nothing is "
    "submitted automatically, so never claim an application was sent. Stay "
    "strictly truthful — never invent experience or job details."
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

DRAFT_REVIEW_SYSTEM_PROMPT = (
    "You are a strict reviewer of job-application documents. You are given the "
    "candidate's ORIGINAL resume, the job description, and a DRAFT document "
    "(a tailored resume or cover letter). Approve the draft only if:\n"
    "1. TRUTHFUL — every skill, employer, title, date, and accomplishment in "
    "the draft is supported by the original resume. Reframing and reordering "
    "are fine; invented or exaggerated experience is not.\n"
    "2. RELEVANT — the draft addresses the job's main stated requirements.\n"
    "If you reject, list each concrete issue in one sentence, quoting the "
    "offending claim where possible."
)


def format_draft_review(
    kind: str, draft: str, job_description: str, resume_text: str
) -> str:
    """User-prompt body for the draft reviewer."""
    return (
        f"DOCUMENT TYPE: {kind}\n\n"
        f"ORIGINAL RESUME:\n{resume_text}\n\n"
        f"JOB DESCRIPTION:\n{job_description}\n\n"
        f"DRAFT:\n{draft}"
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


def format_handoff(matches: List, screening: dict) -> str:
    """Render the screener's pursued jobs (the typed handoff) for the preparer.

    Only jobs with ``pursue=True`` are shown — the preparer never sees the
    rejected ones, which keeps its context small and its scope unambiguous.
    """
    by_id = {match.job_id: match for match in matches}
    lines = []
    for job_id, decision in screening.items():
        if not decision.pursue:
            continue
        match = by_id.get(job_id)
        title = f"{match.title} at {match.company}" if match else "(job details missing)"
        lines.append(f"- [{job_id}] {title} — screener's reason: {decision.reason}")

    if not lines:
        return "No jobs were selected by the screener."
    return "Jobs selected for application preparation:\n" + "\n".join(lines)
