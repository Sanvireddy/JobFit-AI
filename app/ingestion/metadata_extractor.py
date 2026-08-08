"""Structured metadata extraction from job descriptions via a local LLM.

A schema-driven prompt (see ``app/llm/prompts/extraction_prompt.txt``) asks the
model for strictly valid JSON matching :class:`JobMetadata`. The output is
validated with Pydantic; on validation failure the error is fed back to the
model and the call is retried, which fixes most malformed responses.
"""

import json
import logging
from functools import lru_cache

import ollama

from app.config import EXTRACTION_PROMPT_PATH, OLLAMA_MODEL
from app.observability import add_trace_metadata, traceable
from app.schemas.job_metadata import JobMetadata

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


class MetadataExtractionError(Exception):
    """Raised when the LLM cannot produce valid metadata after retries."""


@lru_cache(maxsize=1)
def _prompt_template() -> str:
    template = EXTRACTION_PROMPT_PATH.read_text()
    schema = json.dumps(JobMetadata.model_json_schema(), indent=2)
    return template.replace("{output_json}", schema)


def build_prompt(job_description: str) -> str:
    return _prompt_template().replace("{job_description}", job_description)


def _strip_code_fences(content: str) -> str:
    return content.replace("```json", "").replace("```", "").strip()


@traceable(run_type="llm", name="ollama.chat")
def _ollama_chat(prompt: str) -> str:
    """One local-Ollama completion, traced as an ``llm`` run.

    Ollama is not a LangChain model, so it is invisible to LangSmith without
    this wrapper; the model's own token counts are surfaced as metadata.
    """
    response = ollama.chat(
        model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}]
    )
    add_trace_metadata(
        model=OLLAMA_MODEL,
        prompt_tokens=response.get("prompt_eval_count"),
        completion_tokens=response.get("eval_count"),
    )
    return response["message"]["content"]


@traceable(run_type="chain", name="extract_metadata")
def extract_metadata(job_description: str, max_retries: int = MAX_RETRIES) -> JobMetadata:
    """Extract validated ``JobMetadata`` from raw job description text.

    Raises:
        MetadataExtractionError: if no valid output is produced after
            ``max_retries`` attempts.
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        prompt = build_prompt(job_description)
        if last_error:
            prompt += (
                "\n\n## PREVIOUS ATTEMPT FAILED\n"
                "Your previous response failed Pydantic validation with this error:\n"
                f"{last_error}\n"
                "Return corrected JSON that fixes this specific issue. "
                "Do NOT repeat the same mistake."
            )

        content = _strip_code_fences(_ollama_chat(prompt))
        try:
            result = JobMetadata.model_validate(json.loads(content))
            add_trace_metadata(attempts=attempt, succeeded=True)
            return result
        except Exception as exc:
            last_error = str(exc)
            logger.warning("Extraction attempt %d/%d failed: %s", attempt, max_retries, exc)

    add_trace_metadata(attempts=max_retries, succeeded=False, last_error=last_error)
    raise MetadataExtractionError(
        f"Failed to extract valid metadata after {max_retries} attempts: {last_error}"
    )
