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

        response = ollama.chat(
            model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}]
        )
        content = _strip_code_fences(response["message"]["content"])
        try:
            return JobMetadata.model_validate(json.loads(content))
        except Exception as exc:
            last_error = str(exc)
            logger.warning("Extraction attempt %d/%d failed: %s", attempt, max_retries, exc)

    raise MetadataExtractionError(
        f"Failed to extract valid metadata after {max_retries} attempts: {last_error}"
    )
