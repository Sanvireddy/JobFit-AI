"""JobFit-AI application package.

Loads environment variables from a project-root ``.env`` file (if present) as
early as possible, so every entrypoint picks up secrets like ``GROQ_API_KEY``
and ``LINKEDIN_EMAIL`` / ``LINKEDIN_PASSWORD`` without needing manual
``export``s. Real values live in ``.env`` (gitignored); ``.env.example``
documents the expected keys.
"""

from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv not installed: fall back to real env vars.
    load_dotenv = None

if load_dotenv is not None:
    # override=False so an explicitly exported variable still wins over .env.
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
