"""Central configuration for JobFit-AI.

All filesystem paths are anchored to the repository root so every entrypoint
works regardless of the current working directory. Model names can be
overridden through environment variables without touching code.
"""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

# --- Storage ---------------------------------------------------------------
DB_PATH = ROOT_DIR / "jobs.db"
FAISS_INDEX_PATH = ROOT_DIR / "job_desc.index"
OUTPUT_DIR = ROOT_DIR / "outputs"

# --- Prompts ---------------------------------------------------------------
EXTRACTION_PROMPT_PATH = ROOT_DIR / "app" / "llm" / "prompts" / "extraction_prompt.txt"

# --- Models ----------------------------------------------------------------
# Sentence-transformers model used for job/resume embeddings.
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSIONS = 384  # output dimension of all-MiniLM-L6-v2

# Local Ollama model used for structured metadata extraction.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

# Groq-hosted model that powers the agent loop (must support tool calling).
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
