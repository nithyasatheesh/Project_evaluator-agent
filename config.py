"""
config.py
=========
Central configuration for the AI Assignment Evaluator.

All tunable constants live here so that no other module contains
hardcoded "magic" values. Nothing in this module performs I/O other
than reading environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# General
# --------------------------------------------------------------------------
APP_NAME: str = "AI Assignment Evaluator"
APP_VERSION: str = "1.0.0"

# Base temp directory used while extracting master ZIP / nested ZIPs.
# A fresh sub-directory is created per run by parser.py.
TEMP_DIR_PREFIX: str = "ai_evaluator_"

# --------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4.1")
OPENAI_API_KEY_ENV_VAR: str = "OPENAI_API_KEY"
OPENAI_TEMPERATURE: float = 0.2
OPENAI_MAX_OUTPUT_TOKENS: int = 4000
OPENAI_REQUEST_TIMEOUT_SECONDS: int = 120

# Retry policy for transient OpenAI failures (rate limits, timeouts, etc.)
OPENAI_MAX_RETRIES: int = 3
OPENAI_RETRY_BACKOFF_SECONDS: float = 2.0

# --------------------------------------------------------------------------
# Concurrency
# --------------------------------------------------------------------------
# Default number of students evaluated in parallel. Exposed in the UI as
# an adjustable slider so faculty can tune it against their OpenAI rate
# limits without touching code.
DEFAULT_MAX_WORKERS: int = 4
MIN_MAX_WORKERS: int = 1
MAX_MAX_WORKERS: int = 10

# --------------------------------------------------------------------------
# Parser limits
# --------------------------------------------------------------------------
# Maximum nested-zip extraction depth. Prevents zip-bomb style recursion.
MAX_ZIP_EXTRACTION_DEPTH: int = 5

# Maximum number of bytes read from any single file before truncation.
MAX_FILE_READ_BYTES: int = 2_000_000  # ~2 MB

# Maximum number of rows read/summarised from a CSV file.
MAX_CSV_PREVIEW_ROWS: int = 20

# --------------------------------------------------------------------------
# Supported file extensions, grouped by the StudentContext bucket they
# populate. Extending support for a new file type only requires adding
# the extension here and a matching reader in readers.py.
# --------------------------------------------------------------------------
ARCHIVE_EXTENSIONS: set[str] = {".zip"}

DOC_EXTENSIONS: set[str] = {".pdf", ".docx", ".txt", ".md"}
NOTEBOOK_EXTENSIONS: set[str] = {".ipynb"}
HTML_EXTENSIONS: set[str] = {".html", ".htm"}
PYTHON_EXTENSIONS: set[str] = {".py"}
SHELL_EXTENSIONS: set[str] = {".sh", ".bash"}
SQL_EXTENSIONS: set[str] = {".sql"}
CSV_EXTENSIONS: set[str] = {".csv"}
CONFIG_EXTENSIONS: set[str] = {".json", ".yaml", ".yml"}
IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg"}

ALL_SUPPORTED_EXTENSIONS: set[str] = (
    DOC_EXTENSIONS
    | NOTEBOOK_EXTENSIONS
    | HTML_EXTENSIONS
    | PYTHON_EXTENSIONS
    | SHELL_EXTENSIONS
    | SQL_EXTENSIONS
    | CSV_EXTENSIONS
    | CONFIG_EXTENSIONS
    | IMAGE_EXTENSIONS
    | ARCHIVE_EXTENSIONS
)

# Filenames (case-insensitive, no extension check) that should always be
# treated as documentation even without a recognised extension.
DOCUMENTATION_FILENAME_HINTS: set[str] = {"readme", "report", "documentation"}

# --------------------------------------------------------------------------
# Prompt context truncation
# --------------------------------------------------------------------------
# Each StudentContext field is truncated to this many characters before
# being injected into the LLM prompt, to keep token usage bounded and
# predictable regardless of submission size.
MAX_CHARS_PER_CONTEXT_FIELD: int = 12_000
TRUNCATION_NOTICE: str = "\n\n[... content truncated for length ...]\n"

# --------------------------------------------------------------------------
# Rubric column aliases
# --------------------------------------------------------------------------
# Maps a canonical rubric field to the set of column header aliases (all
# lower-cased, whitespace-normalised) that may appear in an uploaded
# Excel rubric. The rubric parser never hardcodes actual criteria names.
RUBRIC_CRITERION_ALIASES: set[str] = {
    "criterion",
    "criteria",
    "evaluation criteria",
    "criterion name",
    "criteria name",
    "parameter",
    "parameters",
    "evaluation parameter",
    "assessment criteria",
    "rubric criteria",
    "component",
    "category",
    "aspect",
}
RUBRIC_MAX_SCORE_ALIASES: set[str] = {
    "max score",
    "max marks",
    "marks",
    "weight",
    "weightage",
    "maximum score",
    "maximum marks",
    "score",
    "points",
    "max points",
    "total marks",
    "total score",
    "out of",
}
RUBRIC_DESCRIPTION_ALIASES: set[str] = {
    "description",
    "evaluation parameters",
    "details",
    "expectations",
    "guidelines",
    "criteria description",
    "remarks",
}

# Number of leading rows scanned when looking for a header row in the
# rubric spreadsheet (some templates have a title row above the header).
RUBRIC_HEADER_SCAN_ROWS: int = 10

# --------------------------------------------------------------------------
# Grading scale
# --------------------------------------------------------------------------
# Ordered (descending) list of (minimum percentage, grade label) tuples.
# The first tuple whose threshold the achieved percentage meets or
# exceeds determines the grade.
GRADE_BOUNDARIES: list[tuple[float, str]] = [
    (90.0, "A+"),
    (80.0, "A"),
    (70.0, "B+"),
    (60.0, "B"),
    (50.0, "C"),
    (40.0, "D"),
    (0.0, "F"),
]

# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
DEFAULT_OUTPUT_FILENAME: str = "evaluation_results.xlsx"

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("EVALUATOR_LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_openai_api_key() -> str | None:
    """Return the configured OpenAI API key, if any.

    Reads from the environment only. The Streamlit UI layer is
    responsible for allowing a faculty member to paste a key at runtime
    and setting it into the environment / passing it explicitly to the
    evaluator, so this function stays side-effect free and UI-agnostic.
    """
    return os.environ.get(OPENAI_API_KEY_ENV_VAR)


def resolve_temp_root() -> Path:
    """Return a Path to the system temp directory used for extraction."""
    import tempfile

    return Path(tempfile.gettempdir())
