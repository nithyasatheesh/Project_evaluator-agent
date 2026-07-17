"""
utils.py
========
Shared dataclasses, logging setup and small pure-function helpers used
across the AI Assignment Evaluator. Keeping these in one module avoids
duplicated definitions between parser.py, rubric.py, evaluator.py and
exporter.py.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from typing import Any

import config


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def setup_logger(name: str) -> logging.Logger:
    """Create (or fetch) a module-level logger with consistent formatting.

    Safe to call multiple times for the same name; handlers are only
    attached once to avoid duplicated log lines.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
        logger.propagate = False
    return logger


# --------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------
@dataclass
class StudentContext:
    """All extracted content for a single student's submission.

    Every text bucket is a concatenation of the relevant files found for
    that student, already truncated to a safe size for LLM consumption.
    """

    name: str
    documentation: str = ""
    notebook_content: str = ""
    python_code: str = ""
    shell_scripts: str = ""
    sql_content: str = ""
    csv_summary: str = ""
    config_files: str = ""
    images: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    file_count: int = 0

    def is_empty(self) -> bool:
        """Return True if no meaningful content was extracted at all."""
        text_buckets = (
            self.documentation,
            self.notebook_content,
            self.python_code,
            self.shell_scripts,
            self.sql_content,
            self.csv_summary,
            self.config_files,
        )
        return not any(bucket.strip() for bucket in text_buckets) and not self.images


@dataclass
class RubricCriterion:
    """A single row of the evaluation rubric."""

    name: str
    max_score: float
    description: str = ""


@dataclass
class Rubric:
    """The full set of criteria extracted from the uploaded Excel rubric."""

    criteria: list[RubricCriterion]
    warnings: list[str] = field(default_factory=list)

    @property
    def total_max_score(self) -> float:
        return sum(c.max_score for c in self.criteria)

    @property
    def criterion_names(self) -> list[str]:
        return [c.name for c in self.criteria]


@dataclass
class EvaluationResult:
    """Final, scored evaluation output for a single student."""

    student: str
    scores: dict[str, float] = field(default_factory=dict)
    language_feedback: str = ""
    analysis_feedback: str = ""
    clarity_feedback: str = ""
    overall_feedback: str = ""
    strengths: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    total_score: float = 0.0
    max_score: float = 0.0
    percentage: float = 0.0
    grade: str = ""
    warnings: list[str] = field(default_factory=list)
    evaluation_failed: bool = False


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
def truncate_text(text: str, max_chars: int = config.MAX_CHARS_PER_CONTEXT_FIELD) -> str:
    """Truncate text to max_chars, appending a truncation notice if cut."""
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + config.TRUNCATION_NOTICE


def safe_decode(raw: bytes) -> str:
    """Decode bytes to text, trying utf-8 first and falling back gracefully."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def normalize_header(value: Any) -> str:
    """Normalise a spreadsheet header cell for alias matching.

    Lower-cases, strips surrounding whitespace and collapses internal
    whitespace so headers like "Max  Score" and "max score" both match.
    """
    if value is None:
        return ""
    text = str(value).strip().lower()
    return re.sub(r"\s+", " ", text)


def sanitize_sheet_name(name: str) -> str:
    """Sanitise a string for safe use as an Excel sheet name (<=31 chars)."""
    invalid = set('[]:*?/\\')
    cleaned = "".join(ch for ch in name if ch not in invalid).strip()
    return cleaned[:31] if cleaned else "Sheet1"


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp value to the inclusive [minimum, maximum] range."""
    return max(minimum, min(maximum, value))


def compute_grade(percentage: float) -> str:
    """Map a percentage score to a letter grade using GRADE_BOUNDARIES."""
    for threshold, label in config.GRADE_BOUNDARIES:
        if percentage >= threshold:
            return label
    return config.GRADE_BOUNDARIES[-1][1]


def coerce_float(value: Any, default: float = 0.0) -> float:
    """Best-effort conversion of an arbitrary value to float.

    Handles the score shapes LLMs commonly drift into even when asked
    for a plain number: "18/25", "18 out of 25", "18 pts", a dict like
    {"score": 18, "max": 25}, or a numeric string. Extracts the FIRST
    number found rather than naively stripping non-digit characters,
    which would otherwise mangle "18/25" into 1825.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("score", "value", "achieved", "points", "marks"):
            if key in value:
                return coerce_float(value[key], default)
        return default
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if match:
        try:
            return float(match.group())
        except ValueError:
            return default
    return default


def normalize_key(value: Any) -> str:
    """Normalise an arbitrary key (e.g. an LLM-returned criterion name)
    for tolerant, case/whitespace-insensitive matching against rubric
    criterion names."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def coerce_string_list(value: Any) -> list[str]:
    """Best-effort conversion of an arbitrary value into a list of strings.

    LLMs asked for a JSON list sometimes instead return a single
    newline/bullet-separated string, or a dict of labelled points. This
    normalises all of those shapes into a clean list so downstream
    columns (e.g. Areas of Strength / Areas of Improvement) are never
    silently left empty just because the shape wasn't a bare list.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, dict):
        return [str(item).strip() for item in value.values() if str(item).strip()]
    if isinstance(value, str):
        if not value.strip():
            return []
        # Split on newlines or bullet/number markers, then strip the marker itself.
        pieces = re.split(r"\n+", value.strip())
        if len(pieces) == 1:
            pieces = re.split(r"(?<=[.;])\s{2,}", value.strip())
        cleaned = [re.sub(r"^[\-\*•‣\d]+[\.\)]?\s*", "", p).strip() for p in pieces]
        return [c for c in cleaned if c]
    return [str(value).strip()] if str(value).strip() else []


def join_nonempty(parts: list[str], separator: str = "\n\n") -> str:
    """Join non-empty, stripped string parts with a separator."""
    return separator.join(p.strip() for p in parts if p and p.strip())
