"""
rubric.py
=========
Parses a faculty-uploaded Excel rubric into a generic Rubric object.

The parser never hardcodes actual criteria (e.g. "Code Quality",
"EDA", "Model Accuracy"). Instead it identifies which *columns* of the
spreadsheet represent the criterion name, the maximum score and an
optional description, by matching column headers against the alias
sets defined in config.py. This lets the same code evaluate ML, NLP,
SQL, Power BI, or any other assignment rubric without modification.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import config
from utils import Rubric, RubricCriterion, coerce_float, normalize_header, setup_logger

logger = setup_logger(__name__)


class RubricParseError(Exception):
    """Raised when the rubric spreadsheet cannot be parsed into criteria."""


def _find_column(headers: list[str], aliases: set[str]) -> int | None:
    """Return the index of the first header matching one of the aliases."""
    for idx, header in enumerate(headers):
        if header in aliases:
            return idx
    return None


def parse_rubric(source: Path | BinaryIO | str) -> Rubric:
    """Parse an uploaded Excel rubric file into a Rubric.

    `source` may be a filesystem path or a file-like object (e.g. a
    Streamlit UploadedFile), since pandas.read_excel accepts both.

    Raises RubricParseError if no usable criterion/max-score columns can
    be identified, or if no valid criteria rows are found.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RubricParseError("pandas is not installed") from exc

    try:
        raw = pd.read_excel(source, header=None, dtype=object)
    except Exception as exc:  # noqa: BLE001
        raise RubricParseError(f"Could not open rubric spreadsheet: {exc}") from exc

    if raw.empty:
        raise RubricParseError("The rubric spreadsheet is empty.")

    header_row_idx, criterion_col, max_score_col, description_col = _locate_header_row(raw)

    if header_row_idx is None or criterion_col is None or max_score_col is None:
        raise RubricParseError(
            "Could not identify a 'Criterion' column and a 'Max Score' column in the "
            "rubric. Supported header names include: "
            f"{sorted(config.RUBRIC_CRITERION_ALIASES)} for the criterion name and "
            f"{sorted(config.RUBRIC_MAX_SCORE_ALIASES)} for the maximum score."
        )

    warnings: list[str] = []
    criteria: list[RubricCriterion] = []
    data_rows = raw.iloc[header_row_idx + 1 :]

    for row_num, row in data_rows.iterrows():
        name_val = row[criterion_col]
        if name_val is None or str(name_val).strip() == "" or str(name_val).strip().lower() == "nan":
            continue

        name = str(name_val).strip()
        max_score_raw = row[max_score_col]
        max_score = coerce_float(max_score_raw, default=-1.0)

        if max_score < 0:
            warnings.append(
                f"Row {row_num + 1}: criterion '{name}' has an invalid or missing max score and was skipped."
            )
            continue

        description = ""
        if description_col is not None:
            desc_val = row[description_col]
            if desc_val is not None and str(desc_val).strip().lower() != "nan":
                description = str(desc_val).strip()

        criteria.append(RubricCriterion(name=name, max_score=max_score, description=description))

    if not criteria:
        raise RubricParseError(
            "No valid rubric rows were found. Ensure each row has a criterion name and a "
            "numeric maximum score."
        )

    return Rubric(criteria=criteria, warnings=warnings)


def _locate_header_row(raw) -> tuple[int | None, int | None, int | None, int | None]:
    """Scan the first several rows to find the one that looks like a header.

    Returns (header_row_index, criterion_col_idx, max_score_col_idx,
    description_col_idx). Any of these may be None if not found.
    """
    max_scan_rows = min(config.RUBRIC_HEADER_SCAN_ROWS, len(raw))
    for row_idx in range(max_scan_rows):
        headers = [normalize_header(v) for v in raw.iloc[row_idx].tolist()]
        criterion_col = _find_column(headers, config.RUBRIC_CRITERION_ALIASES)
        max_score_col = _find_column(headers, config.RUBRIC_MAX_SCORE_ALIASES)
        description_col = _find_column(headers, config.RUBRIC_DESCRIPTION_ALIASES)
        if criterion_col is not None and max_score_col is not None:
            return row_idx, criterion_col, max_score_col, description_col
    return None, None, None, None


def rubric_to_prompt_text(rubric: Rubric) -> str:
    """Render the rubric as a readable text block for injection into the LLM prompt."""
    lines = ["Evaluation Rubric:"]
    for criterion in rubric.criteria:
        line = f"- {criterion.name} (Max Score: {criterion.max_score})"
        if criterion.description:
            line += f": {criterion.description}"
        lines.append(line)
    lines.append(f"\nTotal Maximum Score: {rubric.total_max_score}")
    return "\n".join(lines)
