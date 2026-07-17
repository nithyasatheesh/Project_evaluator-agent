"""
exporter.py
===========
Builds the final Excel workbook of evaluation results: one row per
student, one column per rubric criterion (fully dynamic — never
hardcoded), plus totals, percentage, grade, qualitative feedback,
strengths, improvements and parser warnings.
"""

from __future__ import annotations

import io
from pathlib import Path

from utils import EvaluationResult, Rubric, sanitize_sheet_name, setup_logger

logger = setup_logger(__name__)

# Columns that hold longer free-text content and should get wrapped text
# and a wider column width in the exported sheet.
_WRAP_COLUMNS = {
    "Language Feedback",
    "Analysis Feedback",
    "Clarity Feedback",
    "Overall Feedback",
    "Strengths",
    "Improvements",
    "Parser Warnings",
}

_LIST_JOIN_SEPARATOR = "\n"


def _build_rows(results: list[EvaluationResult], rubric: Rubric) -> list[dict]:
    """Turn EvaluationResult objects into flat dict rows for a DataFrame."""
    criterion_names = rubric.criterion_names
    rows: list[dict] = []
    for result in results:
        row: dict = {"Student": result.student}
        for name in criterion_names:
            row[name] = result.scores.get(name, 0.0)
        row["Total"] = round(result.total_score, 2)
        row["Max Score"] = round(result.max_score, 2)
        row["Percentage"] = round(result.percentage, 2)
        row["Grade"] = result.grade
        row["Language Feedback"] = result.language_feedback
        row["Analysis Feedback"] = result.analysis_feedback
        row["Clarity Feedback"] = result.clarity_feedback
        row["Overall Feedback"] = result.overall_feedback
        row["Strengths"] = _LIST_JOIN_SEPARATOR.join(f"- {s}" for s in result.strengths)
        row["Improvements"] = _LIST_JOIN_SEPARATOR.join(f"- {s}" for s in result.improvements)
        row["Parser Warnings"] = _LIST_JOIN_SEPARATOR.join(result.warnings)
        row["Evaluation Failed"] = "Yes" if result.evaluation_failed else "No"
        rows.append(row)
    return rows


def _style_workbook(buffer: io.BytesIO, sheet_name: str) -> io.BytesIO:
    """Apply header styling, column widths and text wrapping in-place."""
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    buffer.seek(0)
    workbook = load_workbook(buffer)
    sheet = workbook[sheet_name]

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    headers = [cell.value for cell in sheet[1]]
    for col_idx, header in enumerate(headers, start=1):
        cell = sheet.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_alignment

        column_letter = get_column_letter(col_idx)
        if header in _WRAP_COLUMNS:
            sheet.column_dimensions[column_letter].width = 45
        elif header == "Student":
            sheet.column_dimensions[column_letter].width = 22
        else:
            sheet.column_dimensions[column_letter].width = 16

    for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row):
        for cell in row:
            header = headers[cell.column - 1]
            if header in _WRAP_COLUMNS:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            else:
                cell.alignment = Alignment(vertical="top")

    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions

    out_buffer = io.BytesIO()
    workbook.save(out_buffer)
    out_buffer.seek(0)
    return out_buffer


def export_results_to_bytes(results: list[EvaluationResult], rubric: Rubric, sheet_name: str = "Evaluation Results") -> bytes:
    """Build the evaluation Excel workbook and return its raw bytes."""
    import pandas as pd

    if not results:
        raise ValueError("No evaluation results to export.")

    rows = _build_rows(results, rubric)
    df = pd.DataFrame(rows)

    safe_sheet_name = sanitize_sheet_name(sheet_name)
    raw_buffer = io.BytesIO()
    with pd.ExcelWriter(raw_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=safe_sheet_name)

    styled_buffer = _style_workbook(raw_buffer, safe_sheet_name)
    return styled_buffer.getvalue()


def export_results_to_file(results: list[EvaluationResult], rubric: Rubric, output_path: Path) -> Path:
    """Build the evaluation Excel workbook and write it to output_path."""
    data = export_results_to_bytes(results, rubric)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    logger.info("Wrote evaluation results workbook to '%s'", output_path)
    return output_path
