"""
app.py
======
Streamlit front-end for the AI Assignment Evaluator.

Faculty upload a problem statement, an Excel rubric and one master ZIP
containing every student's submission. The app parses the ZIP into
per-student context, sends each submission to OpenAI GPT-4.1 for
rubric-based scoring (in parallel, with graceful per-student failure
handling), and produces a downloadable Excel workbook of results.
"""

from __future__ import annotations

import shutil
import tempfile
import traceback
from pathlib import Path

import streamlit as st

import config
from evaluator import evaluate_all
from exporter import export_results_to_bytes
from parser import MasterZipParseError, parse_master_zip
from readers import ReaderError, read_file_content
from rubric import RubricParseError, parse_rubric
from utils import setup_logger

logger = setup_logger(__name__)

st.set_page_config(page_title=config.APP_NAME, page_icon="📝", layout="wide")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _save_uploaded_file(uploaded_file, dest_dir: Path) -> Path:
    """Persist a Streamlit UploadedFile to disk and return its Path."""
    dest_path = dest_dir / uploaded_file.name
    with dest_path.open("wb") as fh:
        fh.write(uploaded_file.getbuffer())
    return dest_path


def _read_problem_statement(uploaded_file, work_dir: Path) -> str:
    """Save and extract text content from the uploaded problem statement file."""
    path = _save_uploaded_file(uploaded_file, work_dir)
    try:
        return read_file_content(path)
    except ReaderError as exc:
        raise RuntimeError(f"Could not read problem statement file: {exc}") from exc


def _reset_results() -> None:
    st.session_state.pop("excel_bytes", None)
    st.session_state.pop("results_summary", None)
    st.session_state.pop("output_filename", None)


# --------------------------------------------------------------------------
# Sidebar — configuration
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    default_key = config.get_openai_api_key() or ""
    api_key_input = st.text_input(
        "OpenAI API Key",
        value=default_key,
        type="password",
        help="Reads from the OPENAI_API_KEY environment variable by default. "
        "Paste a key here to override it for this session only.",
    )

    max_workers = st.slider(
        "Parallel evaluations",
        min_value=config.MIN_MAX_WORKERS,
        max_value=config.MAX_MAX_WORKERS,
        value=config.DEFAULT_MAX_WORKERS,
        help="Number of student submissions evaluated concurrently. Lower this if you hit "
        "OpenAI rate limits.",
    )

    st.caption(f"Model: `{config.OPENAI_MODEL}`")
    st.caption(f"Version: {config.APP_VERSION}")


# --------------------------------------------------------------------------
# Main page
# --------------------------------------------------------------------------
st.title(f"📝 {config.APP_NAME}")
st.markdown(
    "Automatically evaluate an entire batch of student assignment submissions against "
    "a rubric using OpenAI GPT-4.1 — works for ML, DL, NLP, GenAI, RAG, Python, SQL, "
    "Power BI, Tableau, Data Engineering, or any other coding assignment."
)

col1, col2 = st.columns(2)

with col1:
    problem_statement_file = st.file_uploader(
        "1. Problem Statement (PDF or DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=False,
        on_change=_reset_results,
    )

with col2:
    rubric_file = st.file_uploader(
        "2. Evaluation Rubric (Excel)",
        type=["xlsx", "xls"],
        accept_multiple_files=False,
        on_change=_reset_results,
    )

master_zip_file = st.file_uploader(
    "3. Master ZIP of All Student Submissions",
    type=["zip"],
    accept_multiple_files=False,
    on_change=_reset_results,
    help="One ZIP containing a folder (or file, or nested ZIP) per student. "
    "Do not upload individual student submissions separately.",
)

additional_instructions = st.text_area(
    "4. Additional Evaluation Instructions (optional)",
    placeholder="e.g. Penalize hardcoded credentials heavily. Award bonus consideration for "
    "clear documentation. Treat missing requirements.txt as a minor deduction only.",
    height=100,
)

evaluate_clicked = st.button("🚀 Evaluate Submissions", type="primary", use_container_width=True)


# --------------------------------------------------------------------------
# Evaluation flow
# --------------------------------------------------------------------------
if evaluate_clicked:
    _reset_results()

    if not api_key_input:
        st.error("Please provide an OpenAI API key in the sidebar.")
        st.stop()
    if not problem_statement_file:
        st.error("Please upload a problem statement (PDF or DOCX).")
        st.stop()
    if not rubric_file:
        st.error("Please upload an evaluation rubric (Excel).")
        st.stop()
    if not master_zip_file:
        st.error("Please upload the master ZIP of student submissions.")
        st.stop()

    work_dir = Path(tempfile.mkdtemp(prefix=config.TEMP_DIR_PREFIX))
    status_box = st.status("Starting evaluation...", expanded=True)

    try:
        # --- Step 1: Problem statement -----------------------------------
        status_box.write("Reading problem statement...")
        problem_statement_text = _read_problem_statement(problem_statement_file, work_dir)

        # --- Step 2: Rubric -------------------------------------------------
        status_box.write("Parsing rubric...")
        rubric = parse_rubric(rubric_file)
        if rubric.warnings:
            for warning in rubric.warnings:
                status_box.write(f"⚠️ Rubric warning: {warning}")
        status_box.write(
            f"Rubric loaded with {len(rubric.criteria)} criteria "
            f"(total max score: {rubric.total_max_score})."
        )

        # --- Step 3: Master ZIP ----------------------------------------
        status_box.write("Extracting master ZIP and detecting students...")
        master_zip_path = _save_uploaded_file(master_zip_file, work_dir)
        students = parse_master_zip(master_zip_path)
        status_box.write(f"Detected {len(students)} student submission(s).")

        empty_students = [s.name for s in students if s.is_empty()]
        if empty_students:
            status_box.write(
                f"⚠️ {len(empty_students)} submission(s) had no readable content: "
                f"{', '.join(empty_students)}"
            )

        # --- Step 4: Evaluate --------------------------------------------
        status_box.write(f"Evaluating {len(students)} submission(s) with {config.OPENAI_MODEL}...")
        progress_bar = st.progress(0.0)
        progress_text = st.empty()

        def _on_progress(completed: int, total: int, student_name: str) -> None:
            progress_bar.progress(completed / total if total else 1.0)
            progress_text.text(f"Evaluated {completed}/{total}: {student_name}")

        results = evaluate_all(
            students=students,
            problem_statement=problem_statement_text,
            rubric=rubric,
            additional_instructions=additional_instructions,
            api_key=api_key_input,
            max_workers=max_workers,
            progress_callback=_on_progress,
        )

        failed_count = sum(1 for r in results if r.evaluation_failed)
        if failed_count:
            status_box.write(f"⚠️ {failed_count} student(s) could not be automatically evaluated.")

        # --- Step 5: Export ------------------------------------------------
        status_box.write("Building Excel report...")
        excel_bytes = export_results_to_bytes(results, rubric)

        output_filename = (
            f"{Path(master_zip_file.name).stem}_evaluation_results.xlsx"
            if master_zip_file.name
            else config.DEFAULT_OUTPUT_FILENAME
        )

        st.session_state["excel_bytes"] = excel_bytes
        st.session_state["output_filename"] = output_filename
        st.session_state["results_summary"] = [
            {
                "Student": r.student,
                "Total": round(r.total_score, 2),
                "Max": round(r.max_score, 2),
                "Percentage": round(r.percentage, 2),
                "Grade": r.grade,
                "Failed": r.evaluation_failed,
            }
            for r in results
        ]

        status_box.update(label="Evaluation complete!", state="complete", expanded=False)

    except (RubricParseError, MasterZipParseError, RuntimeError) as exc:
        status_box.update(label="Evaluation failed", state="error")
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001 - surface unexpected errors without a crash
        status_box.update(label="Evaluation failed", state="error")
        logger.exception("Unexpected error during evaluation")
        st.error(f"An unexpected error occurred: {exc}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# Results display / download (persists across reruns via session_state)
# --------------------------------------------------------------------------
if "excel_bytes" in st.session_state:
    st.subheader("Results Summary")
    st.dataframe(st.session_state["results_summary"], use_container_width=True)

    st.download_button(
        label="⬇️ Download Full Evaluation Report (Excel)",
        data=st.session_state["excel_bytes"],
        file_name=st.session_state["output_filename"],
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
