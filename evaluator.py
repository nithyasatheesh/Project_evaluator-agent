"""
evaluator.py
============
Core LLM evaluation logic. Builds a rubric-grounded prompt per student,
calls OpenAI GPT-4.1 in strict JSON mode, validates/clamps the returned
scores against the rubric, and computes total/percentage/grade.

evaluate_all() orchestrates the whole batch with a ThreadPoolExecutor so
faculty aren't waiting on students to be graded one at a time, while
still guaranteeing that one student's failure (bad submission, API
error, malformed JSON) never stops the rest of the batch.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import config
from rubric import rubric_to_prompt_text
from utils import (
    EvaluationResult,
    Rubric,
    StudentContext,
    clamp,
    coerce_float,
    compute_grade,
    setup_logger,
)

logger = setup_logger(__name__)

ProgressCallback = Callable[[int, int, str], None]

SYSTEM_PROMPT = """You are an expert, fair, and detail-oriented academic evaluator. \
You grade student assignment submissions strictly according to a provided rubric. \
You are subject-agnostic: the assignment may be Machine Learning, Deep Learning, NLP, \
GenAI, RAG, Python, Linux Shell Scripting, SQL, Power BI, Tableau, Data Engineering, or \
any other technical/coding topic. Never invent your own criteria — only use the rubric \
criteria given to you, using their exact names as JSON keys. Base every score strictly \
on evidence present in the submission context. Do not reward content that is not \
present in the submission. Do not exceed the maximum score for any criterion. \
Respond with ONLY a single valid JSON object and nothing else — no markdown, no code \
fences, no commentary before or after it."""

_JSON_SCHEMA_INSTRUCTIONS = """Return ONLY a JSON object with exactly this shape:
{{
  "scores": {{
{score_keys}
  }},
  "qualitative_feedback": {{
    "language_feedback": "string",
    "analysis_feedback": "string",
    "clarity_feedback": "string",
    "overall_feedback": "string"
  }},
  "strengths": ["string", "..."],
  "improvements": ["string", "..."]
}}

Rules:
- The keys inside "scores" must be exactly the rubric criterion names listed above, no more, no fewer.
- Each score must be a number between 0 and that criterion's max score (inclusive).
- "strengths" and "improvements" must each be a list of short, specific, actionable bullet strings.
- All feedback strings must be specific to THIS submission, not generic boilerplate.
- If the submission is empty, missing, or unreadable, give 0 for every criterion and explain why in "overall_feedback"."""


class EvaluationError(Exception):
    """Raised when a single student's evaluation cannot be completed."""


def _format_student_context(student: StudentContext) -> str:
    """Render a StudentContext into a labelled text block for the prompt."""
    sections = [
        ("Documentation / Report", student.documentation),
        ("Notebook Content", student.notebook_content),
        ("Python Code", student.python_code),
        ("Shell Scripts", student.shell_scripts),
        ("SQL", student.sql_content),
        ("CSV Data Summary", student.csv_summary),
        ("Configuration Files", student.config_files),
    ]
    parts = [f"### {title} ###\n{content}" for title, content in sections if content and content.strip()]
    if student.images:
        parts.append("### Images Submitted ###\n" + "\n".join(student.images))
    if not parts:
        return "(No readable content was found in this submission.)"
    return "\n\n".join(parts)


def build_prompt(
    problem_statement: str,
    rubric: Rubric,
    student: StudentContext,
    additional_instructions: str,
) -> list[dict[str, str]]:
    """Build the OpenAI chat messages list for evaluating one student."""
    score_keys = "\n".join(f'    "{c.name}": <score out of {c.max_score}>,' for c in rubric.criteria)
    schema_block = _JSON_SCHEMA_INSTRUCTIONS.format(score_keys=score_keys.rstrip(","))

    user_prompt = f"""=== PROBLEM STATEMENT ===
{problem_statement.strip() if problem_statement else "(No problem statement provided.)"}

=== {rubric_to_prompt_text(rubric)} ===

=== ADDITIONAL EVALUATION INSTRUCTIONS ===
{additional_instructions.strip() if additional_instructions else "(None provided.)"}

=== STUDENT SUBMISSION: {student.name} ===
{_format_student_context(student)}

=== OUTPUT FORMAT ===
{schema_block}
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _extract_json(raw_text: str) -> dict:
    """Robustly parse a JSON object out of an LLM response string."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"Could not parse JSON from LLM response: {exc}") from exc
        raise EvaluationError("LLM response did not contain a valid JSON object.")


def _call_openai_with_retries(client, messages: list[dict[str, str]]) -> str:
    """Call the OpenAI chat completions API with retry/backoff on failure."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, config.OPENAI_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                temperature=config.OPENAI_TEMPERATURE,
                max_tokens=config.OPENAI_MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"},
                timeout=config.OPENAI_REQUEST_TIMEOUT_SECONDS,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise EvaluationError("OpenAI returned an empty response.")
            return content
        except Exception as exc:  # noqa: BLE001 - broad on purpose, retried uniformly
            last_exc = exc
            logger.warning(
                "OpenAI request failed (attempt %d/%d): %s", attempt, config.OPENAI_MAX_RETRIES, exc
            )
            if attempt < config.OPENAI_MAX_RETRIES:
                time.sleep(config.OPENAI_RETRY_BACKOFF_SECONDS * attempt)
    raise EvaluationError(f"OpenAI request failed after {config.OPENAI_MAX_RETRIES} attempts: {last_exc}")


def _score_from_llm_payload(payload: dict, rubric: Rubric, result: EvaluationResult) -> None:
    """Populate result.scores from the raw LLM payload, clamped to rubric maxima."""
    raw_scores = payload.get("scores", {})
    if not isinstance(raw_scores, dict):
        raw_scores = {}
        result.warnings.append("LLM response 'scores' field was not an object; defaulted all scores to 0.")

    for criterion in rubric.criteria:
        raw_value = raw_scores.get(criterion.name)
        if raw_value is None:
            result.warnings.append(f"Missing score for criterion '{criterion.name}'; defaulted to 0.")
            result.scores[criterion.name] = 0.0
            continue
        numeric = coerce_float(raw_value, default=0.0)
        clamped = clamp(numeric, 0.0, criterion.max_score)
        if clamped != numeric:
            result.warnings.append(
                f"Score for '{criterion.name}' was out of range ({numeric}) and was clamped to {clamped}."
            )
        result.scores[criterion.name] = clamped


def _feedback_from_llm_payload(payload: dict, result: EvaluationResult) -> None:
    """Populate qualitative feedback, strengths and improvements from the payload."""
    feedback = payload.get("qualitative_feedback", {})
    if not isinstance(feedback, dict):
        feedback = {}
    result.language_feedback = str(feedback.get("language_feedback", "") or "")
    result.analysis_feedback = str(feedback.get("analysis_feedback", "") or "")
    result.clarity_feedback = str(feedback.get("clarity_feedback", "") or "")
    result.overall_feedback = str(feedback.get("overall_feedback", "") or "")

    strengths = payload.get("strengths", [])
    result.strengths = [str(s) for s in strengths] if isinstance(strengths, list) else []

    improvements = payload.get("improvements", [])
    result.improvements = [str(s) for s in improvements] if isinstance(improvements, list) else []


def evaluate_student(
    client,
    problem_statement: str,
    rubric: Rubric,
    student: StudentContext,
    additional_instructions: str,
) -> EvaluationResult:
    """Evaluate a single student's submission and return a scored EvaluationResult.

    Never raises: any failure (parsing errors, API errors, malformed
    JSON) is captured into a zero-scored EvaluationResult with
    `evaluation_failed=True` and a descriptive warning so the caller can
    keep processing the rest of the batch.
    """
    result = EvaluationResult(
        student=student.name,
        max_score=rubric.total_max_score,
        warnings=list(student.warnings),
    )

    if student.is_empty():
        result.evaluation_failed = True
        result.scores = {c.name: 0.0 for c in rubric.criteria}
        result.overall_feedback = "No readable submission content was found; this student could not be evaluated."
        result.warnings.append("Submission was empty or unreadable; scored 0 across all criteria.")
        return result

    try:
        messages = build_prompt(problem_statement, rubric, student, additional_instructions)
        raw_response = _call_openai_with_retries(client, messages)
        payload = _extract_json(raw_response)
    except EvaluationError as exc:
        logger.error("Evaluation failed for student '%s': %s", student.name, exc)
        result.evaluation_failed = True
        result.scores = {c.name: 0.0 for c in rubric.criteria}
        result.overall_feedback = f"Automated evaluation failed: {exc}"
        result.warnings.append(f"Evaluation error: {exc}")
        return result
    except Exception as exc:  # noqa: BLE001 - final safety net per student
        logger.exception("Unexpected error evaluating student '%s'", student.name)
        result.evaluation_failed = True
        result.scores = {c.name: 0.0 for c in rubric.criteria}
        result.overall_feedback = f"Automated evaluation failed unexpectedly: {exc}"
        result.warnings.append(f"Unexpected evaluation error: {exc}")
        return result

    _score_from_llm_payload(payload, rubric, result)
    _feedback_from_llm_payload(payload, result)

    result.total_score = sum(result.scores.values())
    result.percentage = (result.total_score / result.max_score * 100.0) if result.max_score > 0 else 0.0
    result.grade = compute_grade(result.percentage)
    return result


def evaluate_all(
    students: list[StudentContext],
    problem_statement: str,
    rubric: Rubric,
    additional_instructions: str,
    api_key: str,
    max_workers: int = config.DEFAULT_MAX_WORKERS,
    progress_callback: Optional[ProgressCallback] = None,
) -> list[EvaluationResult]:
    """Evaluate every student in parallel and return results in input order.

    A single shared OpenAI client is reused across worker threads (the
    official OpenAI Python client is thread-safe for this usage pattern).
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    total = len(students)
    results: list[Optional[EvaluationResult]] = [None] * total
    completed = 0

    with ThreadPoolExecutor(max_workers=max(config.MIN_MAX_WORKERS, min(max_workers, config.MAX_MAX_WORKERS))) as pool:
        future_to_index = {
            pool.submit(
                evaluate_student, client, problem_statement, rubric, student, additional_instructions
            ): idx
            for idx, student in enumerate(students)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            student_name = students[idx].name
            try:
                results[idx] = future.result()
            except Exception as exc:  # noqa: BLE001 - absolute last resort per student
                logger.exception("Evaluation task crashed for student '%s'", student_name)
                results[idx] = EvaluationResult(
                    student=student_name,
                    max_score=rubric.total_max_score,
                    scores={c.name: 0.0 for c in rubric.criteria},
                    evaluation_failed=True,
                    overall_feedback=f"Evaluation task crashed: {exc}",
                    warnings=[f"Evaluation task crashed: {exc}"],
                )
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total, student_name)

    return [r for r in results if r is not None]
