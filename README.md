# AI Assignment Evaluator

A Streamlit application that automatically evaluates an entire batch of student
assignment submissions against a faculty-supplied rubric, using OpenAI GPT-4.1.

It is **subject-agnostic** — the same code evaluates Machine Learning, Deep
Learning, NLP, GenAI, RAG, Python, Linux Shell Scripting, SQL, Power BI,
Tableau, Data Engineering, or any other coding assignment, because it never
hardcodes rubric criteria or assignment-specific logic. Everything is driven
by whatever problem statement and rubric the faculty member uploads.

---

## Features

- **One master ZIP upload** — faculty upload a single ZIP containing every
  student's submission (one folder, file, or nested ZIP per student).
  Individual submissions never need to be uploaded separately.
- **Generic rubric parsing** — reads any Excel rubric, matching common
  header aliases (`Criterion` / `Criteria` / `Evaluation Criteria`,
  `Max Score` / `Max Marks` / `Marks` / `Weight`, `Description` /
  `Evaluation Parameters` / `Details`). No criteria are hardcoded.
- **Multi-format submission parsing** — `.zip` (incl. nested), `.ipynb`,
  `.html`, `.pdf`, `.docx`, `.txt`, `.md`, `.py`, `.sh`, `.csv`, `.sql`,
  `.json`, `.yaml` / `.yml`, `.png`, `.jpg`, `.jpeg`.
- **Strict, rubric-grounded LLM evaluation** — GPT-4.1 is prompted with the
  problem statement, rubric, submission context and any additional
  instructions, and must return structured JSON. Scores are clamped so
  they can never exceed a criterion's maximum.
- **Resilient batch processing** — corrupted archives, unreadable files, or
  a single student's failed evaluation are captured as warnings and never
  stop the rest of the batch. Evaluations run in parallel via a
  `ThreadPoolExecutor`.
- **One-click Excel report** — one row per student, one column per rubric
  criterion, totals, percentage, grade, qualitative feedback, strengths,
  improvements and parser warnings.

---

## Installation

Requires **Python 3.11+**.

```bash
cd EvaluatorAgent
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### OpenAI API Key

Set your key as an environment variable before launching (recommended):

```bash
export OPENAI_API_KEY="sk-..."          # Windows (PowerShell): $env:OPENAI_API_KEY="sk-..."
```

Alternatively, paste the key directly into the sidebar of the running app —
it will be used for that session only and is never written to disk.

---

## How to Run

```bash
streamlit run app.py
```

Then, in the browser tab that opens:

1. Upload the **Problem Statement** (PDF or DOCX).
2. Upload the **Evaluation Rubric** (Excel — `.xlsx`).
3. Upload the **Master ZIP** containing every student's submission.
4. Optionally add **Additional Evaluation Instructions**.
5. Click **Evaluate Submissions** and watch the progress bar / status log.
6. Download the generated Excel report.

---

## Folder Structure

```
EvaluatorAgent/
├── app.py           # Streamlit UI and evaluation orchestration
├── config.py         # All tunable constants (model, extensions, aliases, grading scale)
├── parser.py          # Master ZIP parsing, student detection, nested ZIP extraction
├── readers.py         # Per-file-format content readers (pdf, docx, ipynb, html, csv, ...)
├── rubric.py           # Generic Excel rubric parser (alias-based column detection)
├── evaluator.py         # Prompt building, OpenAI calls, scoring, ThreadPoolExecutor batch runner
├── exporter.py           # Excel report generation
├── utils.py                # Shared dataclasses, logging, text/number helpers
├── requirements.txt
└── README.md
```

### Module responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Every constant: model name, file extensions, rubric column aliases, grading scale, concurrency limits. Nothing else imports environment variables directly. |
| `utils.py` | `StudentContext`, `RubricCriterion`, `Rubric`, `EvaluationResult` dataclasses; logging setup; text/number helper functions. |
| `readers.py` | One function per supported file format, all raising a common `ReaderError` on failure. |
| `parser.py` | Extracts the master ZIP (and any nested ZIPs), discovers student roots, and builds a `StudentContext` per student by dispatching each file to `readers.py`. |
| `rubric.py` | Reads the uploaded Excel rubric and maps arbitrary column headers to canonical fields via alias matching. |
| `evaluator.py` | Builds the LLM prompt from problem statement + rubric + submission context, calls OpenAI GPT-4.1 in JSON mode with retries, validates/clamps scores, and parallelizes the whole batch. |
| `exporter.py` | Converts `EvaluationResult` objects into a formatted Excel workbook. |
| `app.py` | Streamlit UI: uploads, progress bar, status messages, results table, download button. |

---

## Supported Submission File Types

| Type | Extensions | Extracted into |
|---|---|---|
| Documentation | `.pdf`, `.docx`, `.txt`, `.md`, `.html`, `.htm` | Documentation |
| Notebook | `.ipynb` | Notebook Content |
| Python | `.py` | Python Code |
| Shell | `.sh`, `.bash` | Shell Scripts |
| SQL | `.sql` | SQL |
| Tabular | `.csv` | CSV Summary |
| Config | `.json`, `.yaml`, `.yml` | Configuration Files |
| Images | `.png`, `.jpg`, `.jpeg` | Images list (filename + size only) |
| Archive | `.zip` | Recursively extracted (up to 5 levels deep) |

Unsupported file types encountered inside a submission are silently ignored;
unreadable or corrupted files generate a warning that is surfaced in the
final report's "Parser Warnings" column instead of stopping the batch.

---

## Output Excel Columns

`Student` · one column per rubric criterion, headed `"<Criterion Name> (Max
<N>)"` with that student's achieved score in the cell · `Total` (a live
`=SUM(...)` formula over the criterion columns) · `Max Score` · `Percentage`
(a live formula, `Total / Max Score`) · `Grade` · `Language Feedback` ·
`Analysis Feedback` · `Clarity Feedback` · `Overall Feedback` · `Areas of
Strength` · `Areas of Improvement` · `Parser Warnings` · `Evaluation Failed`

`Total` and `Percentage` are real Excel formulas, not static numbers — if a
faculty member manually tweaks a criterion score in the sheet, both
recalculate automatically.

---

## Screenshots

> _Add screenshots here after running the app locally._

`docs/screenshot-upload.png` — Upload screen with the four inputs.

`docs/screenshot-progress.png` — Evaluation in progress with the status log
and progress bar.

`docs/screenshot-results.png` — Results summary table and Excel download
button.
