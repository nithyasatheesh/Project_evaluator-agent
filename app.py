import os
import zipfile
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# ------------------------------
# CONFIG
# ------------------------------

st.set_page_config(page_title="Project Evaluation Agent", layout="wide")

st.title("📊 Project Evaluation Agent")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password"
)

model_name = st.sidebar.selectbox(
    "Model",
    [
        "gpt-4.1",
        "gpt-4o",
        "gpt-4.1-mini"
    ]
)

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# ------------------------------
# Helpers
# ------------------------------

def read_text_file(file):
    return file.read().decode(errors="ignore")


def save_uploaded_file(uploaded_file, folder):
    path = os.path.join(folder, uploaded_file.name)

    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return path


def extract_zip(zip_path, output_folder):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_folder)


def collect_submission_dirs(root):

    folders = []

    for item in Path(root).iterdir():

        if item.is_dir():
            folders.append(item)

    if folders:
        return folders

    return [Path(root)]


def load_html_text(folder):

    text = ""

    for file in Path(folder).rglob("*"):

        if file.suffix.lower() in [".html", ".htm", ".txt", ".md", ".py", ".js", ".css"]:
            try:
                text += "\n\n"
                text += file.read_text(errors="ignore")
            except:
                pass

    return text[:60000]


def rubric_to_text(df):

    return df.to_markdown(index=False)


def evaluate_submission(
        llm,
        problem_statement,
        rubric,
        prompt_text,
        submission_text
):

    user_prompt = f"""
You are an expert hackathon evaluator.

Problem Statement

{problem_statement}

Rubric

{rubric}

Additional Instructions

{prompt_text}

Submission

{submission_text}

Evaluate the submission.

Return EXACTLY in markdown.

# Overall Score

## Criterion Scores

| Criterion | Score | Feedback |

## Strengths

## Weaknesses

## Suggestions

"""

    response = llm.invoke([HumanMessage(content=user_prompt)])

    return response.content


# ------------------------------
# Uploads
# ------------------------------

problem_file = st.file_uploader(
    "Problem Statement",
    type=[
        "txt",
        "pdf",
        "docx"
    ]
)

rubric_file = st.file_uploader(
    "Rubric Excel",
    type=[
        "xlsx",
        "xls"
    ]
)

prompt_file = st.file_uploader(
    "Prompt (optional)",
    type=[
        "txt",
        "md"
    ]
)

dataset_file = st.file_uploader(
    "Dataset (optional)",
    type=[
        "zip",
        "csv",
        "xlsx"
    ]
)

submission_files = st.file_uploader(
    "Student Submissions",
    type=[
        "zip",
        "html",
        "htm"
    ],
    accept_multiple_files=True
)

# ------------------------------
# Run
# ------------------------------

if st.button("Evaluate Projects"):

    if not api_key:
        st.error("Enter API key.")
        st.stop()

    if problem_file is None:
        st.error("Upload problem statement.")
        st.stop()

    if rubric_file is None:
        st.error("Upload rubric.")
        st.stop()

    llm = ChatOpenAI(
        model=model_name,
        temperature=0
    )

    temp_dir = tempfile.mkdtemp()

    # --------------------------
    # Problem Statement
    # --------------------------

    problem_path = save_uploaded_file(problem_file, temp_dir)

    if problem_path.endswith(".txt"):
        problem_statement = open(problem_path, encoding="utf8").read()
    else:
        problem_statement = f"Problem statement file: {problem_file.name}"

    # --------------------------
    # Prompt
    # --------------------------

    prompt_text = ""

    if prompt_file:
        prompt_path = save_uploaded_file(prompt_file, temp_dir)
        prompt_text = open(prompt_path, encoding="utf8").read()

    # --------------------------
    # Rubric
    # --------------------------

    rubric_df = pd.read_excel(rubric_file)

    rubric_text = rubric_to_text(rubric_df)

    # --------------------------
    # Submission Extraction
    # --------------------------

    submissions_root = os.path.join(temp_dir, "subs")

    os.makedirs(submissions_root, exist_ok=True)

    for f in submission_files:

        path = save_uploaded_file(f, submissions_root)

        if path.endswith(".zip"):

            out = os.path.join(
                submissions_root,
                Path(path).stem
            )

            os.makedirs(out, exist_ok=True)

            extract_zip(path, out)

    submission_dirs = []

    for item in Path(submissions_root).iterdir():

        if item.is_dir():

            submission_dirs.extend(
                collect_submission_dirs(item)
            )

    if not submission_dirs:

        submission_dirs = [
            Path(submissions_root)
        ]

    # --------------------------
    # Evaluate
    # --------------------------

    results = []

    progress = st.progress(0)

    for i, folder in enumerate(submission_dirs):

        submission_text = load_html_text(folder)

        result = evaluate_submission(
            llm,
            problem_statement,
            rubric_text,
            prompt_text,
            submission_text
        )

        results.append(
            {
                "Team": folder.name,
                "Evaluation": result
            }
        )

        progress.progress(
            (i + 1) / len(submission_dirs)
        )

    df = pd.DataFrame(results)

    st.success("Completed!")

    st.dataframe(df)

    excel_path = os.path.join(temp_dir, "evaluation.xlsx")

    df.to_excel(excel_path, index=False)

    with open(excel_path, "rb") as f:

        st.download_button(
            "Download Excel",
            f,
            file_name="evaluation.xlsx"
        )

    shutil.rmtree(temp_dir, ignore_errors=True)
