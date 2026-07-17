"""
parser.py
=========
Master ZIP parser. Given a single ZIP uploaded by faculty containing one
sub-folder (or file, or nested ZIP) per student, this module:

  1. Safely extracts the master ZIP to a temporary working directory.
  2. Recursively discovers and extracts any nested ZIPs (e.g. a student
     who submitted their work as `submission.zip` inside their folder).
  3. Detects every student "root" (a folder or a single loose file).
  4. Walks each student root, dispatches every supported file to the
     matching reader in readers.py, and assembles a StudentContext.
  5. Collects human-readable warnings for anything that could not be
     processed (corrupted archives, unreadable files, empty
     submissions) without ever aborting the whole batch.
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

import config
from readers import ReaderError, describe_image, read_file_content
from utils import StudentContext, join_nonempty, setup_logger, truncate_text

logger = setup_logger(__name__)

# Maps each supported extension set to the StudentContext attribute it
# should be appended to. Order matters only for readability.
_BUCKET_MAP: list[tuple[set[str], str]] = [
    (config.DOC_EXTENSIONS | config.HTML_EXTENSIONS, "documentation"),
    (config.NOTEBOOK_EXTENSIONS, "notebook_content"),
    (config.PYTHON_EXTENSIONS, "python_code"),
    (config.SHELL_EXTENSIONS, "shell_scripts"),
    (config.SQL_EXTENSIONS, "sql_content"),
    (config.CSV_EXTENSIONS, "csv_summary"),
    (config.CONFIG_EXTENSIONS, "config_files"),
]


class MasterZipParseError(Exception):
    """Raised when the master ZIP itself cannot be opened at all."""


def create_work_dir() -> Path:
    """Create and return a fresh temporary working directory."""
    return Path(tempfile.mkdtemp(prefix=config.TEMP_DIR_PREFIX))


def cleanup_work_dir(work_dir: Path) -> None:
    """Best-effort removal of the temporary working directory."""
    shutil.rmtree(work_dir, ignore_errors=True)


def _is_safe_member(name: str) -> bool:
    """Reject zip members that could escape the extraction directory."""
    normalized = Path(name)
    if normalized.is_absolute():
        return False
    return ".." not in normalized.parts


def extract_zip(zip_path: Path, dest_dir: Path, warnings: list[str]) -> None:
    """Extract a ZIP file member-by-member, tolerating individual bad entries.

    Adds a warning (rather than raising) if the archive is corrupted or
    partially unreadable, so callers can continue processing siblings.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            for member in archive.infolist():
                if not _is_safe_member(member.filename):
                    warnings.append(f"Skipped unsafe path in archive '{zip_path.name}': {member.filename}")
                    continue
                try:
                    archive.extract(member, dest_dir)
                except (OSError, zipfile.BadZipFile, NotImplementedError) as exc:
                    warnings.append(
                        f"Could not extract '{member.filename}' from '{zip_path.name}': {exc}"
                    )
    except zipfile.BadZipFile as exc:
        warnings.append(f"Corrupted ZIP archive '{zip_path.name}': {exc}")
    except OSError as exc:
        warnings.append(f"Could not open ZIP archive '{zip_path.name}': {exc}")


def _extract_nested_zips(root: Path, warnings: list[str]) -> None:
    """Iteratively find and extract nested ZIPs up to a maximum depth.

    Each discovered ZIP is extracted into a sibling directory named after
    the archive (without extension) and then removed, so subsequent
    passes only see freshly extracted content (which may itself contain
    further nested ZIPs, hence the depth-bounded loop).
    """
    for depth in range(config.MAX_ZIP_EXTRACTION_DEPTH):
        nested_zips = [p for p in root.rglob("*.zip") if p.is_file()]
        if not nested_zips:
            return
        for zip_path in nested_zips:
            target_dir = zip_path.parent / f"{zip_path.stem}_extracted"
            suffix = 1
            while target_dir.exists():
                target_dir = zip_path.parent / f"{zip_path.stem}_extracted_{suffix}"
                suffix += 1
            extract_zip(zip_path, target_dir, warnings)
            try:
                zip_path.unlink()
            except OSError as exc:
                warnings.append(f"Could not remove nested archive '{zip_path.name}' after extraction: {exc}")
    remaining = [p for p in root.rglob("*.zip") if p.is_file()]
    if remaining:
        warnings.append(
            f"Reached maximum ZIP nesting depth ({config.MAX_ZIP_EXTRACTION_DEPTH}); "
            f"{len(remaining)} archive(s) left unextracted."
        )


def _find_effective_root(root: Path) -> Path:
    """Descend through single-child wrapper directories.

    Many ZIP tools wrap all content in one top-level folder matching the
    archive name. If that is the only thing at `root`, treat its
    contents as the true student list instead of creating a single
    "student" named after the wrapper folder.
    """
    current = root
    while True:
        children = [c for c in current.iterdir() if not c.name.startswith("__MACOSX")]
        if len(children) == 1 and children[0].is_dir():
            current = children[0]
            continue
        return current


def _discover_student_roots(root: Path) -> list[tuple[str, Path]]:
    """Return a list of (student_name, path) pairs at the effective root.

    A directory becomes one student. A loose file directly at the
    effective root also becomes one student (named after its stem),
    which supports the "Student_D/assignment.html placed directly at
    top level" style of submission.
    """
    effective_root = _find_effective_root(root)
    entries = sorted(
        (e for e in effective_root.iterdir() if not e.name.startswith("__MACOSX") and e.name != ".DS_Store"),
        key=lambda p: p.name.lower(),
    )

    students: list[tuple[str, Path]] = []
    for entry in entries:
        if entry.is_dir():
            students.append((entry.name, entry))
        elif entry.is_file() and entry.suffix.lower() != ".zip":
            students.append((entry.stem, entry))
    return students


def _iter_student_files(student_path: Path) -> list[Path]:
    """Return every regular file belonging to a student's submission."""
    if student_path.is_file():
        return [student_path]
    return [p for p in student_path.rglob("*") if p.is_file() and not p.name.startswith(".")]


def _build_student_context(name: str, student_path: Path) -> StudentContext:
    """Walk a student's files and assemble a fully populated StudentContext."""
    context = StudentContext(name=name)
    buckets: dict[str, list[str]] = {attr: [] for _, attr in _BUCKET_MAP}

    files = _iter_student_files(student_path)
    context.file_count = len(files)

    if not files:
        context.warnings.append("No files found for this student.")
        return context

    for file_path in files:
        suffix = file_path.suffix.lower()
        try:
            rel_name = file_path.name if student_path.is_file() else str(file_path.relative_to(student_path))
        except ValueError:
            rel_name = file_path.name

        if suffix in config.ARCHIVE_EXTENSIONS:
            context.warnings.append(f"Archive '{rel_name}' could not be fully extracted and was skipped.")
            continue

        if suffix in config.IMAGE_EXTENSIONS:
            try:
                context.images.append(describe_image(file_path))
            except ReaderError as exc:
                context.warnings.append(str(exc))
            continue

        target_attr = next((attr for exts, attr in _BUCKET_MAP if suffix in exts), None)
        if target_attr is None:
            # Unsupported extension: silently ignored per spec.
            continue

        try:
            content = read_file_content(file_path)
            buckets[target_attr].append(f"### {rel_name} ###\n{content}")
        except ReaderError as exc:
            context.warnings.append(f"Unreadable file '{rel_name}': {exc}")
        except Exception as exc:  # noqa: BLE001 - never let one bad file kill the batch
            context.warnings.append(f"Unexpected error reading '{rel_name}': {exc}")

    for attr, parts in buckets.items():
        setattr(context, attr, truncate_text(join_nonempty(parts)))

    if context.is_empty():
        context.warnings.append(
            "No supported/readable content could be extracted from this student's submission."
        )

    return context


def parse_master_zip(master_zip_path: Path) -> list[StudentContext]:
    """Parse a master ZIP of student submissions into StudentContext objects.

    Never raises for per-student problems; those are captured as
    warnings on the relevant StudentContext. Raises MasterZipParseError
    only if the master ZIP itself is completely unusable.
    """
    work_dir = create_work_dir()
    global_warnings: list[str] = []
    try:
        extract_zip(master_zip_path, work_dir, global_warnings)
        if not any(work_dir.iterdir()):
            raise MasterZipParseError(
                "The master ZIP could not be extracted or contains no files. "
                + " ".join(global_warnings)
            )

        _extract_nested_zips(work_dir, global_warnings)
        student_roots = _discover_student_roots(work_dir)

        if not student_roots:
            raise MasterZipParseError(
                "No student folders or files were found inside the master ZIP."
            )

        students: list[StudentContext] = []
        for name, path in student_roots:
            try:
                student_context = _build_student_context(name, path)
            except Exception as exc:  # noqa: BLE001 - isolate per-student failures
                logger.exception("Failed to build context for student '%s'", name)
                student_context = StudentContext(name=name, warnings=[f"Failed to parse submission: {exc}"])
            students.append(student_context)

        # Batch-level warnings (e.g. corrupted nested archives found while
        # extracting the master ZIP) apply to the whole run, not one
        # student in particular, so surface them on every row.
        for warning in global_warnings:
            batch_warning = f"[Batch] {warning}"
            for student_context in students:
                if batch_warning not in student_context.warnings:
                    student_context.warnings.append(batch_warning)

        return students
    finally:
        cleanup_work_dir(work_dir)
