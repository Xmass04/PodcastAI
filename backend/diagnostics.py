from __future__ import annotations

import json
import re
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FOLDER = PROJECT_ROOT / "logs"
LATEST_REPORT_PATH = LOG_FOLDER / "latest.json"


@dataclass
class DiagnosticReport:
    run_id: str
    stage: str
    script_name: str
    status: str
    started_at: str
    finished_at: str
    duration_seconds: float
    return_code: int | None = None
    attempt: int = 1
    error_type: str | None = None
    error_message: str | None = None
    source_file: str | None = None
    source_function: str | None = None
    source_line: int | None = None
    likely_cause: str | None = None
    suggested_fix: str | None = None
    retryable: bool = False
    stdout: str = ""
    stderr: str = ""
    traceback_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def make_run_id() -> str:
    timestamp = utc_now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def extract_traceback_location(
    text: str,
) -> tuple[str | None, str | None, int | None]:
    """
    Extract the deepest Python traceback location from stderr/output.
    """

    matches = list(
        re.finditer(
            r'File "([^"]+)", line (\d+), in ([^\n]+)',
            text,
        )
    )

    if not matches:
        return None, None, None

    match = matches[-1]

    return (
        match.group(1),
        match.group(3).strip(),
        int(match.group(2)),
    )


def classify_issue(
    text: str,
) -> tuple[
    str | None,
    str | None,
    str | None,
    bool,
]:
    """
    Convert common technical failures into useful developer guidance.

    Returns:
    error_type, likely_cause, suggested_fix, retryable
    """

    lowered = text.lower()

    rules = [
        (
            (
                "rate limit",
                "429",
                "too many requests",
            ),
            "RateLimitError",
            "The AI provider temporarily limited the number of requests.",
            (
                "Wait briefly and retry. Reduce concurrent audio requests "
                "if this happens repeatedly."
            ),
            True,
        ),
        (
            (
                "timed out",
                "timeout",
                "connection reset",
                "connection aborted",
                "temporarily unavailable",
                "503",
                "502",
            ),
            "TemporaryConnectionError",
            "A temporary network or AI-service interruption occurred.",
            "Retry the task. The diagnostic runner can retry this automatically.",
            True,
        ),
        (
            (
                "access is denied",
                "permissionerror",
                "permission denied",
                "being used by another process",
                "file is open",
            ),
            "PermissionError",
            "Windows or another application has locked a file or folder.",
            (
                "Close the audio player, File Explorer preview, or any program "
                "using the file, then retry."
            ),
            False,
        ),
        (
            (
                "openai_api_key",
                "api key",
                "authentication",
                "401",
            ),
            "AuthenticationError",
            "The OpenAI API key is missing, invalid, or unavailable.",
            (
                "Check OPENAI_API_KEY in .env locally or in the hosted "
                "platform's secrets."
            ),
            False,
        ),
        (
            (
                "study notes not found",
                "study_notes.txt was not created",
                "study-notes file is empty",
            ),
            "MissingDependency",
            "Podcast generation needs study notes, but they are missing or empty.",
            "Generate fresh study notes before creating the podcast script.",
            False,
        ),
        (
            (
                "podcast script was not found",
                "podcast_script.txt was not created",
                "podcast script is empty",
            ),
            "MissingDependency",
            "Podcast audio needs a valid podcast script first.",
            "Generate the podcast script before audio.",
            False,
        ),
        (
            (
                "no selectable text",
                "no readable text",
                "extracted-text file is empty",
                "no extracted text",
            ),
            "ExtractionError",
            "The PDF or image did not produce usable text.",
            (
                "Use a clearer image, a text-based PDF, or enable OCR for "
                "scanned pages."
            ),
            False,
        ),
        (
            (
                "jsondecodeerror",
                "invalid json",
                "could not be loaded",
            ),
            "InvalidJSON",
            "A generator returned or saved malformed JSON.",
            (
                "Keep the raw response in the report and regenerate the "
                "affected feature."
            ),
            True,
        ),
        (
            (
                "insufficient_quota",
                "quota",
                "billing",
            ),
            "QuotaError",
            "The API account does not currently have enough available credit.",
            "Check API billing and usage limits before retrying.",
            False,
        ),
        (
            (
                "file not found",
                "filenotfounderror",
                "missing backend file",
            ),
            "FileNotFoundError",
            "A required source or generated file is missing.",
            "Check the reported path and generate its dependency first.",
            False,
        ),
    ]

    for (
        keywords,
        error_type,
        likely_cause,
        suggested_fix,
        retryable,
    ) in rules:
        if any(keyword in lowered for keyword in keywords):
            return (
                error_type,
                likely_cause,
                suggested_fix,
                retryable,
            )

    error_match = re.search(
        r"([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)):\s*([^\n]+)",
        text,
    )

    error_type = (
        error_match.group(1)
        if error_match
        else "BackendTaskError"
    )

    likely_cause = (
        error_match.group(2).strip()
        if error_match
        else "The backend task reported a failure."
    )

    return (
        error_type,
        likely_cause,
        (
            "Open the full diagnostic report and inspect the deepest "
            "traceback location."
        ),
        False,
    )


def save_report(
    report: DiagnosticReport,
) -> Path:
    LOG_FOLDER.mkdir(parents=True, exist_ok=True)

    day_folder = (
        LOG_FOLDER
        / utc_now().strftime("%Y-%m-%d")
    )

    day_folder.mkdir(parents=True, exist_ok=True)

    report_path = (
        day_folder
        / f"{report.run_id}_{report.script_name}.json"
    )

    content = json.dumps(
        asdict(report),
        indent=2,
        ensure_ascii=False,
    )

    report_path.write_text(
        content,
        encoding="utf-8",
    )

    LATEST_REPORT_PATH.write_text(
        content,
        encoding="utf-8",
    )

    return report_path


def record_subprocess_result(
    *,
    stage: str,
    script_name: str,
    started_at: datetime,
    finished_at: datetime,
    return_code: int,
    stdout: str,
    stderr: str,
    attempt: int,
    metadata: dict[str, Any] | None = None,
) -> DiagnosticReport:
    combined = "\n".join(
        part for part in (stdout, stderr) if part
    )

    lowered = combined.lower()

    printed_failure = (
        "generation failed" in lowered
        or "analysis failed" in lowered
        or "\nerror:" in lowered
        or lowered.startswith("error:")
    )

    success = (
        return_code == 0
        and not printed_failure
    )

    source_file, source_function, source_line = (
        extract_traceback_location(combined)
    )

    error_type = None
    likely_cause = None
    suggested_fix = None
    retryable = False
    error_message = None

    if not success:
        (
            error_type,
            likely_cause,
            suggested_fix,
            retryable,
        ) = classify_issue(combined)

        error_message = (
            likely_cause
            or "The task failed without an explanation."
        )

    report = DiagnosticReport(
        run_id=make_run_id(),
        stage=stage,
        script_name=script_name,
        status="success" if success else "failed",
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        duration_seconds=round(
            (finished_at - started_at).total_seconds(),
            3,
        ),
        return_code=return_code,
        attempt=attempt,
        error_type=error_type,
        error_message=error_message,
        source_file=source_file,
        source_function=source_function,
        source_line=source_line,
        likely_cause=likely_cause,
        suggested_fix=suggested_fix,
        retryable=retryable,
        stdout=stdout,
        stderr=stderr,
        metadata=metadata or {},
    )

    save_report(report)

    return report


def record_python_exception(
    *,
    stage: str,
    script_name: str,
    started_at: datetime,
    error: BaseException,
    attempt: int = 1,
    metadata: dict[str, Any] | None = None,
) -> DiagnosticReport:
    finished_at = utc_now()
    traceback_text = "".join(
        traceback.format_exception(
            type(error),
            error,
            error.__traceback__,
        )
    )

    source_file, source_function, source_line = (
        extract_traceback_location(traceback_text)
    )

    (
        error_type,
        likely_cause,
        suggested_fix,
        retryable,
    ) = classify_issue(traceback_text)

    report = DiagnosticReport(
        run_id=make_run_id(),
        stage=stage,
        script_name=script_name,
        status="failed",
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        duration_seconds=round(
            (finished_at - started_at).total_seconds(),
            3,
        ),
        attempt=attempt,
        error_type=error_type or type(error).__name__,
        error_message=str(error),
        source_file=source_file,
        source_function=source_function,
        source_line=source_line,
        likely_cause=likely_cause,
        suggested_fix=suggested_fix,
        retryable=retryable,
        traceback_text=traceback_text,
        metadata=metadata or {},
    )

    save_report(report)

    return report


def load_latest_report() -> dict[str, Any] | None:
    if not LATEST_REPORT_PATH.exists():
        return None

    try:
        return json.loads(
            LATEST_REPORT_PATH.read_text(
                encoding="utf-8",
                errors="replace",
            )
        )
    except (json.JSONDecodeError, OSError):
        return None


def wait_before_retry(attempt: int) -> None:
    """
    Small exponential delay: 2 seconds, then 4 seconds.
    """

    time.sleep(min(2 ** attempt, 4))