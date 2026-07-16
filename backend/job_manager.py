from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_FOLDER = PROJECT_ROOT / "backend"
JOBS_FOLDER = PROJECT_ROOT / "data" / "jobs"
WORKER_PATH = BACKEND_FOLDER / "worker.py"

FINAL_STATES = {
    "completed",
    "completed_with_errors",
    "failed",
    "cancelled",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_jobs_folder() -> None:
    JOBS_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )


def generate_job_id() -> str:
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    return (
        f"{timestamp}_"
        f"{uuid.uuid4().hex[:10]}"
    )


def job_folder(job_id: str) -> Path:
    return JOBS_FOLDER / job_id


def job_file(job_id: str) -> Path:
    return job_folder(job_id) / "job.json"


def log_file(job_id: str) -> Path:
    return job_folder(job_id) / "worker.log"


def cancel_file(job_id: str) -> Path:
    return job_folder(job_id) / "cancel.request"


def write_json_atomic(
    path: Path,
    data: dict[str, Any],
) -> None:
    """
    Write JSON through a temporary file so the frontend never reads a
    half-written progress report.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{path.stem}_",
        suffix=".tmp",
        dir=path.parent,
    )

    temporary_path = Path(
        temporary_name
    )

    try:
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
        ) as temporary_file:
            json.dump(
                data,
                temporary_file,
                indent=2,
                ensure_ascii=False,
            )

        os.replace(
            temporary_path,
            path,
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink(
                missing_ok=True
            )


def read_job(job_id: str) -> dict[str, Any] | None:
    path = job_file(job_id)

    if not path.exists():
        return None

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        )

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return None


def normalise_task(
    task: dict[str, Any],
    task_number: int,
) -> dict[str, Any]:
    script_name = str(
        task.get("script_name", "")
    ).strip()

    if not script_name:
        raise ValueError(
            f"Task {task_number} has no script_name."
        )

    return {
        "task_id": (
            str(
                task.get(
                    "task_id",
                    f"task_{task_number}",
                )
            )
        ),
        "name": str(
            task.get(
                "name",
                Path(script_name).stem.replace(
                    "_",
                    " ",
                ).title(),
            )
        ),
        "script_name": script_name,
        "status": "queued",
        "progress": 0,
        "message": "Waiting",
        "attempt": 0,
        "max_attempts": max(
            1,
            int(
                task.get(
                    "max_attempts",
                    2,
                )
            ),
        ),
        "environment": {
            str(key): str(value)
            for key, value in (
                task.get(
                    "environment",
                    {},
                )
                or {}
            ).items()
        },
        "input_text": task.get(
            "input_text"
        ),
        "started_at": None,
        "finished_at": None,
        "duration_seconds": None,
        "return_code": None,
        "stdout": "",
        "stderr": "",
        "error": None,
        "diagnostic_report": None,
    }


def create_job(
    *,
    tasks: list[dict[str, Any]],
    title: str = "PodcastAI generation",
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Create a durable job description.

    Each task must include script_name, for example:
    {"name": "Summary", "script_name": "ai_summarizer.py"}
    """

    if not tasks:
        raise ValueError(
            "A background job needs at least one task."
        )

    ensure_jobs_folder()

    job_id = generate_job_id()
    folder = job_folder(job_id)

    folder.mkdir(
        parents=True,
        exist_ok=False,
    )

    normalised_tasks = [
        normalise_task(
            task,
            task_number,
        )
        for task_number, task in enumerate(
            tasks,
            start=1,
        )
    ]

    job = {
        "job_id": job_id,
        "title": title,
        "status": "queued",
        "progress": 0,
        "message": "Job created",
        "created_at": utc_now_iso(),
        "started_at": None,
        "finished_at": None,
        "current_task_index": None,
        "current_task_name": None,
        "completed_tasks": 0,
        "failed_tasks": 0,
        "total_tasks": len(
            normalised_tasks
        ),
        "cancel_requested": False,
        "worker_pid": None,
        "metadata": metadata or {},
        "tasks": normalised_tasks,
    }

    write_json_atomic(
        job_file(job_id),
        job,
    )

    return job_id


def start_job(job_id: str) -> int:
    """
    Start worker.py in a separate process and return its PID.

    On Windows, creation flags detach the worker from Streamlit's console
    without opening another terminal window.
    """

    if not WORKER_PATH.exists():
        raise FileNotFoundError(
            f"Worker file was not found: {WORKER_PATH}"
        )

    job = read_job(job_id)

    if not job:
        raise FileNotFoundError(
            f"Job was not found: {job_id}"
        )

    if job.get("status") not in {
        "queued",
        "failed",
    }:
        existing_pid = job.get(
            "worker_pid"
        )

        if existing_pid:
            return int(existing_pid)

        raise RuntimeError(
            "Only queued or failed jobs can be started."
        )

    worker_executable = Path(sys.executable)

    command = [
        str(worker_executable),
        "-m",
        "backend.worker",
        job_id,
    ]

    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"

    bootstrap_log_path = job_folder(job_id) / "worker_bootstrap.log"

    bootstrap_log = bootstrap_log_path.open(
        "a",
        encoding="utf-8",
    )

    popen_arguments: dict[str, Any] = {
        "cwd": PROJECT_ROOT,
        "env": environment,
        "stdin": subprocess.DEVNULL,
        "stdout": bootstrap_log,
        "stderr": subprocess.STDOUT,
        "close_fds": True,
    }

    if os.name == "nt":
        startup_info = subprocess.STARTUPINFO()
        startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup_info.wShowWindow = subprocess.SW_HIDE

        popen_arguments["startupinfo"] = startup_info
        popen_arguments["creationflags"] = subprocess.CREATE_NO_WINDOW
    else:
        popen_arguments["start_new_session"] = True

    try:
        process = subprocess.Popen(
            command,
            **popen_arguments,
        )
    finally:
        bootstrap_log.close()

    job["worker_pid"] = process.pid
    job["message"] = "Worker starting"

    write_json_atomic(
        job_file(job_id),
        job,
    )

    return process.pid


def create_and_start_job(
    *,
    tasks: list[dict[str, Any]],
    title: str = "PodcastAI generation",
    metadata: dict[str, Any] | None = None,
) -> str:
    job_id = create_job(
        tasks=tasks,
        title=title,
        metadata=metadata,
    )

    start_job(job_id)

    return job_id


def request_cancel(job_id: str) -> bool:
    job = read_job(job_id)

    if not job:
        return False

    if job.get("status") in FINAL_STATES:
        return False

    cancel_file(job_id).write_text(
        utc_now_iso(),
        encoding="utf-8",
    )

    job["cancel_requested"] = True
    job["message"] = (
        "Cancellation requested"
    )

    write_json_atomic(
        job_file(job_id),
        job,
    )

    return True


def is_finished(job: dict[str, Any] | None) -> bool:
    return bool(
        job
        and job.get("status") in FINAL_STATES
    )


def list_jobs(
    limit: int = 20,
) -> list[dict[str, Any]]:
    ensure_jobs_folder()

    jobs: list[dict[str, Any]] = []

    folders = sorted(
        (
            folder
            for folder in JOBS_FOLDER.iterdir()
            if folder.is_dir()
        ),
        key=lambda folder: (
            folder.stat().st_mtime
        ),
        reverse=True,
    )

    for folder in folders:
        job = read_job(
            folder.name
        )

        if job:
            jobs.append(job)

        if len(jobs) >= limit:
            break

    return jobs