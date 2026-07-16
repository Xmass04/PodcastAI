from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_FOLDER = PROJECT_ROOT / "backend"

if str(BACKEND_FOLDER) not in sys.path:
    sys.path.insert(
        0,
        str(BACKEND_FOLDER),
    )

from diagnostics import (  # noqa: E402
    record_subprocess_result,
    utc_now,
    wait_before_retry,
)

from job_manager import (  # noqa: E402
    cancel_file,
    job_file,
    log_file,
    read_job,
    write_json_atomic,
)


PROGRESS_PATTERN = re.compile(
    r"^PROGRESS\|"
    r"(?P<percent>\d{1,3})\|"
    r"(?P<message>.*)$"
)

RECORDING_PATTERN = re.compile(
    r"Recording\s+(?P<current>\d+)\s*/\s*(?P<total>\d+)",
    flags=re.IGNORECASE,
)

RESOURCE_READY_PATTERN = re.compile(
    r"^RESOURCE_READY\|(?P<name>.+)$",
    flags=re.IGNORECASE,
)


def utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def append_worker_log(
    job_id: str,
    message: str,
) -> None:
    path = log_file(job_id)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as output:
        output.write(
            f"[{utc_now_iso()}] "
            f"{message.rstrip()}\n"
        )


def save_job(
    job_id: str,
    job: dict[str, Any],
) -> None:
    write_json_atomic(
        job_file(job_id),
        job,
    )


def calculate_overall_progress(
    job: dict[str, Any],
    task_index: int,
    task_progress: int,
) -> int:
    total_tasks = max(
        1,
        int(
            job.get(
                "total_tasks",
                len(job.get("tasks", [])),
            )
        ),
    )

    completed_fraction = (
        task_index
        + max(
            0,
            min(
                task_progress,
                100,
            ),
        )
        / 100
    )

    return min(
        99,
        int(
            completed_fraction
            / total_tasks
            * 100
        ),
    )


def cancellation_requested(
    job_id: str,
) -> bool:
    return cancel_file(
        job_id
    ).exists()


def classify_terminal_success(
    return_code: int,
    stdout: str,
    stderr: str,
) -> bool:
    combined = "\n".join(
        part
        for part in (
            stdout,
            stderr,
        )
        if part
    ).lower()

    printed_failure = (
        "generation failed" in combined
        or "analysis failed" in combined
        or "\nerror:" in combined
        or combined.startswith("error:")
    )

    return (
        return_code == 0
        and not printed_failure
    )


def update_from_output_line(
    *,
    job_id: str,
    job: dict[str, Any],
    task_index: int,
    line: str,
) -> None:
    task = job["tasks"][
        task_index
    ]

    resource_match = RESOURCE_READY_PATTERN.match(
        line.strip()
    )

    if resource_match:
        resource_name = resource_match.group("name").strip()
        ready_resources = job.setdefault("ready_resources", [])

        if resource_name and resource_name not in ready_resources:
            ready_resources.append(resource_name)

        job["message"] = f"{resource_name} ready"
        save_job(job_id, job)
        return

    progress_match = (
        PROGRESS_PATTERN.match(
            line.strip()
        )
    )

    if progress_match:
        task_progress = max(
            0,
            min(
                int(
                    progress_match.group(
                        "percent"
                    )
                ),
                100,
            ),
        )

        message = (
            progress_match.group(
                "message"
            ).strip()
            or task["name"]
        )

        task["progress"] = task_progress
        task["message"] = message
        job["message"] = message
        job["progress"] = calculate_overall_progress(
            job,
            task_index,
            task_progress,
        )

    else:
        recording_match = RECORDING_PATTERN.search(
            line.strip()
        )

        if recording_match:
            current = int(
                recording_match.group("current")
            )
            total = max(
                1,
                int(
                    recording_match.group("total")
                ),
            )

            # Reserve the final 10% of this task for combining/saving.
            task_progress = min(
                90,
                max(
                    2,
                    int(current / total * 90),
                ),
            )

            message = (
                f"Recording audio clip "
                f"{current}/{total}"
            )

            task["progress"] = task_progress
            task["message"] = message
            job["message"] = message
            job["progress"] = calculate_overall_progress(
                job,
                task_index,
                task_progress,
            )

        elif "combining" in line.lower():
            task["progress"] = max(
                int(task.get("progress", 0)),
                95,
            )
            task["message"] = "Combining audio..."
            job["message"] = "Combining audio..."
            job["progress"] = calculate_overall_progress(
                job,
                task_index,
                95,
            )

        elif line.strip():
            task["message"] = line.strip()[-300:]
            job["message"] = (
                f"{task['name']}: "
                f"{line.strip()[-200:]}"
            )

    save_job(
        job_id,
        job,
    )


def run_task_attempt(
    *,
    job_id: str,
    job: dict[str, Any],
    task_index: int,
    attempt: int,
) -> tuple[
    bool,
    str,
    str,
    int,
    dict[str, Any],
]:
    task = job["tasks"][
        task_index
    ]

    script_path = (
        BACKEND_FOLDER
        / task["script_name"]
    )

    if not script_path.exists():
        return (
            False,
            "",
            (
                "ERROR: Missing backend file: "
                f"{script_path}"
            ),
            1,
            {
                "retryable": False,
                "suggested_fix": (
                    "Restore the missing backend "
                    "file and retry."
                ),
            },
        )

    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"

    environment.update(
        task.get(
            "environment",
            {},
        )
        or {}
    )

    command = [
        sys.executable,
        "-u",
        str(script_path),
    ]

    started_at = utc_now()

    popen_arguments: dict[str, Any] = {
        "cwd": PROJECT_ROOT,
        "env": environment,
        "stdin": (
            subprocess.PIPE
            if task.get(
                "input_text"
            ) is not None
            else subprocess.DEVNULL
        ),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
    }

    if os.name == "nt":
        startup_info = subprocess.STARTUPINFO()
        startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup_info.wShowWindow = subprocess.SW_HIDE

        popen_arguments["startupinfo"] = startup_info
        popen_arguments["creationflags"] = (
            subprocess.CREATE_NO_WINDOW
        )

    process = subprocess.Popen(
        command,
        **popen_arguments,
    )

    if (
        task.get("input_text")
        is not None
        and process.stdin
    ):
        process.stdin.write(
            str(
                task.get(
                    "input_text"
                )
            )
        )

        if not str(
            task.get(
                "input_text"
            )
        ).endswith("\n"):
            process.stdin.write("\n")

        process.stdin.close()

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    while True:
        if cancellation_requested(
            job_id
        ):
            process.terminate()

            try:
                process.wait(
                    timeout=5
                )
            except subprocess.TimeoutExpired:
                process.kill()

            return (
                False,
                "\n".join(
                    stdout_lines
                ),
                "Job cancelled.",
                130,
                {
                    "cancelled": True,
                    "retryable": False,
                },
            )

        output_line = (
            process.stdout.readline()
            if process.stdout
            else ""
        )

        if output_line:
            clean_line = output_line.rstrip(
                "\n"
            )

            stdout_lines.append(
                clean_line
            )

            append_worker_log(
                job_id,
                output_line,
            )

            update_from_output_line(
                job_id=job_id,
                job=job,
                task_index=task_index,
                line=output_line,
            )

        if process.poll() is not None:
            if process.stdout:
                remaining_output = process.stdout.read()

                if remaining_output:
                    for remaining_line in remaining_output.splitlines():
                        stdout_lines.append(
                            remaining_line
                        )

                        append_worker_log(
                            job_id,
                            remaining_line,
                        )

                        update_from_output_line(
                            job_id=job_id,
                            job=job,
                            task_index=task_index,
                            line=remaining_line,
                        )

            break

        if not output_line:
            time.sleep(0.1)

    return_code = (
        process.returncode
        if process.returncode
        is not None
        else 1
    )

    stdout = "\n".join(
        stdout_lines
    )

    # stderr is merged into stdout to prevent Windows pipe deadlocks.
    stderr = ""

    finished_at = utc_now()

    report = record_subprocess_result(
        stage=task["name"],
        script_name=task[
            "script_name"
        ],
        started_at=started_at,
        finished_at=finished_at,
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
        attempt=attempt,
        metadata={
            "job_id": job_id,
            "task_id": task[
                "task_id"
            ],
            "background_worker": True,
        },
    )

    report_dictionary = {
        "status": report.status,
        "retryable": report.retryable,
        "error_type": report.error_type,
        "likely_cause": report.likely_cause,
        "suggested_fix": report.suggested_fix,
        "source_file": report.source_file,
        "source_function": report.source_function,
        "source_line": report.source_line,
    }

    success = (
        report.status
        == "success"
        and classify_terminal_success(
            return_code,
            stdout,
            stderr,
        )
    )

    return (
        success,
        stdout,
        stderr,
        return_code,
        report_dictionary,
    )


def run_job(job_id: str) -> None:
    job = read_job(job_id)

    if not job:
        raise FileNotFoundError(
            f"Job was not found: {job_id}"
        )

    job["status"] = "running"
    job["started_at"] = (
        utc_now_iso()
    )
    job["message"] = (
        "Background worker started"
    )
    job["progress"] = 0

    save_job(
        job_id,
        job,
    )

    append_worker_log(
        job_id,
        "Worker started.",
    )

    for task_index, task in enumerate(
        job["tasks"]
    ):
        if cancellation_requested(
            job_id
        ):
            job["status"] = (
                "cancelled"
            )
            job["cancel_requested"] = (
                True
            )
            job["message"] = (
                "Job cancelled"
            )
            break

        job["current_task_index"] = (
            task_index
        )
        job["current_task_name"] = (
            task["name"]
        )

        task["status"] = "running"
        task["progress"] = 1
        task["message"] = (
            f"Starting {task['name']}"
        )
        task["started_at"] = (
            utc_now_iso()
        )

        job["message"] = (
            task["message"]
        )
        job["progress"] = (
            calculate_overall_progress(
                job,
                task_index,
                1,
            )
        )

        save_job(
            job_id,
            job,
        )

        task_started_monotonic = (
            time.monotonic()
        )

        task_success = False

        for attempt in range(
            1,
            task["max_attempts"]
            + 1,
        ):
            task["attempt"] = attempt
            task["message"] = (
                f"{task['name']} "
                f"(attempt {attempt})"
            )

            save_job(
                job_id,
                job,
            )

            (
                success,
                stdout,
                stderr,
                return_code,
                diagnostic,
            ) = run_task_attempt(
                job_id=job_id,
                job=job,
                task_index=task_index,
                attempt=attempt,
            )

            task["stdout"] = stdout
            task["stderr"] = stderr
            task["return_code"] = (
                return_code
            )
            task[
                "diagnostic_report"
            ] = diagnostic

            if diagnostic.get(
                "cancelled"
            ):
                job["status"] = (
                    "cancelled"
                )
                job[
                    "cancel_requested"
                ] = True
                job["message"] = (
                    "Job cancelled"
                )
                break

            if success:
                task_success = True
                break

            if (
                diagnostic.get(
                    "retryable"
                )
                and attempt
                < task[
                    "max_attempts"
                ]
            ):
                task["status"] = (
                    "retrying"
                )
                task["message"] = (
                    "Temporary issue detected. "
                    "Retrying..."
                )

                save_job(
                    job_id,
                    job,
                )

                wait_before_retry(
                    attempt
                )

                continue

            break

        task["finished_at"] = (
            utc_now_iso()
        )

        task["duration_seconds"] = (
            round(
                time.monotonic()
                - task_started_monotonic,
                3,
            )
        )

        if job["status"] == (
            "cancelled"
        ):
            task["status"] = (
                "cancelled"
            )
            task["message"] = (
                "Cancelled"
            )
            break

        if task_success:
            task["status"] = (
                "completed"
            )
            task["progress"] = 100
            task["message"] = (
                f"{task['name']} ready"
            )

            job["completed_tasks"] += 1

        else:
            task["status"] = "failed"
            task["progress"] = 100
            task["message"] = (
                f"{task['name']} failed"
            )

            task["error"] = (
                task.get(
                    "diagnostic_report",
                    {},
                ).get(
                    "likely_cause"
                )
                or stderr
                or stdout
                or "Unknown error"
            )

            job["failed_tasks"] += 1

        job["progress"] = int(
            (
                task_index + 1
            )
            / max(
                1,
                job[
                    "total_tasks"
                ],
            )
            * 100
        )

        job["message"] = (
            task["message"]
        )

        save_job(
            job_id,
            job,
        )

    job["current_task_index"] = None
    job["current_task_name"] = None
    job["finished_at"] = (
        utc_now_iso()
    )

    if job["status"] == "cancelled":
        job["progress"] = min(
            job.get(
                "progress",
                0,
            ),
            99,
        )

    elif job["failed_tasks"] == 0:
        job["status"] = "completed"
        job["progress"] = 100
        job["message"] = (
            "All requested content is ready"
        )

    elif job["completed_tasks"] > 0:
        job["status"] = (
            "completed_with_errors"
        )
        job["progress"] = 100
        job["message"] = (
            "Generation finished with "
            "some failed tasks"
        )

    else:
        job["status"] = "failed"
        job["progress"] = 100
        job["message"] = (
            "Generation failed"
        )

    save_job(
        job_id,
        job,
    )

    append_worker_log(
        job_id,
        (
            "Worker finished with status: "
            f"{job['status']}"
        ),
    )


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: python backend/worker.py "
            "<job_id>"
        )
        raise SystemExit(2)

    run_job(
        sys.argv[1]
    )


if __name__ == "__main__":
    main()