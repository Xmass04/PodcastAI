from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_FOLDER = PROJECT_ROOT / "backend"


@dataclass(frozen=True)
class ParallelTask:
    name: str
    script_name: str


TASKS = [
    ParallelTask("Summary", "ai_summarizer.py"),
    ParallelTask("Flashcards", "flashcard_generator.py"),
    ParallelTask("Quiz", "quiz_generator.py"),
]


def hidden_process_options() -> dict:
    options: dict = {}

    if os.name == "nt":
        startup_info = subprocess.STARTUPINFO()
        startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup_info.wShowWindow = subprocess.SW_HIDE

        options["startupinfo"] = startup_info
        options["creationflags"] = subprocess.CREATE_NO_WINDOW

    return options


def run_task(task: ParallelTask) -> tuple[str, bool, str]:
    script_path = BACKEND_FOLDER / task.script_name

    if not script_path.exists():
        return (
            task.name,
            False,
            f"Missing backend file: {script_path}",
        )

    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        check=False,
        **hidden_process_options(),
    )

    output = "\n".join(
        part
        for part in (result.stdout, result.stderr)
        if part
    ).strip()

    lowered = output.lower()
    printed_failure = (
        "generation failed" in lowered
        or "analysis failed" in lowered
        or "\nerror:" in lowered
        or lowered.startswith("error:")
    )

    success = result.returncode == 0 and not printed_failure

    return task.name, success, output


def main() -> None:
    print(
        "PROGRESS|2|Starting Summary, Flashcards and Quiz together...",
        flush=True,
    )

    completed = 0
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {
            executor.submit(run_task, task): task
            for task in TASKS
        }

        for future in as_completed(future_map):
            task = future_map[future]

            try:
                name, success, output = future.result()
            except Exception as error:
                name = task.name
                success = False
                output = str(error)

            completed += 1
            percent = min(95, int(completed / len(TASKS) * 95))

            if success:
                print(
                    f"RESOURCE_READY|{name}",
                    flush=True,
                )
                print(
                    f"PROGRESS|{percent}|{name} ready",
                    flush=True,
                )
            else:
                failures.append(
                    f"{name} failed:\n{output}"
                )
                print(
                    f"PROGRESS|{percent}|{name} failed",
                    flush=True,
                )

    if failures:
        print("\nERROR: Quick Pack generation failed:")
        print("\n\n".join(failures))
        raise SystemExit(1)

    print(
        "PROGRESS|100|Quick Pack ready",
        flush=True,
    )
    print("SUCCESS: Summary, Flashcards and Quiz created in parallel.")


if __name__ == "__main__":
    main()