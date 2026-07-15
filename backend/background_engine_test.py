from __future__ import annotations

import json
import time
from pathlib import Path

from job_manager import (
    create_and_start_job,
    is_finished,
    read_job,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_TEXT_PATH = (
    PROJECT_ROOT
    / "data"
    / "extracted_text"
    / "output.txt"
)


def main() -> None:
    if not EXTRACTED_TEXT_PATH.exists():
        print(
            "Upload and analyse a document first so "
            "data/extracted_text/output.txt exists."
        )
        return

    job_id = create_and_start_job(
        title="PodcastAI background-engine test",
        tasks=[
            {
                "name": "Document analysis",
                "script_name": "document_analyzer.py",
            },
            {
                "name": "Summary",
                "script_name": "ai_summarizer.py",
            },
        ],
        metadata={
            "test_job": True,
        },
    )

    print(f"Created job: {job_id}")

    while True:
        job = read_job(job_id)

        if not job:
            print(
                "The job progress file could not be read."
            )
            break

        print(
            json.dumps(
                {
                    "status": job.get(
                        "status"
                    ),
                    "progress": job.get(
                        "progress"
                    ),
                    "message": job.get(
                        "message"
                    ),
                    "completed_tasks": job.get(
                        "completed_tasks"
                    ),
                    "failed_tasks": job.get(
                        "failed_tasks"
                    ),
                },
                indent=2,
            )
        )

        if is_finished(job):
            break

        time.sleep(1)


if __name__ == "__main__":
    main()