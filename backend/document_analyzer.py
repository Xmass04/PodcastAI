import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_PATH = PROJECT_ROOT / ".env"

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "extracted_text"
    / "output.txt"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "analysis.json"
)


class DocumentAnalysis(BaseModel):

    document_title: str

    document_type: Literal[
        "Revision Notes",
        "Textbook",
        "Research Paper",
        "Course Handbook",
        "Assignment",
        "Lecture Notes",
        "Meeting Notes",
        "Report",
        "General Document",
    ]

    confidence: int = Field(ge=0, le=100)

    topics_found: int

    difficulty: Literal[
        "Easy",
        "Medium",
        "Hard",
    ]

    estimated_reading_minutes: int

    estimated_revision_minutes: int

    recommended_output: Literal[
        "Study Pack",
        "Summary",
        "Research Pack",
        "Work Pack",
    ]

    reason: str


def analyse_document(text: str):

    load_dotenv(dotenv_path=ENV_PATH)

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found.")

    client = OpenAI(api_key=api_key)

    response = client.responses.parse(

        model="gpt-5.5",

        input=[
            {
                "role": "system",
                "content":
                """
You are an expert document analyst.

Analyse the document and return structured information.

Estimate:

• document type

• confidence

• number of major topics

• reading time

• revision time

• difficulty

Recommend ONE output:

Study Pack

Summary

Research Pack

Work Pack

Base your answer ONLY on the supplied document.
"""
            },
            {
                "role":"user",
                "content":text,
            }
        ],

        text_format=DocumentAnalysis,
    )

    return response.output_parsed


def save_analysis(analysis):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(

        json.dumps(
            analysis.model_dump(),
            indent=4,
            ensure_ascii=False,
        ),

        encoding="utf-8",
    )


def main():

    if not INPUT_PATH.exists():

        print("No extracted text found.")

        return

    text = INPUT_PATH.read_text(
        encoding="utf-8"
    )

    print("Analysing document...")

    analysis = analyse_document(text)

    save_analysis(analysis)

    print("\nAnalysis complete!")

    print(f"\nSaved to:\n{OUTPUT_PATH}")

    print("\nSummary")

    print("----------------")

    print(f"Title: {analysis.document_title}")

    print(f"Type: {analysis.document_type}")

    print(f"Confidence: {analysis.confidence}%")

    print(f"Topics: {analysis.topics_found}")

    print(f"Difficulty: {analysis.difficulty}")

    print(
        f"Reading Time: "
        f"{analysis.estimated_reading_minutes} min"
    )

    print(
        f"Revision Time: "
        f"{analysis.estimated_revision_minutes} min"
    )

    print(
        f"Recommended: "
        f"{analysis.recommended_output}"
    )


if __name__ == "__main__":
    main()
 