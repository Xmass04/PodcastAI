from __future__ import annotations

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


DocumentMode = Literal[
    "study",
    "story",
    "work",
    "research",
    "general",
]


class DocumentAnalysis(BaseModel):
    document_title: str = Field(
        description=(
            "A short title based only on the supplied document."
        )
    )

    document_mode: DocumentMode = Field(
        description=(
            "The best processing mode for this document."
        )
    )

    document_type: str = Field(
        description=(
            "A clear human-readable type such as Revision Notes, "
            "Novel, Course Handbook, Research Paper or Report."
        )
    )

    confidence: int = Field(
        ge=0,
        le=100,
        description=(
            "Confidence that the selected document mode is correct."
        ),
    )

    topics_or_chapters_found: int = Field(
        ge=1,
        description=(
            "Estimated number of major topics, sections or chapters."
        ),
    )

    difficulty: Literal[
        "Easy",
        "Medium",
        "Hard",
    ]

    estimated_reading_minutes: int = Field(
        ge=1,
    )

    estimated_learning_minutes: int = Field(
        ge=1,
    )

    recommended_pack: Literal[
        "Study Pack",
        "Story Pack",
        "Work Brief",
        "Research Pack",
        "General Summary",
    ]

    recommended_outputs: list[str] = Field(
        description=(
            "The most useful outputs for this particular document."
        )
    )

    audio_style: Literal[
        "Tutor Conversation",
        "Story Narration",
        "Professional Briefing",
        "Research Discussion",
        "General Explanation",
    ]

    reason: str = Field(
        description=(
            "A brief explanation of why this mode was selected."
        )
    )

    warnings: list[str] = Field(
        description=(
            "Potential concerns such as poor OCR, incomplete text, "
            "mixed document types or unsuitable content."
        )
    )


def load_document_text() -> str:
    """Load the text extracted from the current PDF or image."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "No extracted text was found. "
            "Upload and analyse a PDF or image first."
        )

    text = INPUT_PATH.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()

    if not text:
        raise ValueError(
            "The extracted-text file is empty."
        )

    return text


def analyse_document(text: str) -> DocumentAnalysis:
    """Detect the best PodcastAI mode for the document."""

    load_dotenv(dotenv_path=ENV_PATH)

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY was not found."
        )

    client = OpenAI(api_key=api_key)

    response = client.responses.parse(
        model="gpt-5.5",
        input=[
            {
                "role": "system",
                "content": (
                    "You are PodcastAI's document-routing system. "
                    "Analyse the supplied document and choose the most "
                    "appropriate experience.\n\n"
                    "Use these modes:\n"
                    "- study: revision notes, textbooks, lessons, "
                    "training content or educational material\n"
                    "- story: novels, fiction, short stories, plays "
                    "or narrative writing\n"
                    "- work: reports, policies, meeting notes, manuals, "
                    "handbooks or professional documents\n"
                    "- research: academic papers, scientific reports "
                    "or formal research\n"
                    "- general: material that does not clearly fit "
                    "the other categories\n\n"
                    "Audio styles must match the content:\n"
                    "- study -> Tutor Conversation\n"
                    "- story -> Story Narration\n"
                    "- work -> Professional Briefing\n"
                    "- research -> Research Discussion\n"
                    "- general -> General Explanation\n\n"
                    "Use only information supported by the document. "
                    "Do not invent titles, chapters or subject matter."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Analyse this document and recommend the best "
                    "PodcastAI mode and outputs.\n\n"
                    f"DOCUMENT:\n{text}"
                ),
            },
        ],
        text_format=DocumentAnalysis,
    )

    analysis = response.output_parsed

    if analysis is None:
        raise RuntimeError(
            "The AI did not return a valid document analysis."
        )

    return analysis


def save_analysis(
    analysis: DocumentAnalysis,
) -> None:
    """Save the structured analysis for the web interface."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            analysis.model_dump(),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def print_analysis(
    analysis: DocumentAnalysis,
) -> None:
    """Display a readable terminal summary."""

    print("\nAnalysis complete!")
    print("-" * 50)
    print(f"Title: {analysis.document_title}")
    print(f"Mode: {analysis.document_mode}")
    print(f"Type: {analysis.document_type}")
    print(f"Confidence: {analysis.confidence}%")
    print(
        "Topics/chapters: "
        f"{analysis.topics_or_chapters_found}"
    )
    print(f"Difficulty: {analysis.difficulty}")
    print(
        "Reading time: "
        f"{analysis.estimated_reading_minutes} min"
    )
    print(
        "Learning time: "
        f"{analysis.estimated_learning_minutes} min"
    )
    print(
        "Recommended pack: "
        f"{analysis.recommended_pack}"
    )
    print(f"Audio style: {analysis.audio_style}")

    print("\nRecommended outputs:")

    for output in analysis.recommended_outputs:
        print(f"- {output}")

    if analysis.warnings:
        print("\nWarnings:")

        for warning in analysis.warnings:
            print(f"- {warning}")

    print(f"\nSaved to: {OUTPUT_PATH}")


def main() -> None:
    try:
        print("Analysing document type...")

        text = load_document_text()
        analysis = analyse_document(text)

        save_analysis(analysis)
        print_analysis(analysis)

    except Exception as error:
        print(
            "\nERROR: Document analysis failed: "
            f"{error}"
        )


if __name__ == "__main__":
    main()