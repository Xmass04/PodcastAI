from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

ANALYSIS_PATH = PROJECT_ROOT / "data" / "analysis" / "analysis.json"
EXTRACTED_TEXT_PATH = PROJECT_ROOT / "data" / "extracted_text" / "output.txt"
STUDY_NOTES_PATH = PROJECT_ROOT / "data" / "summaries" / "study_notes.txt"
SUMMARY_PATH = PROJECT_ROOT / "data" / "summaries" / "summary.txt"
OUTPUT_PATH = PROJECT_ROOT / "data" / "podcasts" / "podcast_script.txt"

MODEL_NAME = "gpt-5.5"


def read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return text or None


def load_analysis() -> dict[str, Any]:
    fallback = {
        "document_mode": "study",
        "audio_style": "Tutor Conversation",
        "document_title": "PodcastAI Episode",
    }

    if not ANALYSIS_PATH.exists():
        return fallback

    try:
        return json.loads(
            ANALYSIS_PATH.read_text(encoding="utf-8", errors="replace")
        )
    except (json.JSONDecodeError, OSError):
        return fallback


def choose_source_text(document_mode: str) -> tuple[str, str]:
    if document_mode == "study":
        notes = read_text(STUDY_NOTES_PATH)
        if notes:
            return notes, "structured study notes"

    if document_mode in {"work", "research", "general"}:
        summary = read_text(SUMMARY_PATH)
        if summary:
            return summary, "summary"

    extracted = read_text(EXTRACTED_TEXT_PATH)
    if extracted:
        return extracted, "extracted document text"

    raise FileNotFoundError(
        "No usable source text was found. Upload and analyse a PDF or image first."
    )


def build_instructions(document_mode: str) -> str:
    if document_mode == "story":
        return (
            "You are a professional story-audio script writer.\n\n"
            "Transform the supplied story or novel material into a polished, "
            "engaging narrated adaptation.\n\n"
            "Requirements:\n"
            "- Use the speaker label 'Narrator:' for every spoken section.\n"
            "- Do not use Alex or Jamie.\n"
            "- Preserve the story's tone, setting, main events and characters.\n"
            "- Do not invent major plot events.\n"
            "- Do not reproduce a long copyrighted work word-for-word.\n"
            "- Retell and adapt the supplied material naturally.\n"
            "- Include a clear title and optional section headings.\n"
            "- Make the narration vivid, smooth and suitable for listening.\n"
            "- End at a natural stopping point with a complete final sentence.\n"
            "- Never finish mid-sentence."
        )

    if document_mode == "work":
        return (
            "You are a professional workplace briefing writer.\n\n"
            "Create a concise, useful audio briefing between two presenters.\n\n"
            "Alex explains responsibilities, risks, deadlines and key decisions.\n"
            "Jamie asks clarifying questions and highlights practical actions.\n\n"
            "Requirements:\n"
            "- Every spoken turn must begin with 'Alex:' or 'Jamie:'.\n"
            "- Focus on key actions, responsibilities, deadlines, policies and risks.\n"
            "- Avoid exam language.\n"
            "- Use only information supported by the document.\n"
            "- End with a clear action recap and complete goodbye.\n"
            "- Never finish mid-sentence."
        )

    if document_mode == "research":
        return (
            "You are a professional research-podcast writer.\n\n"
            "Create an accurate discussion between Alex and Jamie.\n\n"
            "Alex explains the research question, methods and findings.\n"
            "Jamie asks critical questions about evidence, limitations and meaning.\n\n"
            "Requirements:\n"
            "- Every spoken turn must begin with 'Alex:' or 'Jamie:'.\n"
            "- Explain technical ideas in accessible language.\n"
            "- Separate findings from interpretation.\n"
            "- Mention limitations only when supported by the source.\n"
            "- Do not invent statistics, claims or conclusions.\n"
            "- Finish with a concise recap and complete closing."
        )

    if document_mode == "general":
        return (
            "You are a professional explanatory-podcast writer.\n\n"
            "Create a clear conversation between Alex and Jamie.\n\n"
            "Requirements:\n"
            "- Every spoken turn must begin with 'Alex:' or 'Jamie:'.\n"
            "- Alex explains the main information.\n"
            "- Jamie asks natural questions and simplifies difficult points.\n"
            "- Use only information supported by the document.\n"
            "- Organise the episode into a beginning, explanation and recap.\n"
            "- End naturally with a complete final sentence."
        )

    return (
        "You are a professional educational podcast writer.\n\n"
        "Create an engaging tutor conversation between Alex and Jamie.\n\n"
        "Alex is a calm teacher who explains ideas and gives examples.\n"
        "Jamie is a curious learner who asks questions and checks understanding.\n\n"
        "Requirements:\n"
        "- Every spoken turn must begin with 'Alex:' or 'Jamie:'.\n"
        "- Teach rather than simply reading the notes.\n"
        "- Include an introduction, organised teaching sections, active-recall "
        "questions, a final recap and a friendly outro.\n"
        "- Use only information supported by the supplied material.\n"
        "- The ending must be complete and natural.\n"
        "- Never finish mid-sentence."
    )


def generate_podcast_script(source_text: str, analysis: dict[str, Any]) -> str:
    load_dotenv(dotenv_path=ENV_PATH)
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY was not found.")

    document_mode = str(analysis.get("document_mode", "study")).lower()
    audio_style = str(analysis.get("audio_style", "Tutor Conversation"))
    document_title = str(analysis.get("document_title", "PodcastAI Episode"))

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=MODEL_NAME,
        instructions=build_instructions(document_mode),
        input=(
            f"Document title: {document_title}\n"
            f"Document mode: {document_mode}\n"
            f"Requested audio style: {audio_style}\n\n"
            "Create the complete audio script from the material below.\n\n"
            f"MATERIAL:\n{source_text}"
        ),
        max_output_tokens=3500,
    )

    script = response.output_text.strip()
    if not script:
        raise RuntimeError("The AI returned an empty podcast script.")

    return script


def save_script(script: str) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(script, encoding="utf-8")


def main() -> None:
    try:
        analysis = load_analysis()
        document_mode = str(analysis.get("document_mode", "study")).lower()
        source_text, source_name = choose_source_text(document_mode)

        print(f"Generating {document_mode} audio script from {source_name}...")

        script = generate_podcast_script(source_text, analysis)
        save_script(script)

        print("\nSUCCESS: Podcast script created!")
        print(f"Mode: {document_mode}")
        print(f"Saved to: {OUTPUT_PATH}")

    except Exception as error:
        print(f"\nERROR: Podcast script generation failed: {error}")


if __name__ == "__main__":
    main()