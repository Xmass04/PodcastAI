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
    / "flashcards"
    / "flashcards.json"
)


class Flashcard(BaseModel):
    topic: str = Field(
        description="The document topic covered by this card."
    )
    priority: Literal[
        "must_know",
        "important",
        "extra"
    ]
    difficulty: Literal[
        "easy",
        "medium",
        "hard"
    ]
    question: str
    answer: str
    exam_tip: str


class FlashcardPack(BaseModel):
    title: str
    flashcards: list[Flashcard]


def generate_flashcards(
    text: str,
    number_of_cards: int = 20,
) -> FlashcardPack:
    """Generate structured revision flashcards from document text."""

    load_dotenv(dotenv_path=ENV_PATH)

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY was not found. Check your .env file."
        )

    if not text.strip():
        raise ValueError("The extracted document text is empty.")

    client = OpenAI(api_key=api_key)

    response = client.responses.parse(
        model="gpt-5.6-luna",
        input=[
            {
                "role": "system",
                "content": (
                    "You are an expert educational tutor creating "
                    "accurate revision flashcards. Use only the supplied "
                    "document. Do not invent information."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Create exactly {number_of_cards} useful flashcards "
                    "from the document below.\n\n"
                    "Requirements:\n"
                    "- Cover the most important topics across the document.\n"
                    "- Avoid duplicate questions.\n"
                    "- Keep questions clear and specific.\n"
                    "- Keep answers concise but complete.\n"
                    "- Use must_know for essential knowledge.\n"
                    "- Use important for useful supporting knowledge.\n"
                    "- Use extra only for lower-priority detail.\n"
                    "- Give a practical exam or revision tip for every card.\n"
                    "- If the document is not exam-based, use a learning tip.\n"
                    "- Do not use information absent from the source.\n\n"
                    f"DOCUMENT:\n{text}"
                ),
            },
        ],
        text_format=FlashcardPack,
    )

    flashcard_pack = response.output_parsed

    if flashcard_pack is None:
        raise RuntimeError(
            "The AI did not return a valid flashcard pack."
        )

    return flashcard_pack


def save_flashcards(
    flashcard_pack: FlashcardPack,
    output_path: Path,
) -> None:
    """Save the flashcards as readable JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            flashcard_pack.model_dump(),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    if not INPUT_PATH.exists():
        print("❌ Extracted text was not found.")
        print("Run backend/pdf_reader.py first.")
        return

    text = INPUT_PATH.read_text(
        encoding="utf-8"
    )

    try:
        print("Generating flashcards...")

        flashcard_pack = generate_flashcards(
            text=text,
            number_of_cards=20,
        )

        save_flashcards(
            flashcard_pack,
            OUTPUT_PATH,
        )

        print("\n✅ Flashcards created successfully!")
        print(
            f"Created: {len(flashcard_pack.flashcards)} cards"
        )
        print(f"Saved to: {OUTPUT_PATH}")

    except Exception as error:
        print(
            f"\n❌ Flashcard generation failed: {error}"
        )


if __name__ == "__main__":
    main()