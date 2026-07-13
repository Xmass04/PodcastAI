import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

STUDY_NOTES_PATH = (
    PROJECT_ROOT
    / "data"
    / "summaries"
    / "study_notes.txt"
)

EXTRACTED_TEXT_PATH = (
    PROJECT_ROOT
    / "data"
    / "extracted_text"
    / "output.txt"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "quizzes"
    / "quiz.json"
)


Difficulty = Literal["easy", "medium", "hard"]


class MultipleChoiceQuestion(BaseModel):
    topic: str
    difficulty: Difficulty
    question: str

    options: list[str] = Field(
        min_length=4,
        max_length=4,
        description="Exactly four possible answers.",
    )

    correct_answer: str = Field(
        description="Must exactly match one item from options."
    )

    explanation: str


class TrueFalseQuestion(BaseModel):
    topic: str
    difficulty: Difficulty
    statement: str
    correct_answer: bool
    explanation: str


class ShortAnswerQuestion(BaseModel):
    topic: str
    difficulty: Difficulty
    question: str
    model_answer: str
    marking_points: list[str]


class QuizPack(BaseModel):
    title: str
    instructions: str
    multiple_choice: list[MultipleChoiceQuestion]
    true_false: list[TrueFalseQuestion]
    short_answer: list[ShortAnswerQuestion]


def load_source_text() -> str:
    """
    Prefer structured study notes because they are cleaner and cheaper
    to process. Fall back to the original extracted PDF text.
    """

    if STUDY_NOTES_PATH.exists():
        text = STUDY_NOTES_PATH.read_text(encoding="utf-8")

        if text.strip():
            print("Using structured study notes.")
            return text

    if EXTRACTED_TEXT_PATH.exists():
        text = EXTRACTED_TEXT_PATH.read_text(encoding="utf-8")

        if text.strip():
            print("Using extracted PDF text.")
            return text

    raise FileNotFoundError(
        "No usable study notes or extracted PDF text were found."
    )


def generate_quiz(text: str) -> QuizPack:
    """Generate a structured educational quiz."""

    load_dotenv(dotenv_path=ENV_PATH)

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY was not found. Check the .env file."
        )

    if not text.strip():
        raise ValueError("The source text is empty.")

    client = OpenAI(api_key=api_key)

    response = client.responses.parse(
        model="gpt-5.6-luna",
        input=[
            {
                "role": "system",
                "content": (
                    "You are an expert educational assessment writer. "
                    "Create accurate questions using only the supplied "
                    "study material. Never invent unsupported facts."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Create a revision quiz from the material below.\n\n"
                    "Produce exactly:\n"
                    "- 10 multiple-choice questions\n"
                    "- 5 true-or-false questions\n"
                    "- 5 short-answer questions\n\n"
                    "Requirements:\n"
                    "- Cover different topics across the material.\n"
                    "- Avoid duplicate questions.\n"
                    "- Include a mixture of easy, medium and hard items.\n"
                    "- Each multiple-choice item must have four distinct "
                    "options and only one correct answer.\n"
                    "- The correct answer must exactly match one option.\n"
                    "- False statements must be believable, not silly.\n"
                    "- Short-answer questions must include clear marking "
                    "points.\n"
                    "- Explain every answer briefly.\n"
                    "- Do not test information absent from the source.\n\n"
                    f"STUDY MATERIAL:\n{text}"
                ),
            },
        ],
        text_format=QuizPack,
    )

    quiz = response.output_parsed

    if quiz is None:
        raise RuntimeError("The AI did not return a valid quiz.")

    validate_quiz(quiz)

    return quiz


def validate_quiz(quiz: QuizPack) -> None:
    """Perform additional checks before saving the quiz."""

    if len(quiz.multiple_choice) != 10:
        raise ValueError(
            "The generated quiz does not contain 10 multiple-choice "
            "questions."
        )

    if len(quiz.true_false) != 5:
        raise ValueError(
            "The generated quiz does not contain 5 true/false questions."
        )

    if len(quiz.short_answer) != 5:
        raise ValueError(
            "The generated quiz does not contain 5 short-answer questions."
        )

    for question in quiz.multiple_choice:
        if question.correct_answer not in question.options:
            raise ValueError(
                "A multiple-choice correct answer is missing from its "
                "options."
            )

        if len(set(question.options)) != 4:
            raise ValueError(
                "A multiple-choice question contains duplicate options."
            )


def save_quiz(quiz: QuizPack) -> None:
    """Save the quiz as formatted JSON."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            quiz.model_dump(),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    try:
        source_text = load_source_text()

        print("Generating quiz...")

        quiz = generate_quiz(source_text)
        save_quiz(quiz)

        total_questions = (
            len(quiz.multiple_choice)
            + len(quiz.true_false)
            + len(quiz.short_answer)
        )

        print("\n✅ Quiz created successfully!")
        print(f"Created: {total_questions} questions")
        print(f"Saved to: {OUTPUT_PATH}")

    except Exception as error:
        print(f"\n❌ Quiz generation failed: {error}")


if __name__ == "__main__":
    main()