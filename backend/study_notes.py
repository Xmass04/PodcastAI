import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
INPUT_PATH = PROJECT_ROOT / "data" / "extracted_text" / "output.txt"
OUTPUT_PATH = PROJECT_ROOT / "data" / "summaries" / "study_notes.txt"


def generate_study_notes(text: str) -> str:
    """Generate structured study notes from extracted document text."""

    load_dotenv(dotenv_path=ENV_PATH)

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY was not found. Check the .env file."
        )

    if not text.strip():
        raise ValueError("The extracted document text is empty.")

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model="gpt-5.5",
        instructions=(
            "You are an expert educational tutor. "
            "Create accurate study notes using only the supplied document. "
            "Do not invent facts."
        ),
        input=(
            "Turn the following document into structured revision notes.\n\n"
            "Use this exact structure where relevant:\n"
            "# Topic Overview\n"
            "# Main Concepts\n"
            "# Important Facts and Figures\n"
            "# Key Terms and Definitions\n"
            "# Processes or Steps\n"
            "# Common Misunderstandings\n"
            "# Questions a Student Should Be Able to Answer\n\n"
            "Requirements:\n"
            "- Use clear headings.\n"
            "- Use concise bullet points.\n"
            "- Explain difficult language simply.\n"
            "- Preserve important dates, values and requirements.\n"
            "- Exclude contents pages, repeated headers and irrelevant text.\n"
            "- State when the source does not contain enough information.\n\n"
            f"DOCUMENT:\n{text}"
        ),
        max_output_tokens=2200,
    )

    notes = response.output_text.strip()

    if not notes:
        raise RuntimeError("The AI returned empty study notes.")

    return notes


def main() -> None:
    if not INPUT_PATH.exists():
        print("❌ Extracted text was not found.")
        print("Run backend/pdf_reader.py first.")
        return

    text = INPUT_PATH.read_text(encoding="utf-8")

    try:
        print("Generating structured study notes...")

        notes = generate_study_notes(text)

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(notes, encoding="utf-8")

        print("\n✅ Study notes created successfully!")
        print(f"Saved to: {OUTPUT_PATH}")

    except Exception as error:
        print(f"\n❌ Study-note generation failed: {error}")


if __name__ == "__main__":
    main()