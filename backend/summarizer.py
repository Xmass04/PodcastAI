import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def summarize_text(text: str) -> str:
    """Generate a clear study summary using the OpenAI API."""

    load_dotenv(dotenv_path=ENV_PATH)

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY was not found. Check your .env file."
        )

    if not text.strip():
        raise ValueError("The extracted text is empty.")

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model="gpt-5.5",
        instructions=(
            "You are an expert educational study assistant. "
            "Create accurate revision material using only the document "
            "provided by the user."
        ),
        input=(
            "Summarise the following document clearly.\n\n"
            "Requirements:\n"
            "- Begin with a short overview.\n"
            "- Use clear headings.\n"
            "- Use bullet points for key information.\n"
            "- Explain difficult terms simply.\n"
            "- Keep important dates, figures, rules and requirements.\n"
            "- Do not invent information.\n"
            "- Do not include irrelevant table-of-contents text.\n\n"
            f"DOCUMENT:\n{text}"
        ),
        max_output_tokens=1200,
    )

    if not response.output_text:
        raise RuntimeError("The AI returned an empty summary.")

    return response.output_text


if __name__ == "__main__":
    print(
        "This file provides the summarize_text() function.\n"
        "Run backend/ai_summarizer.py to summarise the extracted PDF."
    )