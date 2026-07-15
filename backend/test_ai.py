import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def main() -> None:
    load_dotenv(dotenv_path=ENV_PATH)

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ OPENAI_API_KEY was not found.")
        print(f"Checked: {ENV_PATH}")
        return

    client = OpenAI(api_key=api_key)

    try:
        response = client.responses.create(
            model="gpt-5.5",
            instructions=(
                "You are testing an educational study application. "
                "Reply clearly and briefly."
            ),
            input=(
                "Write one sentence confirming that the "
                "PodcastAI connection is working."
            ),
            max_output_tokens=100,
        )

        print("\n✅ AI connection successful!\n")
        print(response.output_text)

    except Exception as error:
        print("\n❌ AI request failed.")
        print(error)


if __name__ == "__main__":
    main()