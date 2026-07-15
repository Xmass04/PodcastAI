from pathlib import Path

from summarizer import summarize_text


INPUT_FILE = Path("data/extracted_text/output.txt")
OUTPUT_FILE = Path("data/summaries/summary.txt")


def main():

    if not INPUT_FILE.exists():
        print("No extracted text found.")
        return

    text = INPUT_FILE.read_text(encoding="utf-8")

    print("Generating summary...")

    summary = summarize_text(text)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_FILE.write_text(summary, encoding="utf-8")

    print("Summary saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()