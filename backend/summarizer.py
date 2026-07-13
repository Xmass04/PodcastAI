from collections import Counter
from pathlib import Path
import re


STOP_WORDS = {
    "the", "and", "for", "that", "with", "this", "from", "into", "their",
    "there", "they", "have", "has", "had", "was", "were", "are", "is", "be",
    "been", "being", "of", "to", "in", "on", "at", "by", "as", "an", "a",
    "or", "but", "if", "then", "than", "it", "its", "we", "you", "your",
    "can", "will", "may", "must", "should", "would", "could", "not"
}


def clean_text(text: str) -> str:
    """Remove page markers, repeated whitespace and table-of-contents dot leaders."""

    text = re.sub(
        r"=+\s*PAGE\s+\d+\s*=+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\.{4,}", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_into_sentences(text: str) -> list[str]:
    """Split document text into usable sentences."""

    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [
        sentence.strip()
        for sentence in sentences
        if 40 <= len(sentence.strip()) <= 500
    ]


def get_word_frequencies(sentences: list[str]) -> Counter:
    """Count useful words across all sentences."""

    words: list[str] = []

    for sentence in sentences:
        sentence_words = re.findall(r"\b[a-zA-Z]{3,}\b", sentence.lower())

        for word in sentence_words:
            if word not in STOP_WORDS:
                words.append(word)

    return Counter(words)


def score_sentence(
    sentence: str,
    word_frequencies: Counter,
) -> float:
    """Score a sentence based on important words it contains."""

    words = re.findall(r"\b[a-zA-Z]{3,}\b", sentence.lower())

    if not words:
        return 0.0

    score = sum(word_frequencies[word] for word in words)
    return score / len(words)


def create_summary(text: str, sentence_limit: int = 8) -> str:
    """Create an extractive summary from the most important sentences."""

    cleaned_text = clean_text(text)
    sentences = split_into_sentences(cleaned_text)

    if not sentences:
        return "No suitable text was found for summarisation."

    word_frequencies = get_word_frequencies(sentences)

    scored_sentences = [
        (score_sentence(sentence, word_frequencies), index, sentence)
        for index, sentence in enumerate(sentences)
    ]

    top_sentences = sorted(
        scored_sentences,
        key=lambda item: item[0],
        reverse=True,
    )[:sentence_limit]

    top_sentences.sort(key=lambda item: item[1])

    return "\n\n".join(
        f"• {sentence}"
        for _, _, sentence in top_sentences
    )


def main() -> None:
    input_path = Path("data/extracted_text/output.txt")
    output_path = Path("data/summaries/summary.txt")

    if not input_path.exists():
        print("❌ Extracted text was not found.")
        print("Run the PDF reader first.")
        return

    extracted_text = input_path.read_text(encoding="utf-8")

    if not extracted_text.strip():
        print("❌ The extracted text file is empty.")
        return

    summary = create_summary(extracted_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary, encoding="utf-8")

    print("\n✅ Smarter summary created successfully!")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
    
    