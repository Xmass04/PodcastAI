import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

MODEL_NAME = "gpt-5.5"
MAX_CHARS_PER_CHUNK = 45_000
MAX_CHUNKS = 8
MAX_ATTEMPTS = 3


def _clean_text(text: str) -> str:
    """Normalise extracted document text."""

    cleaned_lines: list[str] = []

    for line in text.splitlines():
        cleaned_line = " ".join(line.split())

        if cleaned_line:
            cleaned_lines.append(cleaned_line)

    return "\n".join(cleaned_lines).strip()


def _split_text(
    text: str,
    chunk_size: int = MAX_CHARS_PER_CHUNK,
) -> list[str]:
    """
    Split a document into manageable chunks without cutting words where
    possible.
    """

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text) and len(chunks) < MAX_CHUNKS:
        end = min(
            start + chunk_size,
            len(text),
        )

        if end < len(text):
            paragraph_break = text.rfind(
                "\n",
                start,
                end,
            )

            sentence_break = text.rfind(
                ". ",
                start,
                end,
            )

            split_position = max(
                paragraph_break,
                sentence_break,
            )

            if split_position > start + (chunk_size // 2):
                end = split_position + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end

    return chunks


def _extract_response_text(response: object) -> str:
    """
    Safely retrieve text from an OpenAI Responses API result.
    """

    output_text = getattr(
        response,
        "output_text",
        None,
    )

    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    collected_text: list[str] = []

    output_items = getattr(
        response,
        "output",
        None,
    )

    if not output_items:
        return ""

    for output_item in output_items:
        content_items = getattr(
            output_item,
            "content",
            None,
        )

        if not content_items:
            continue

        for content_item in content_items:
            text_value = getattr(
                content_item,
                "text",
                None,
            )

            if isinstance(text_value, str) and text_value.strip():
                collected_text.append(
                    text_value.strip()
                )

    return "\n\n".join(collected_text).strip()


def _request_summary(
    client: OpenAI,
    document_text: str,
    instructions: str,
    max_output_tokens: int,
) -> str:
    """
    Send one summary request with retries.
    """

    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.responses.create(
                model=MODEL_NAME,
                instructions=instructions,
                input=document_text,
                reasoning={
                    "effort": "low",
                },
                max_output_tokens=max_output_tokens,
            )

            result = _extract_response_text(
                response
            )

            if result:
                return result

            response_status = getattr(
                response,
                "status",
                "unknown",
            )

            incomplete_details = getattr(
                response,
                "incomplete_details",
                None,
            )

            last_error = RuntimeError(
                "The AI returned no usable text. "
                f"Response status: {response_status}. "
                f"Incomplete details: {incomplete_details}."
            )

        except Exception as error:
            last_error = error

        if attempt < MAX_ATTEMPTS:
            wait_seconds = 2 ** attempt

            print(
                "Summary attempt "
                f"{attempt}/{MAX_ATTEMPTS} failed. "
                f"Retrying in {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)

    if last_error:
        raise RuntimeError(
            "Summary generation failed after "
            f"{MAX_ATTEMPTS} attempts. "
            f"Final error: {type(last_error).__name__}: "
            f"{last_error}"
        ) from last_error

    raise RuntimeError(
        "Summary generation failed for an unknown reason."
    )


def _summarize_single_chunk(
    client: OpenAI,
    chunk: str,
    chunk_number: int,
    total_chunks: int,
) -> str:
    """
    Summarise one section of a larger document.
    """

    instructions = (
        "You are an expert educational study assistant. "
        "Create accurate revision material using only the supplied "
        "document section. Do not invent information."
    )

    prompt = (
        f"DOCUMENT SECTION {chunk_number} OF {total_chunks}\n\n"
        "Summarise this section clearly.\n\n"
        "Requirements:\n"
        "- Capture the important facts and ideas.\n"
        "- Preserve important dates, figures, rules and requirements.\n"
        "- Explain difficult terms simply.\n"
        "- Ignore repeated headers, footers and table-of-contents text.\n"
        "- Do not invent information.\n"
        "- Keep the result concise enough to combine with other sections.\n\n"
        f"DOCUMENT SECTION:\n{chunk}"
    )

    return _request_summary(
        client=client,
        document_text=prompt,
        instructions=instructions,
        max_output_tokens=1600,
    )


def _combine_chunk_summaries(
    client: OpenAI,
    section_summaries: list[str],
) -> str:
    """
    Combine section summaries into one polished final study summary.
    """

    combined_sections = "\n\n".join(
        (
            f"SECTION SUMMARY {index}\n"
            f"{summary}"
        )
        for index, summary in enumerate(
            section_summaries,
            start=1,
        )
    )

    instructions = (
        "You are an expert educational study assistant. "
        "Combine supplied section summaries into one accurate and "
        "well-organised revision summary. Use only the supplied material."
    )

    prompt = (
        "Create one final study summary from the section summaries below.\n\n"
        "Requirements:\n"
        "- Begin with a short overview.\n"
        "- Use clear headings.\n"
        "- Use bullet points for key information.\n"
        "- Explain difficult terms simply.\n"
        "- Preserve important dates, figures, rules and requirements.\n"
        "- Remove repetition.\n"
        "- Do not invent information.\n"
        "- Do not mention chunks or section-summary numbers.\n\n"
        f"SECTION SUMMARIES:\n{combined_sections}"
    )

    return _request_summary(
        client=client,
        document_text=prompt,
        instructions=instructions,
        max_output_tokens=2200,
    )


def summarize_text(text: str) -> str:
    """
    Generate a reliable study summary using the OpenAI Responses API.

    Small documents are summarised in one request. Larger documents are
    divided into manageable sections and then combined.
    """

    load_dotenv(
        dotenv_path=ENV_PATH
    )

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY was not found. "
            "Check the local .env file or the hosted Streamlit Secrets."
        )

    cleaned_text = _clean_text(
        text
    )

    if not cleaned_text:
        raise ValueError(
            "The extracted document text is empty."
        )

    client = OpenAI(
        api_key=api_key,
        timeout=120.0,
        max_retries=2,
    )

    chunks = _split_text(
        cleaned_text
    )

    print(
        f"Preparing summary from {len(chunks)} document section(s)..."
    )

    if len(chunks) == 1:
        instructions = (
            "You are an expert educational study assistant. "
            "Create accurate revision material using only the document "
            "provided by the user. Do not invent information."
        )

        prompt = (
            "Summarise the following document clearly.\n\n"
            "Requirements:\n"
            "- Begin with a short overview.\n"
            "- Use clear headings.\n"
            "- Use bullet points for key information.\n"
            "- Explain difficult terms simply.\n"
            "- Keep important dates, figures, rules and requirements.\n"
            "- Do not invent information.\n"
            "- Ignore irrelevant table-of-contents text.\n"
            "- Remove repeated headers and footers.\n\n"
            f"DOCUMENT:\n{cleaned_text}"
        )

        return _request_summary(
            client=client,
            document_text=prompt,
            instructions=instructions,
            max_output_tokens=2200,
        )

    section_summaries: list[str] = []

    for chunk_number, chunk in enumerate(
        chunks,
        start=1,
    ):
        print(
            "Summarising document section "
            f"{chunk_number}/{len(chunks)}..."
        )

        section_summary = _summarize_single_chunk(
            client=client,
            chunk=chunk,
            chunk_number=chunk_number,
            total_chunks=len(chunks),
        )

        section_summaries.append(
            section_summary
        )

    print(
        "Combining section summaries..."
    )

    final_summary = _combine_chunk_summaries(
        client=client,
        section_summaries=section_summaries,
    )

    if not final_summary.strip():
        raise RuntimeError(
            "The final combined summary was empty."
        )

    return final_summary.strip()


if __name__ == "__main__":
    print(
        "This file provides the summarize_text() function.\n"
        "Run backend/ai_summarizer.py to summarise the extracted PDF."
    )