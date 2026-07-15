import base64
import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

IMAGE_FOLDER = PROJECT_ROOT / "data" / "images"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "extracted_text"
    / "output.txt"
)

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def encode_image_as_data_url(image_path: Path) -> str:
    """Convert a local image into a base64 data URL."""

    mime_type, _ = mimetypes.guess_type(image_path.name)

    if not mime_type:
        raise ValueError(
            f"Could not determine the image type: {image_path.name}"
        )

    image_bytes = image_path.read_bytes()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    return f"data:{mime_type};base64,{encoded_image}"


def find_images(folder_path: Path) -> list[Path]:
    """Return supported images in alphabetical order."""

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Image folder was not found: {folder_path}"
        )

    image_paths = sorted(
        path
        for path in folder_path.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        raise FileNotFoundError(
            "No JPG, JPEG, PNG or WEBP images were found."
        )

    return image_paths


def extract_text_from_image(
    client: OpenAI,
    image_path: Path,
) -> str:
    """Extract readable text from one image."""

    image_data_url = encode_image_as_data_url(image_path)

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=(
            "You are an accurate document transcription assistant. "
            "Extract the text visible in the supplied image."
        ),
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Transcribe all readable text from this page.\n\n"
                            "Rules:\n"
                            "- Preserve headings and paragraph order.\n"
                            "- Preserve bullet points and numbered lists.\n"
                            "- Preserve important symbols, dates and figures.\n"
                            "- Do not summarise the text.\n"
                            "- Do not add information not shown in the image.\n"
                            "- If a word is unreadable, write [unclear].\n"
                            "- Ignore decorative elements that contain no "
                            "useful information."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                        "detail": "high",
                    },
                ],
            }
        ],
        max_output_tokens=3000,
    )

    extracted_text = response.output_text.strip()

    if not extracted_text:
        return "[No readable text was detected in this image.]"

    return extracted_text


def save_extracted_pages(
    extracted_pages: list[tuple[str, str]],
    output_path: Path,
) -> None:
    """Save all image-page text into the shared extraction file."""

    sections: list[str] = []

    for page_number, (filename, text) in enumerate(
        extracted_pages,
        start=1,
    ):
        sections.append(
            f"========== IMAGE PAGE {page_number} ==========\n"
            f"Source image: {filename}\n\n"
            f"{text}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n\n".join(sections),
        encoding="utf-8",
    )


def main() -> None:
    load_dotenv(dotenv_path=ENV_PATH)

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ OPENAI_API_KEY was not found.")
        return

    try:
        image_paths = find_images(IMAGE_FOLDER)

        print(f"Found {len(image_paths)} image page(s).")

        client = OpenAI(api_key=api_key)
        extracted_pages: list[tuple[str, str]] = []

        for page_number, image_path in enumerate(
            image_paths,
            start=1,
        ):
            print(
                f"Reading image {page_number}/"
                f"{len(image_paths)}: {image_path.name}"
            )

            page_text = extract_text_from_image(
                client=client,
                image_path=image_path,
            )

            extracted_pages.append(
                (image_path.name, page_text)
            )

        save_extracted_pages(
            extracted_pages=extracted_pages,
            output_path=OUTPUT_PATH,
        )

        print("\n✅ Image text extracted successfully!")
        print(f"Saved to: {OUTPUT_PATH}")

        print(
            "\nYou can now run the existing summary, notes, "
            "flashcard, quiz and podcast generators."
        )

    except Exception as error:
        print(f"\n❌ Image extraction failed: {error}")


if __name__ == "__main__":
    main()