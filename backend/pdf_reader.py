import sys
from pathlib import Path

import fitz


# Force Windows terminals and Streamlit subprocesses to use UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace",
    )

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(
        encoding="utf-8",
        errors="replace",
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "extracted_text"
    / "output.txt"
)


def extract_text(pdf_path: Path) -> str:
    """Extract selectable text from every page of a PDF."""

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            "The selected file is not a PDF."
        )

    extracted_pages: list[str] = []

    with fitz.open(pdf_path) as document:
        if document.page_count == 0:
            raise ValueError(
                "The PDF contains no pages."
            )

        for page_number, page in enumerate(
            document,
            start=1,
        ):
            page_text = page.get_text("text").strip()

            page_heading = (
                f"========== PAGE {page_number} =========="
            )

            if page_text:
                extracted_pages.append(
                    f"{page_heading}\n\n{page_text}"
                )
            else:
                extracted_pages.append(
                    f"{page_heading}\n\n"
                    "[No selectable text found on this page.]"
                )

    return "\n\n".join(extracted_pages)


def save_text(
    extracted_text: str,
    output_path: Path,
) -> None:
    """Save extracted text using UTF-8 encoding."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        extracted_text,
        encoding="utf-8",
    )


def main() -> None:
    raw_path = input(
        "Enter PDF path: "
    ).strip().strip('"')

    if not raw_path:
        print("ERROR: No PDF path was provided.")
        return

    pdf_path = Path(raw_path)

    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path

    try:
        extracted_text = extract_text(pdf_path)

        save_text(
            extracted_text=extracted_text,
            output_path=OUTPUT_PATH,
        )

        print("\nSUCCESS: Text extracted successfully!")
        print(f"Saved to: {OUTPUT_PATH}")

        if (
            "[No selectable text found on this page.]"
            in extracted_text
        ):
            print(
                "\nNOTICE: One or more pages may be scanned "
                "images and could require OCR."
            )

    except FileNotFoundError as error:
        print(f"\nERROR: {error}")

    except fitz.FileDataError:
        print(
            "\nERROR: The selected file is damaged "
            "or is not a valid PDF."
        )

    except PermissionError:
        print(
            "\nERROR: Permission was denied while reading "
            "or saving the file."
        )

    except Exception as error:
        print(f"\nERROR: Unexpected error: {error}")


if __name__ == "__main__":
    main()