from pathlib import Path

import fitz


def extract_text(pdf_path: Path) -> str:
    """Extract text from every page of a PDF."""

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    extracted_pages: list[str] = []

    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text("text").strip()

            extracted_pages.append(
                f"\n========== PAGE {page_number} ==========\n"
            )

            if page_text:
                extracted_pages.append(page_text)
            else:
                extracted_pages.append(
                    "[No selectable text found on this page]"
                )

    return "\n".join(extracted_pages)


def save_text(text: str, output_path: Path) -> None:
    """Save extracted text to a UTF-8 text file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    raw_path = input("Enter PDF path: ").strip().strip('"')
    pdf_path = Path(raw_path)
    output_path = Path("data/extracted_text/output.txt")

    try:
        extracted_text = extract_text(pdf_path)
        save_text(extracted_text, output_path)

        print("\n✅ Text extracted successfully!")
        print(f"Saved to: {output_path}")

    except FileNotFoundError as error:
        print(f"\n❌ {error}")
    except fitz.FileDataError:
        print("\n❌ The file is damaged or is not a valid PDF.")
    except Exception as error:
        print(f"\n❌ Unexpected error: {error}")


if __name__ == "__main__":
    main()
    