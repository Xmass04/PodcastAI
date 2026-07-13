import fitz


def extract_text(pdf_path):
    document = fitz.open(pdf_path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text


if __name__ == "__main__":
    pdf = input("Enter PDF path: ")

    extracted = extract_text(pdf)

    print("\n----- Extracted Text -----\n")
    print(extracted)
    