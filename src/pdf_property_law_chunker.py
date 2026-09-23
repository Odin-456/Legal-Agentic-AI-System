from pathlib import Path
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "australian_property_law"
    / "C2024C00719.pdf"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pdf_property_law"
    / "C2024C00719_extracted.txt"
)


def extract_pdf_text(pdf_path):
    print("=" * 80)
    print("PDF PROPERTY-LAW PARSER")
    print("=" * 80)
    print()

    if not pdf_path.exists():
        print("ERROR: PDF file not found:")
        print(pdf_path)
        return None

    reader = PdfReader(str(pdf_path))

    print(f"PDF: {pdf_path.name}")
    print(f"Pages: {len(reader.pages)}")
    print()

    all_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        print(
            f"Extracting page {page_number}/{len(reader.pages)}..."
        )

        text = page.extract_text()

        if text:
            all_text.append(
                f"\n\n===== PAGE {page_number} =====\n\n"
            )
            all_text.append(text)

    return "".join(all_text)


def save_text(text, output_path):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(text)

    print()
    print("=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print()
    print(f"Saved to: {output_path}")
    print(f"Characters: {len(text):,}")
    print()
    print("First 3,000 characters:")
    print("-" * 80)
    print(text[:3000])
    print("-" * 80)


if __name__ == "__main__":

    extracted_text = extract_pdf_text(PDF_FILE)

    if extracted_text:
        save_text(
            extracted_text,
            OUTPUT_FILE
        )