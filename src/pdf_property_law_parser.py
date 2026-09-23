from pathlib import Path
from pypdf import PdfReader


# ---------------------------------------------------------
# File locations
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Extract text from PDF
# ---------------------------------------------------------

def extract_pdf_text(pdf_path):
    print("=" * 80)
    print("PDF PROPERTY-LAW PARSER")
    print("=" * 80)
    print()

    print(f"PDF file: {pdf_path}")
    print()

    if not pdf_path.exists():
        print("ERROR: PDF file was not found.")
        print()
        print("Make sure the file is located at:")
        print(pdf_path)
        return None

    print("Loading PDF...")

    reader = PdfReader(str(pdf_path))

    print(f"Pages found: {len(reader.pages)}")
    print()

    all_text = []

    for page_number, page in enumerate(reader.pages, start=1):

        print(f"Extracting page {page_number}/{len(reader.pages)}...")

        text = page.extract_text()

        if text:
            all_text.append(
                f"\n\n===== PAGE {page_number} =====\n\n"
            )
            all_text.append(text)

    extracted_text = "".join(all_text)

    return extracted_text


# ---------------------------------------------------------
# Save extracted text
# ---------------------------------------------------------

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

    print(f"Saved extracted text to:")
    print(output_path)

    print()
    print(f"Characters extracted: {len(text):,}")

    print()
    print("First 3,000 characters:")
    print("-" * 80)
    print(text[:3000])
    print("-" * 80)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    extracted_text = extract_pdf_text(PDF_FILE)

    if extracted_text:
        save_text(
            extracted_text,
            OUTPUT_FILE
        )