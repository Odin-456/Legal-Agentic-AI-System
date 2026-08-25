import json
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("data/processed/ecfr/title-7")
OUTPUT_DIR = Path("data/processed/retrieval/title-7")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# BUILD RETRIEVAL TEXT
# ============================================================

def build_retrieval_text(document):
    """
    Build the text that will later be used for embedding.

    The legal hierarchy is included in the text so that
    semantic retrieval can use both the legal content and
    its structural context.
    """

    text = document.get("text", "").strip()
    metadata = document.get("metadata", {})

    title = metadata.get("title")
    part = metadata.get("part")
    part_title = metadata.get("part_title")
    section = metadata.get("section")
    section_title = metadata.get("section_title")

    paragraph = metadata.get("paragraph")
    subparagraph = metadata.get("subparagraph")
    clause = metadata.get("clause")
    subclause = metadata.get("subclause")

    citation = metadata.get("citation")

    parts = []

    # --------------------------------------------------------
    # Citation
    # --------------------------------------------------------

    if citation:
        parts.append(f"Citation: {citation}")

    # --------------------------------------------------------
    # Part
    # --------------------------------------------------------

    if part:
        parts.append(f"Part: {part}")

    if part_title:
        parts.append(f"Part Title: {part_title}")

    # --------------------------------------------------------
    # Section
    # --------------------------------------------------------

    if section:
        parts.append(f"Section: {section}")

    if section_title:
        parts.append(f"Section Title: {section_title}")

    # --------------------------------------------------------
    # Legal hierarchy
    # --------------------------------------------------------

    hierarchy = []

    if paragraph:
        hierarchy.append(paragraph)

    if subparagraph:
        hierarchy.append(subparagraph)

    if clause:
        hierarchy.append(clause)

    if subclause:
        hierarchy.append(subclause)

    if hierarchy:
        parts.append(
            "Legal Hierarchy: " + " ".join(hierarchy)
        )

    # --------------------------------------------------------
    # Legal text
    # --------------------------------------------------------

    if text:
        parts.append(f"Text: {text}")

    return "\n".join(parts)


# ============================================================
# PREPARE ONE DOCUMENT
# ============================================================

def prepare_document(document):
    """
    Convert one parsed eCFR document into a retrieval-ready
    document while preserving the original metadata.
    """

    metadata = document.get("metadata", {})

    retrieval_text = build_retrieval_text(document)

    return {
        "text": document.get("text", ""),
        "retrieval_text": retrieval_text,
        "metadata": metadata,
    }


# ============================================================
# PROCESS ONE JSON FILE
# ============================================================

def process_file(input_file):
    """
    Process one JSON file and save the retrieval-ready
    version.
    """

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as f:

        documents = json.load(f)

    prepared_documents = []

    for document in documents:

        prepared_document = prepare_document(
            document
        )

        prepared_documents.append(
            prepared_document
        )

    output_file = (
        OUTPUT_DIR /
        input_file.name
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            prepared_documents,
            f,
            ensure_ascii=False,
            indent=2
        )

    return len(prepared_documents), output_file


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("eCFR RETRIEVAL DOCUMENT PREPARATION")
    print("=" * 60)

    # --------------------------------------------------------
    # FIND JSON FILES
    # --------------------------------------------------------

    json_files = sorted(
        INPUT_DIR.glob("*.json")
    )

    print(
        f"\nJSON files found: {len(json_files)}"
    )

    if not json_files:

        print(
            "ERROR: No JSON files found."
        )

        return

    # --------------------------------------------------------
    # PROCESS FILES
    # --------------------------------------------------------

    total_documents = 0
    processed_files = 0
    failed_files = 0

    for index, json_file in enumerate(
        json_files,
        start=1
    ):

        print(
            f"\n[{index}/{len(json_files)}] "
            f"Processing: {json_file.name}"
        )

        try:

            document_count, output_file = process_file(
                json_file
            )

            total_documents += document_count
            processed_files += 1

            print(
                f"  Documents prepared: {document_count}"
            )

            print(
                f"  Saved to: {output_file}"
            )

        except Exception as e:

            failed_files += 1

            print(
                f"  ERROR: {e}"
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("PREPARATION COMPLETE")
    print("=" * 60)

    print(
        f"JSON files found:       {len(json_files)}"
    )

    print(
        f"Files processed:        {processed_files}"
    )

    print(
        f"Files failed:           {failed_files}"
    )

    print(
        f"Total documents:        {total_documents}"
    )

    print(
        f"Output directory:       {OUTPUT_DIR}"
    )

    if failed_files == 0:

        print("\nSUCCESS:")
        print(
            "All documents were prepared successfully."
        )

    else:

        print("\nWARNING:")
        print(
            "Some files failed during preparation."
        )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()