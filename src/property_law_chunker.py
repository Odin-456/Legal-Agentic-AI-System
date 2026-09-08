from pathlib import Path

from langchain_core.documents import Document

from property_law_parser import parse_property_law_corpus


OUTPUT_DIR = Path(
    "data/processed/chunks"
)


def create_chunks(sections):
    """
    Convert parsed NSW property-law sections into
    LangChain Documents.

    The current parser produces one structured section
    containing subsection/paragraph hierarchy in its text.

    Each section is therefore initially kept as one
    retrieval document.

    Legal hierarchy is preserved in the page content.
    """

    documents = []

    for section in sections:

        act = section["act"]
        section_number = section["section"]
        heading = section["heading"]
        text = section["text"]

        if not text.strip():
            continue

        # ---------------------------------------------------------
        # Build retrieval content
        # ---------------------------------------------------------

        content = (
            f"Act: {act}\n"
            f"Section: {section_number}\n"
            f"Heading: {heading}\n\n"
            f"{text}"
        )

        # ---------------------------------------------------------
        # Create LangChain Document
        # ---------------------------------------------------------

        document = Document(
            page_content=content,
            metadata={
                "act": act,
                "section": section_number,
                "heading": heading,
                "source_type": section[
                    "source_type"
                ],
            },
        )

        documents.append(document)

    return documents


def save_chunks(documents):
    """
    Save chunks to a text file for inspection.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / "property_law_chunks.txt"
    )

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as f:

        for index, document in enumerate(
            documents
        ):

            f.write(
                f"DOCUMENT {index}\n"
            )

            f.write(
                "=" * 80
                + "\n"
            )

            f.write(
                document.page_content
                + "\n\n"
            )

            f.write(
                "METADATA:\n"
            )

            f.write(
                str(document.metadata)
                + "\n\n"
            )

    return output_file


def show_section_42(documents):
    """
    Display Section 42 for testing.
    """

    print(
        "\n"
        + "=" * 80
    )

    print(
        "REAL PROPERTY ACT 1900 — SECTION 42"
    )

    print(
        "=" * 80
    )

    for document in documents:

        metadata = document.metadata

        if (
            metadata.get("act")
            == "Real Property Act 1900"
            and metadata.get("section")
            == "42"
        ):

            print()
            print(
                document.page_content
            )

            print()
            print(
                "Metadata:"
            )

            print(
                document.metadata
            )

            return

    print(
        "Section 42 not found."
    )


if __name__ == "__main__":

    print(
        "Parsing NSW property-law corpus..."
    )

    sections = (
        parse_property_law_corpus()
    )

    print()
    print(
        f"Sections: {len(sections)}"
    )

    print()
    print(
        "Creating retrieval documents..."
    )

    documents = create_chunks(
        sections
    )

    print(
        f"Documents created: "
        f"{len(documents)}"
    )

    output_file = save_chunks(
        documents
    )

    print()
    print(
        f"Saved to: {output_file}"
    )

    show_section_42(
        documents
    )