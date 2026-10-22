from pathlib import Path
from langchain_core.documents import Document
from property_law_parser import parse_property_law_corpus

# ============================================================
# OUTPUT PATH
# ============================================================

OUTPUT_DIR = Path(
    "data/processed/chunks"
)

# ============================================================
# CREATE CHUNKS
# ============================================================

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

