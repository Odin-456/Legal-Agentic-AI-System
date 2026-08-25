import json
import re
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("data/processed/retrieval/title-7")
OUTPUT_DIR = Path("data/processed/chunks/title-7")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Maximum approximate number of words per chunk.
#
# We use words here rather than tokens because this preprocessing
# stage does not depend on a particular embedding model tokenizer.
MAX_WORDS = 250

# Number of words repeated between chunks.
OVERLAP_WORDS = 40


# ============================================================
# TEXT UTILITIES
# ============================================================

def count_words(text):
    """Return the approximate number of words in text."""

    if not text:
        return 0

    return len(text.split())


def split_text(text, max_words=MAX_WORDS, overlap_words=OVERLAP_WORDS):
    """
    Split long text into overlapping word-based chunks.

    Short text is returned unchanged.

    Example:

        max_words = 250
        overlap_words = 40

    Chunk 1:
        words 1-250

    Chunk 2:
        words 211-460

    etc.
    """

    words = text.split()

    if len(words) <= max_words:
        return [text]

    if overlap_words >= max_words:
        raise ValueError(
            "OVERLAP_WORDS must be smaller than MAX_WORDS."
        )

    chunks = []

    start = 0

    while start < len(words):

        end = start + max_words

        chunk_words = words[start:end]

        if not chunk_words:
            break

        chunks.append(
            " ".join(chunk_words)
        )

        # If this is the final chunk, stop.
        if end >= len(words):
            break

        # Move forward while keeping overlap.
        start = end - overlap_words

    return chunks


# ============================================================
# CREATE RETRIEVAL TEXT FOR A CHUNK
# ============================================================

def build_retrieval_text(
    metadata,
    text,
    hierarchy
):
    """
    Construct retrieval text while preserving legal context.
    """

    parts = []

    # --------------------------------------------------------
    # Citation
    # --------------------------------------------------------

    citation = metadata.get("citation")

    if citation:
        parts.append(
            f"Citation: {citation}"
        )

    # --------------------------------------------------------
    # Part
    # --------------------------------------------------------

    part = metadata.get("part")

    if part:
        parts.append(
            f"Part: {part}"
        )

    # --------------------------------------------------------
    # Part title
    # --------------------------------------------------------

    part_title = metadata.get("part_title")

    if part_title:
        parts.append(
            f"Part Title: {part_title}"
        )

    # --------------------------------------------------------
    # Section
    # --------------------------------------------------------

    section = metadata.get("section")

    if section:
        parts.append(
            f"Section: {section}"
        )

    # --------------------------------------------------------
    # Section title
    # --------------------------------------------------------

    section_title = metadata.get("section_title")

    if section_title:
        parts.append(
            f"Section Title: {section_title}"
        )

    # --------------------------------------------------------
    # Legal hierarchy
    # --------------------------------------------------------

    hierarchy_values = []

    for level in [
        "paragraph",
        "subparagraph",
        "clause",
        "subclause",
    ]:

        value = hierarchy.get(level)

        if value:
            hierarchy_values.append(value)

    if hierarchy_values:

        parts.append(
            "Legal Hierarchy: "
            + " ".join(hierarchy_values)
        )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    parts.append(
        f"Text: {text}"
    )

    return "\n".join(parts)


# ============================================================
# PROCESS ONE DOCUMENT
# ============================================================

def process_document(document, source_file):
    """
    Convert one retrieval document into one or more chunks.
    """

    text = document.get("text", "")

    metadata = document.get(
        "metadata",
        {}
    )

    if not text.strip():

        return []

    # --------------------------------------------------------
    # Preserve legal hierarchy
    # --------------------------------------------------------

    hierarchy = {
        "paragraph": metadata.get("paragraph"),
        "subparagraph": metadata.get("subparagraph"),
        "clause": metadata.get("clause"),
        "subclause": metadata.get("subclause"),
    }

    # --------------------------------------------------------
    # Split text if necessary
    # --------------------------------------------------------

    text_chunks = split_text(text)

    total_chunks = len(text_chunks)

    output_documents = []

    for chunk_index, chunk_text in enumerate(
        text_chunks
    ):

        # ----------------------------------------------------
        # Copy metadata
        # ----------------------------------------------------

        chunk_metadata = metadata.copy()

        # ----------------------------------------------------
        # Add chunk metadata
        # ----------------------------------------------------

        chunk_metadata["source_file"] = source_file

        chunk_metadata["chunk_index"] = (
            chunk_index
        )

        chunk_metadata["total_chunks"] = (
            total_chunks
        )

        chunk_metadata["chunk_word_count"] = (
            count_words(chunk_text)
        )

        # ----------------------------------------------------
        # Create a stable chunk ID
        # ----------------------------------------------------

        citation = metadata.get(
            "citation",
            "unknown"
        )

        paragraph = metadata.get(
            "paragraph"
        ) or "none"

        subparagraph = metadata.get(
            "subparagraph"
        ) or "none"

        clause = metadata.get(
            "clause"
        ) or "none"

        subclause = metadata.get(
            "subclause"
        ) or "none"

        chunk_id = (
            f"{citation}"
            f"_{paragraph}"
            f"_{subparagraph}"
            f"_{clause}"
            f"_{subclause}"
            f"_chunk_{chunk_index}"
        )

        # Remove characters that are inconvenient
        # in IDs.

        chunk_id = re.sub(
            r"[^A-Za-z0-9_.()-]+",
            "_",
            chunk_id
        )

        chunk_metadata["chunk_id"] = chunk_id

        # ----------------------------------------------------
        # Create retrieval text
        # ----------------------------------------------------

        retrieval_text = build_retrieval_text(
            chunk_metadata,
            chunk_text,
            hierarchy
        )

        # ----------------------------------------------------
        # Create final chunk
        # ----------------------------------------------------

        chunk_document = {
            "text": chunk_text,

            "retrieval_text": retrieval_text,

            "metadata": chunk_metadata,
        }

        output_documents.append(
            chunk_document
        )

    return output_documents


# ============================================================
# PROCESS ONE JSON FILE
# ============================================================

def process_file(input_file, output_file):
    """
    Process one retrieval JSON file.
    """

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as f:

        documents = json.load(f)

    all_chunks = []

    for document in documents:

        chunks = process_document(
            document,
            input_file.name
        )

        all_chunks.extend(
            chunks
        )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    return len(documents), len(all_chunks)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("eCFR STRUCTURE-AWARE CHUNKING")
    print("=" * 60)

    print(
        f"\nInput directory:"
        f" {INPUT_DIR}"
    )

    print(
        f"Output directory:"
        f" {OUTPUT_DIR}"
    )

    print(
        f"\nMaximum words per chunk:"
        f" {MAX_WORDS}"
    )

    print(
        f"Overlap words:"
        f" {OVERLAP_WORDS}"
    )

    # --------------------------------------------------------
    # Find input files
    # --------------------------------------------------------

    json_files = sorted(
        INPUT_DIR.glob("*.json")
    )

    print(
        f"\nJSON files found:"
        f" {len(json_files)}"
    )

    if not json_files:

        print(
            "\nERROR:"
            " No retrieval JSON files found."
        )

        return

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_documents = 0
    total_chunks = 0
    files_processed = 0
    files_failed = 0

    # --------------------------------------------------------
    # Process files
    # --------------------------------------------------------

    for number, input_file in enumerate(
        json_files,
        start=1
    ):

        print(
            f"\n[{number}/{len(json_files)}]"
            f" Processing: {input_file.name}"
        )

        output_file = (
            OUTPUT_DIR /
            input_file.name
        )

        try:

            documents, chunks = process_file(
                input_file,
                output_file
            )

            total_documents += documents
            total_chunks += chunks
            files_processed += 1

            print(
                f"  Documents:"
                f" {documents}"
            )

            print(
                f"  Chunks:"
                f" {chunks}"
            )

            print(
                f"  Saved to:"
                f" {output_file}"
            )

        except Exception as e:

            files_failed += 1

            print(
                f"  ERROR:"
                f" {e}"
            )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print("\n")

    print("=" * 60)
    print("CHUNKING COMPLETE")
    print("=" * 60)

    print(
        f"JSON files found:"
        f"       {len(json_files)}"
    )

    print(
        f"Files processed:"
        f"       {files_processed}"
    )

    print(
        f"Files failed:"
        f"          {files_failed}"
    )

    print(
        f"Original documents:"
        f"   {total_documents}"
    )

    print(
        f"Total chunks:"
        f"          {total_chunks}"
    )

    print(
        f"Output directory:"
        f"      {OUTPUT_DIR}"
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()