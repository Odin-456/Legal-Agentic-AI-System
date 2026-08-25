import json
from pathlib import Path
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("data/processed/ecfr/title-7")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("eCFR DATASET ANALYSIS")
    print("=" * 60)

    json_files = sorted(INPUT_DIR.glob("*.json"))

    print(f"\nJSON files: {len(json_files)}")

    total_documents = 0

    paragraph_counts = Counter()
    subparagraph_counts = Counter()
    clause_counts = Counter()
    subclause_counts = Counter()

    section_counts = Counter()

    text_lengths = []

    empty_text = 0

    # --------------------------------------------------------
    # READ ALL JSON FILES
    # --------------------------------------------------------

    for i, json_file in enumerate(json_files, start=1):

        print(
            f"\rProcessing: {i}/{len(json_files)}",
            end=""
        )

        with open(
            json_file,
            "r",
            encoding="utf-8"
        ) as f:

            documents = json.load(f)

        total_documents += len(documents)

        # ----------------------------------------------------
        # ANALYZE DOCUMENTS
        # ----------------------------------------------------

        for document in documents:

            text = document.get("text", "")
            metadata = document.get("metadata", {})

            # Text statistics
            if not text.strip():
                empty_text += 1

            text_lengths.append(len(text))

            # Hierarchy statistics
            paragraph = metadata.get("paragraph")
            subparagraph = metadata.get("subparagraph")
            clause = metadata.get("clause")
            subclause = metadata.get("subclause")

            if paragraph:
                paragraph_counts[paragraph] += 1

            if subparagraph:
                subparagraph_counts[subparagraph] += 1

            if clause:
                clause_counts[clause] += 1

            if subclause:
                subclause_counts[subclause] += 1

            # Section statistics
            section = metadata.get("section")

            if section:
                section_counts[section] += 1

    print("\n")

    # ========================================================
    # RESULTS
    # ========================================================

    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    print(f"\nTotal JSON files:       {len(json_files)}")
    print(f"Total documents:        {total_documents}")
    print(f"Empty documents:        {empty_text}")

    # --------------------------------------------------------
    # TEXT LENGTH
    # --------------------------------------------------------

    if text_lengths:

        average_length = (
            sum(text_lengths)
            / len(text_lengths)
        )

        shortest = min(text_lengths)
        longest = max(text_lengths)

        print("\nText length:")
        print(
            f"  Average characters: {average_length:.2f}"
        )
        print(
            f"  Shortest:           {shortest}"
        )
        print(
            f"  Longest:            {longest}"
        )

    # --------------------------------------------------------
    # HIERARCHY
    # --------------------------------------------------------

    print("\nHierarchy:")

    print(
        f"  Paragraphs:       {sum(paragraph_counts.values())}"
    )

    print(
        f"  Subparagraphs:    {sum(subparagraph_counts.values())}"
    )

    print(
        f"  Clauses:          {sum(clause_counts.values())}"
    )

    print(
        f"  Subclauses:       {sum(subclause_counts.values())}"
    )

    # --------------------------------------------------------
    # MOST COMMON PARAGRAPH MARKERS
    # --------------------------------------------------------

    print("\nMost common paragraph markers:")

    for marker, count in paragraph_counts.most_common(10):

        print(
            f"  {marker}: {count}"
        )

    # --------------------------------------------------------
    # MOST COMMON SUBPARAGRAPH MARKERS
    # --------------------------------------------------------

    print("\nMost common subparagraph markers:")

    for marker, count in subparagraph_counts.most_common(10):

        print(
            f"  {marker}: {count}"
        )

    # --------------------------------------------------------
    # MOST COMMON CLAUSE MARKERS
    # --------------------------------------------------------

    print("\nMost common clause markers:")

    for marker, count in clause_counts.most_common(10):

        print(
            f"  {marker}: {count}"
        )

    # --------------------------------------------------------
    # MOST COMMON SECTIONS
    # --------------------------------------------------------

    print("\nSections with most documents:")

    for section, count in section_counts.most_common(10):

        print(
            f"  {section}: {count}"
        )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
