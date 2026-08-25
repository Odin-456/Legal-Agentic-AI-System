import xml.etree.ElementTree as ET
from pathlib import Path
import re


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("data/raw/ecfr/title-7")
OUTPUT_DIR = Path("data/processed/ecfr/title-7")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """Clean and normalize extracted XML text."""

    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# XML TEXT EXTRACTION
# ============================================================

def extract_text(element):
    """Extract all text contained within an XML element."""

    return clean_text(" ".join(element.itertext()))


# ============================================================
# LEGAL HIERARCHY LEVEL
# ============================================================

def detect_level(marker):
    """
    Determine the legal hierarchy level of a marker.

    (a)  -> level 1
    (1)  -> level 2
    (i)  -> level 3
    (A)  -> level 4
    """

    if not marker:
        return 0

    # Paragraph: (a), (b), (c)
    if re.fullmatch(r"\([a-z]\)", marker):
        return 1

    # Subparagraph: (1), (2), (3)
    if re.fullmatch(r"\(\d+\)", marker):
        return 2

    # Clause: (i), (ii), (iii)
    if re.fullmatch(r"\([ivxlcdm]+\)", marker, re.IGNORECASE):
        return 3

    # Subclause: (A), (B), (C)
    if re.fullmatch(r"\([A-Z]\)", marker):
        return 4

    return 0


# ============================================================
# EXTRACT ALL CONSECUTIVE LEGAL MARKERS
# ============================================================

def extract_markers(text):
    """
    Extract all consecutive legal markers from the beginning
    of a paragraph.

    Examples:

    "(a) Something"
        -> ["(a)"]

    "(f)(1) Something"
        -> ["(f)", "(1)"]

    "(f)(1)(i) Something"
        -> ["(f)", "(1)", "(i)"]

    "(f)(1)(i)(A) Something"
        -> ["(f)", "(1)", "(i)", "(A)"]
    """

    if not text:
        return []

    marker_pattern = r"\([a-z]\)|\(\d+\)|\([ivxlcdm]+\)|\([A-Z]\)"

    matches = re.match(
        rf"^((?:{marker_pattern})+)",
        text
    )

    if not matches:
        return []

    marker_string = matches.group(1)

    markers = re.findall(
        marker_pattern,
        marker_string
    )

    return markers


# ============================================================
# REMOVE LEGAL MARKERS FROM TEXT
# ============================================================

def remove_markers(text):
    """
    Remove all legal hierarchy markers from the beginning
    of the paragraph.

    Example:

    "(f)(1)(i) The legal name..."

    becomes:

    "The legal name..."
    """

    if not text:
        return ""

    marker_pattern = r"\([a-z]\)|\(\d+\)|\([ivxlcdm]+\)|\([A-Z]\)"

    cleaned_text = re.sub(
        rf"^(?:{marker_pattern})+\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE
    )

    return cleaned_text.strip()


# ============================================================
# UPDATE LEGAL HIERARCHY
# ============================================================

def update_hierarchy(hierarchy, markers):
    """
    Update the current legal hierarchy using the markers
    found in the current paragraph.

    hierarchy structure:

    [
        paragraph,
        subparagraph,
        clause,
        subclause
    ]

    Example:

    Current:
        ["(a)", "(1)", "(i)", "(A)"]

    New paragraph:
        "(b)"

    Result:
        ["(b)", None, None, None]

    New subparagraph:
        "(2)"

    Result:
        ["(b)", "(2)", None, None]

    New clause:
        "(ii)"

    Result:
        ["(b)", "(2)", "(ii)", None]
    """

    if not markers:
        return hierarchy

    for marker in markers:

        level = detect_level(marker)

        if level == 0:
            continue

        # Convert level 1-4 to Python index 0-3.
        index = level - 1

        # Replace the current level.
        hierarchy[index] = marker

        # Clear everything deeper than the current level.
        for deeper_index in range(index + 1, 4):
            hierarchy[deeper_index] = None

    return hierarchy


# ============================================================
# PARSE ONE eCFR PART
# ============================================================

def parse_part(xml_file):
    """
    Parse one eCFR XML part and preserve its legal hierarchy.
    """

    tree = ET.parse(xml_file)
    root = tree.getroot()

    # --------------------------------------------------------
    # PART INFORMATION
    # --------------------------------------------------------

    part_id = root.attrib.get("N")
    part_type = root.attrib.get("TYPE")

    part_heading = root.find("HEAD")

    part_title = ""

    if part_heading is not None:
        part_title = extract_text(part_heading)

    documents = []

    # --------------------------------------------------------
    # FIND ALL SECTIONS
    # --------------------------------------------------------

    for section in root.iter("DIV8"):

        if section.attrib.get("TYPE") != "SECTION":
            continue

        section_id = section.attrib.get("N")

        section_heading = section.find("HEAD")

        section_title = ""

        if section_heading is not None:
            section_title = extract_text(section_heading)

        citation = f"7 CFR {section_id}"

        # ----------------------------------------------------
        # CURRENT LEGAL HIERARCHY
        #
        # Index 0 = paragraph
        # Index 1 = subparagraph
        # Index 2 = clause
        # Index 3 = subclause
        # ----------------------------------------------------

        hierarchy = [None, None, None, None]

        # ----------------------------------------------------
        # PROCESS PARAGRAPHS
        # ----------------------------------------------------

        for paragraph in section.findall("P"):

            text = extract_text(paragraph)

            if not text:
                continue

            # ------------------------------------------------
            # FIND ALL LEGAL MARKERS
            # ------------------------------------------------

            markers = extract_markers(text)

            # ------------------------------------------------
            # UPDATE HIERARCHY
            # ------------------------------------------------

            hierarchy = update_hierarchy(
                hierarchy,
                markers
            )

            # ------------------------------------------------
            # REMOVE MARKERS FROM TEXT
            # ------------------------------------------------

            clean_paragraph_text = remove_markers(text)

            # ------------------------------------------------
            # CREATE DOCUMENT
            # ------------------------------------------------

            document = {
                "text": clean_paragraph_text,

                "metadata": {

                    "title": "7",

                    "part": part_id,

                    "part_title": part_title,

                    "section": section_id,

                    "section_title": section_title,

                    "paragraph": hierarchy[0],

                    "subparagraph": hierarchy[1],

                    "clause": hierarchy[2],

                    "subclause": hierarchy[3],

                    "citation": citation,

                    "source_file": xml_file.name,
                },
            }

            documents.append(document)

    return documents


# ============================================================
# TEST HIERARCHY FUNCTIONS
# ============================================================

def test_hierarchy_functions():

    print("\n")
    print("=" * 60)
    print("HIERARCHY FUNCTION TEST")
    print("=" * 60)

    test_texts = [
        "(a) This is paragraph a.",
        "(1) This is subparagraph 1.",
        "(i) This is clause i.",
        "(f)(1) Any foreign person.",
        "(f)(1)(i) The legal name.",
        "(A) This is a subclause.",
    ]

    for text in test_texts:

        markers = extract_markers(text)

        levels = [
            detect_level(marker)
            for marker in markers
        ]

        cleaned = remove_markers(text)

        print("\nText:")
        print(text)

        print("Markers:")
        print(markers)

        print("Levels:")
        print(levels)

        print("Cleaned text:")
        print(cleaned)


# ============================================================
# TEST REAL eCFR FILE
# ============================================================

def test_real_ecfr_file(documents):

    print("\n")
    print("=" * 60)
    print("REAL eCFR HIERARCHY TEST")
    print("=" * 60)

    # Look specifically for documents containing
    # the hierarchy around (f) and (g).

    for document in documents:

        text = document["text"]

        metadata = document["metadata"]

        # We removed the markers from text, so use metadata
        # to identify the relevant hierarchy.

        paragraph = metadata["paragraph"]
        subparagraph = metadata["subparagraph"]
        clause = metadata["clause"]

        if paragraph in ["(f)", "(g)"]:

            print("\n" + "-" * 60)

            print("Text:")
            print(text[:300])

            print("\nMetadata:")

            print(
                f"  paragraph:    {paragraph}"
            )

            print(
                f"  subparagraph: {subparagraph}"
            )

            print(
                f"  clause:       {clause}"
            )

            print(
                f"  subclause:    {metadata['subclause']}"
            )

            print(
                f"  citation:     {metadata['citation']}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("eCFR XML BATCH PARSER")
    print("=" * 60)

    # --------------------------------------------------------
    # FIND ALL XML FILES
    # --------------------------------------------------------

    xml_files = sorted(
        INPUT_DIR.glob("*.xml")
    )

    print(f"XML files found: {len(xml_files)}")

    if not xml_files:
        print("ERROR: No XML files found.")
        return

    # --------------------------------------------------------
    # PROCESS ALL XML FILES
    # --------------------------------------------------------

    total_documents = 0
    successful_files = 0
    failed_files = []

    for index, xml_file in enumerate(xml_files, start=1):

        print(
            f"\n[{index}/{len(xml_files)}] "
            f"Processing: {xml_file.name}"
        )

        try:

            documents = parse_part(xml_file)

            # ------------------------------------------------
            # OUTPUT FILE
            # ------------------------------------------------

            output_file = (
                OUTPUT_DIR /
                f"{xml_file.stem}.json"
            )

            # ------------------------------------------------
            # SAVE JSON
            # ------------------------------------------------

            import json

            with open(
                output_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    documents,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            document_count = len(documents)

            total_documents += document_count
            successful_files += 1

            print(
                f"  Documents extracted: "
                f"{document_count}"
            )

            print(
                f"  Saved to: "
                f"{output_file}"
            )

        except Exception as e:

            print(
                f"  ERROR: {e}"
            )

            failed_files.append(
                (
                    xml_file.name,
                    str(e)
                )
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("PARSING COMPLETE")
    print("=" * 60)

    print(
        f"XML files found:       {len(xml_files)}"
    )

    print(
        f"Files processed:       {successful_files}"
    )

    print(
        f"Files failed:          {len(failed_files)}"
    )

    print(
        f"Total documents:       {total_documents}"
    )

    print(
        f"Output directory:      {OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # SHOW FAILED FILES
    # --------------------------------------------------------

    if failed_files:

        print("\nFailed files:")

        for filename, error in failed_files:

            print(
                f"  {filename}: {error}"
            )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()