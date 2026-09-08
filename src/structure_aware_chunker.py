from pathlib import Path
import re
import xml.etree.ElementTree as ET

from langchain_core.documents import Document


# ============================================================
# CONFIGURATION
# ============================================================

XML_DIR = Path(
    "data/raw/australian_property_law"
)

OUTPUT_DIR = Path(
    "data/processed/structure_chunks"
)


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(text):
    """
    Clean whitespace from extracted XML text.
    """

    if not text:
        return ""

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def get_text(element):
    """
    Extract all text from an XML element.
    """

    if element is None:
        return ""

    return clean_text(
        "".join(element.itertext())
    )


def get_number(element):
    """
    Extract numbering from an XML element.

    Handles structures such as:

        <no>(1)</no>
        <head><no>(1)</no></head>
        <no>(a)</no>
        <no>(i)</no>
    """

    if element is None:
        return ""

    # Direct number
    no = element.find("./no")

    if no is not None:
        return get_text(no)

    # Number inside head
    no = element.find("./head/no")

    if no is not None:
        return get_text(no)

    return ""


# ============================================================
# XML TEXT EXTRACTION
# ============================================================

def parse_block_direct_text(block):
    """
    Extract only direct <txt> elements from a block.

    Nested lists are deliberately excluded because they
    are handled separately.
    """

    if block is None:
        return ""

    texts = []

    for child in block:

        if child.tag == "txt":

            text = get_text(child)

            if text:
                texts.append(text)

    return " ".join(texts).strip()


# ============================================================
# SECTION INFORMATION
# ============================================================

def get_section_number(level):
    """
    Extract section number from an XML level ID.

    Example:

        sec.42
        sec.45C
        sec.100

    becomes:

        42
        45C
        100
    """

    section_id = level.get(
        "id",
        ""
    )

    match = re.match(
        r"sec\.([^.]+)",
        section_id
    )

    if match:
        return match.group(1)

    return ""


def get_section_heading(level):
    """
    Extract the section heading.
    """

    head = level.find("./head")

    if head is None:
        return ""

    heading = head.find(
        "./heading"
    )

    if heading is None:
        return ""

    return get_text(heading)


# ============================================================
# CREATE STRUCTURED DOCUMENT
# ============================================================

def create_document(
    act,
    section,
    heading,
    text,
    subsection="",
    paragraph="",
    nested_paragraph="",
    hierarchy=""
):
    """
    Create one LangChain Document containing both the
    legal text and its structural metadata.
    """

    if not text.strip():
        return None

    # --------------------------------------------------------
    # Build structural prefix
    # --------------------------------------------------------

    parts = [
        f"Act: {act}",
        f"Section: {section}",
        f"Heading: {heading}",
    ]

    if subsection:
        parts.append(
            f"Subsection: {subsection}"
        )

    if paragraph:
        parts.append(
            f"Paragraph: {paragraph}"
        )

    if nested_paragraph:
        parts.append(
            f"Nested paragraph: {nested_paragraph}"
        )

    prefix = "\n".join(parts)

    page_content = (
        prefix
        + "\n\n"
        + text
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "act": act,
        "section": section,
        "heading": heading,
        "subsection": subsection,
        "paragraph": paragraph,
        "nested_paragraph": nested_paragraph,
        "hierarchy": hierarchy,
        "source_type": (
            "NSW Property Law Legislation"
        ),
    }

    return Document(
        page_content=page_content,
        metadata=metadata
    )


# ============================================================
# PARAGRAPH / LIST PARSING
# ============================================================

def parse_list_items(
    list_element,
    act,
    section,
    heading,
    subsection,
    parent_paragraph=""
):
    """
    Parse paragraph and nested paragraph structures.

    Generic hierarchy:

        subsection
            paragraph
                nested paragraph

    The function does not assume specific numbering.
    """

    documents = []

    for li in list_element.findall("./li"):

        paragraph_number = get_number(li)

        # ----------------------------------------------------
        # Repealed provision
        # ----------------------------------------------------

        if li.get("status") == "repealed":

            text = "[Repealed]"

            hierarchy = (
                "section > subsection > paragraph"
            )

            if parent_paragraph:
                hierarchy = (
                    "section > subsection > "
                    "paragraph > nested_paragraph"
                )

            document = create_document(
                act=act,
                section=section,
                heading=heading,
                text=text,
                subsection=subsection,
                paragraph=(
                    parent_paragraph
                    if parent_paragraph
                    else paragraph_number
                ),
                nested_paragraph=(
                    paragraph_number
                    if parent_paragraph
                    else ""
                ),
                hierarchy=hierarchy
            )

            if document:
                documents.append(
                    document
                )

            continue

        # ----------------------------------------------------
        # Determine whether this is a normal paragraph
        # or a nested paragraph.
        # ----------------------------------------------------

        if parent_paragraph:

            paragraph = parent_paragraph

            nested_paragraph = (
                paragraph_number
            )

        else:

            paragraph = paragraph_number

            nested_paragraph = ""

        # ----------------------------------------------------
        # Extract direct block text
        # ----------------------------------------------------

        direct_texts = []

        for block in li.findall(
            "./block"
        ):

            text = parse_block_direct_text(
                block
            )

            if text:
                direct_texts.append(
                    text
                )

        text = " ".join(
            direct_texts
        ).strip()

        # ----------------------------------------------------
        # Create document for this paragraph
        # ----------------------------------------------------

        if text:

            if parent_paragraph:

                hierarchy = (
                    "section > subsection > "
                    "paragraph > nested_paragraph"
                )

            else:

                hierarchy = (
                    "section > subsection > paragraph"
                )

            document = create_document(
                act=act,
                section=section,
                heading=heading,
                text=text,
                subsection=subsection,
                paragraph=paragraph,
                nested_paragraph=nested_paragraph,
                hierarchy=hierarchy
            )

            if document:
                documents.append(
                    document
                )

        # ----------------------------------------------------
        # Find nested lists
        # ----------------------------------------------------

        nested_lists = []

        for block in li.findall(
            "./block"
        ):

            nested_lists.extend(
                block.findall("./list")
            )

        nested_lists.extend(
            li.findall("./list")
        )

        # ----------------------------------------------------
        # Recursively parse nested paragraphs
        # ----------------------------------------------------

        for nested_list in nested_lists:

            nested_documents = (
                parse_list_items(
                    nested_list,
                    act=act,
                    section=section,
                    heading=heading,
                    subsection=subsection,
                    parent_paragraph=(
                        paragraph
                    )
                )
            )

            documents.extend(
                nested_documents
            )

    return documents


# ============================================================
# SUBSECTION PARSING
# ============================================================

def parse_subsection(
    tier,
    act,
    section,
    heading
):
    """
    Parse one subsection.

    Example:

        (1) Some legal rule
            (a) First condition
            (b) Second condition
                (i) Nested condition
                (ii) Nested condition

    Each legal unit becomes its own retrieval document.
    """

    documents = []

    subsection_number = get_number(
        tier
    )

    # --------------------------------------------------------
    # Direct subsection text
    # --------------------------------------------------------

    direct_texts = []

    for block in tier.findall(
        "./block"
    ):

        text = parse_block_direct_text(
            block
        )

        if text:
            direct_texts.append(
                text
            )

    direct_text = " ".join(
        direct_texts
    ).strip()

    # --------------------------------------------------------
    # Create subsection-level document
    # --------------------------------------------------------

    if direct_text:

        document = create_document(
            act=act,
            section=section,
            heading=heading,
            text=direct_text,
            subsection=subsection_number,
            hierarchy=(
                "section > subsection"
            )
        )

        if document:
            documents.append(
                document
            )

    # --------------------------------------------------------
    # Find lists directly inside subsection blocks
    # --------------------------------------------------------

    lists = []

    for block in tier.findall(
        "./block"
    ):

        lists.extend(
            block.findall("./list")
        )

    # --------------------------------------------------------
    # Parse paragraphs
    # --------------------------------------------------------

    for list_element in lists:

        paragraph_documents = (
            parse_list_items(
                list_element,
                act=act,
                section=section,
                heading=heading,
                subsection=subsection_number
            )
        )

        documents.extend(
            paragraph_documents
        )

    return documents


# ============================================================
# SECTION PARSING
# ============================================================

def parse_section(
    level,
    act
):
    """
    Parse one complete legislation section.

    The section itself is not flattened into one document.

    Instead, its legal hierarchy is preserved.
    """

    section = get_section_number(
        level
    )

    if not section:
        return []

    heading = get_section_heading(
        level
    )

    documents = []

    # --------------------------------------------------------
    # Find subsections
    # --------------------------------------------------------

    subsections = level.findall(
        "./tier[@type='subclause']"
    )

    for subsection in subsections:

        subsection_documents = (
            parse_subsection(
                subsection,
                act=act,
                section=section,
                heading=heading
            )
        )

        documents.extend(
            subsection_documents
        )

    return documents


# ============================================================
# ACT PARSING
# ============================================================

def parse_act(
    xml_file
):
    """
    Parse one NSW property-law Act.
    """

    print(
        f"Parsing: {xml_file.name}"
    )

    tree = ET.parse(
        xml_file
    )

    root = tree.getroot()

    # --------------------------------------------------------
    # Act title
    # --------------------------------------------------------

    act_name = ""

    title = root.find(
        ".//title"
    )

    if title is not None:
        act_name = get_text(
            title
        )

    if not act_name:

        act_name = (
            xml_file.stem
            .replace("_", " ")
            .title()
        )

    # --------------------------------------------------------
    # Find sections
    # --------------------------------------------------------

    documents = []

    section_count = 0

    for level in root.findall(
        ".//level[@type='clause']"
    ):

        section_id = level.get(
            "id",
            ""
        )

        if not section_id.startswith(
            "sec."
        ):
            continue

        section_count += 1

        section_documents = (
            parse_section(
                level,
                act_name
            )
        )

        documents.extend(
            section_documents
        )

    print(
        f"  Sections found: "
        f"{section_count}"
    )

    print(
        f"  Structure chunks: "
        f"{len(documents)}"
    )

    return documents


# ============================================================
# CORPUS PARSING
# ============================================================

def parse_structure_corpus():
    """
    Parse all NSW property-law XML files.
    """

    all_documents = []

    xml_files = sorted(
        XML_DIR.glob("*.xml")
    )

    if not xml_files:

        raise FileNotFoundError(
            f"No XML files found in {XML_DIR}"
        )

    for xml_file in xml_files:

        documents = parse_act(
            xml_file
        )

        all_documents.extend(
            documents
        )

    return all_documents


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(
    documents
):
    """
    Remove exact duplicate structural documents.
    """

    unique = []

    seen = set()

    for document in documents:

        key = (
            document.metadata.get(
                "act",
                ""
            ),
            document.metadata.get(
                "section",
                ""
            ),
            document.metadata.get(
                "subsection",
                ""
            ),
            document.metadata.get(
                "paragraph",
                ""
            ),
            document.metadata.get(
                "nested_paragraph",
                ""
            ),
            document.page_content
        )

        if key in seen:
            continue

        seen.add(key)

        unique.append(
            document
        )

    return unique


# ============================================================
# SAVE FOR INSPECTION
# ============================================================

def save_chunks(
    documents
):
    """
    Save all structure-aware chunks to a text file.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / "structure_aware_chunks.txt"
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


# ============================================================
# DISPLAY SECTION 42
# ============================================================

def show_section_42(
    documents
):
    """
    Display all structure-aware chunks belonging
    to Section 42.

    This is only a test helper.
    The retrieval system itself does NOT depend
    on Section 42.
    """

    print()
    print(
        "=" * 80
    )

    print(
        "REAL PROPERTY ACT 1900 — SECTION 42"
    )

    print(
        "=" * 80
    )

    found = False

    for document in documents:

        metadata = document.metadata

        if (
            metadata.get("act")
            == "Real Property Act 1900"
            and metadata.get("section")
            == "42"
        ):

            found = True

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

            print(
                "-" * 80
            )

    if not found:

        print(
            "Section 42 not found."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "=" * 80
    )

    print(
        "STRUCTURE-AWARE PROPERTY LAW CHUNKING"
    )

    print(
        "=" * 80
    )

    print()

    documents = (
        parse_structure_corpus()
    )

    print()
    print(
        "=" * 80
    )

    print(
        f"Structure chunks created: "
        f"{len(documents)}"
    )

    print(
        "=" * 80
    )

    unique_documents = (
        remove_duplicates(
            documents
        )
    )

    print()
    print(
        f"Unique structure chunks: "
        f"{len(unique_documents)}"
    )

    print(
        f"Duplicate chunks removed: "
        f"{len(documents) - len(unique_documents)}"
    )

    output_file = save_chunks(
        unique_documents
    )

    print()
    print(
        f"Saved to: {output_file}"
    )

    # --------------------------------------------------------
    # Test Section 42
    # --------------------------------------------------------

    show_section_42(
        unique_documents
    )