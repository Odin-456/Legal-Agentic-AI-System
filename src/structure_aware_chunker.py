from pathlib import Path
import re
import xml.etree.ElementTree as ET

from langchain_core.documents import Document


# ============================================================
# CONFIGURATION
# ============================================================

XML_DIR = Path("data/raw/australian_property_law")
OUTPUT_DIR = Path("data/processed/structure_chunks")


# ============================================================
# BASIC TEXT FUNCTIONS
# ============================================================

def clean_text(text):
    if not text:
        return ""

    return re.sub(r"\s+", " ", text).strip()


def get_text(element):
    if element is None:
        return ""

    return clean_text("".join(element.itertext()))


def get_number(element):
    if element is None:
        return ""

    number = element.find("./no")

    if number is None:
        number = element.find("./head/no")

    if number is not None:
        return get_text(number)

    return ""


def get_block_text(block):
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
    section_id = level.get("id", "")

    match = re.match(r"sec\.([^.]+)", section_id)

    if match:
        return match.group(1)

    return ""


def get_section_heading(level):
    heading = level.find("./head/heading")

    if heading is not None:
        return get_text(heading)

    return ""


# ============================================================
# CREATE DOCUMENT
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
    if not text.strip():
        return None

    parts = [
        f"Act: {act}",
        f"Section: {section}",
        f"Heading: {heading}"
    ]

    if subsection:
        parts.append(f"Subsection: {subsection}")

    if paragraph:
        parts.append(f"Paragraph: {paragraph}")

    if nested_paragraph:
        parts.append(
            f"Nested paragraph: {nested_paragraph}"
        )

    page_content = (
        "\n".join(parts)
        + "\n\n"
        + text
    )

    return Document(
        page_content=page_content,
        metadata={
            "act": act,
            "section": section,
            "heading": heading,
            "subsection": subsection,
            "paragraph": paragraph,
            "nested_paragraph": nested_paragraph,
            "hierarchy": hierarchy,
            "source_type": "NSW Property Law Legislation",
        }
    )


# ============================================================
# PARSE PARAGRAPHS
# ============================================================

def parse_list_items(
    list_element,
    act,
    section,
    heading,
    subsection,
    parent_paragraph=""
):
    documents = []

    for item in list_element.findall("./li"):

        number = get_number(item)

        # Handle repealed provisions
        if item.get("status") == "repealed":

            paragraph = (
                parent_paragraph
                if parent_paragraph
                else number
            )

            nested = number if parent_paragraph else ""

            hierarchy = (
                "section > subsection > "
                "paragraph > nested_paragraph"
                if parent_paragraph and subsection
                else "section > paragraph > "
                "nested_paragraph"
                if parent_paragraph
                else "section > subsection > paragraph"
                if subsection
                else "section > paragraph"
            )

            document = create_document(
                act,
                section,
                heading,
                "[Repealed]",
                subsection,
                paragraph,
                nested,
                hierarchy
            )

            if document:
                documents.append(document)

            continue

        # Determine paragraph level
        if parent_paragraph:
            paragraph = parent_paragraph
            nested = number
        else:
            paragraph = number
            nested = ""

        # Read only direct block text
        texts = []

        for block in item.findall("./block"):
            text = get_block_text(block)

            if text:
                texts.append(text)

        text = " ".join(texts).strip()

        if text:

            if parent_paragraph and subsection:
                hierarchy = (
                    "section > subsection > "
                    "paragraph > nested_paragraph"
                )
            elif parent_paragraph:
                hierarchy = (
                    "section > paragraph > "
                    "nested_paragraph"
                )
            elif subsection:
                hierarchy = (
                    "section > subsection > paragraph"
                )
            else:
                hierarchy = "section > paragraph"

            document = create_document(
                act,
                section,
                heading,
                text,
                subsection,
                paragraph,
                nested,
                hierarchy
            )

            if document:
                documents.append(document)

        # Find nested lists
        nested_lists = []

        for block in item.findall("./block"):
            nested_lists.extend(
                block.findall("./list")
            )

        nested_lists.extend(
            item.findall("./list")
        )

        # Parse nested paragraphs
        for nested_list in nested_lists:

            documents.extend(
                parse_list_items(
                    nested_list,
                    act,
                    section,
                    heading,
                    subsection,
                    paragraph
                )
            )

    return documents


# ============================================================
# PARSE SUBSECTION
# ============================================================

def parse_subsection(
    subsection,
    act,
    section,
    heading
):
    documents = []

    subsection_number = get_number(subsection)

    # Subsection text
    texts = []

    for block in subsection.findall("./block"):
        text = get_block_text(block)

        if text:
            texts.append(text)

    text = " ".join(texts).strip()

    if text:

        document = create_document(
            act,
            section,
            heading,
            text,
            subsection=subsection_number,
            hierarchy="section > subsection"
        )

        if document:
            documents.append(document)

    # Paragraph lists
    lists = []

    for block in subsection.findall("./block"):
        lists.extend(
            block.findall("./list")
        )

    for list_element in lists:

        documents.extend(
            parse_list_items(
                list_element,
                act,
                section,
                heading,
                subsection_number
            )
        )

    return documents


# ============================================================
# PARSE SECTION
# ============================================================

def parse_section(level, act):

    section = get_section_number(level)

    if not section:
        return []

    heading = get_section_heading(level)

    documents = []

    # Find subsections
    subsections = level.findall(
        "./tier[@type='subclause']"
    )

    if subsections:

        for subsection in subsections:

            documents.extend(
                parse_subsection(
                    subsection,
                    act,
                    section,
                    heading
                )
            )

    else:

        block = level.find("./block")

        if block is not None:

            text = get_block_text(block)

            if text:

                document = create_document(
                    act,
                    section,
                    heading,
                    text,
                    hierarchy="section"
                )

                if document:
                    documents.append(document)

            # Lists directly in the section
            for list_element in block.findall("./list"):

                documents.extend(
                    parse_list_items(
                        list_element,
                        act,
                        section,
                        heading,
                        ""
                    )
                )

    return documents


# ============================================================
# PARSE ONE ACT
# ============================================================

def parse_act(xml_file):

    print(f"Parsing: {xml_file.name}")

    root = ET.parse(xml_file).getroot()

    # Get the real Act title from the XML
    title = root.find(".//title")

    if title is not None:
        act = get_text(title)
    else:
        act = (
            xml_file.stem
            .replace("_", " ")
            .title()
        )

    documents = []
    section_count = 0

    for level in root.findall(
        ".//level[@type='clause']"
    ):

        if not level.get("id", "").startswith("sec."):
            continue

        section_count += 1

        documents.extend(
            parse_section(level, act)
        )

    print(f"Sections found: {section_count}")
    print(f"Structure chunks: {len(documents)}")

    return documents


# ============================================================
# PARSE ALL ACTS
# ============================================================

def parse_structure_corpus():

    documents = []

    xml_files = sorted(
        XML_DIR.glob("*.xml")
    )

    if not xml_files:
        raise FileNotFoundError(
            f"No XML files found in {XML_DIR}"
        )

    for xml_file in xml_files:

        documents.extend(
            parse_act(xml_file)
        )

    print(
        f"Total structure chunks: {len(documents)}"
    )

    return documents


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(documents):

    unique = []
    seen = set()

    for document in documents:

        metadata = document.metadata

        key = (
            metadata.get("act", ""),
            metadata.get("section", ""),
            metadata.get("subsection", ""),
            metadata.get("paragraph", ""),
            metadata.get("nested_paragraph", ""),
            document.page_content
        )

        if key not in seen:

            seen.add(key)
            unique.append(document)

    print(
        f"Unique structure chunks: {len(unique)}"
    )

    return unique


# ============================================================
# SAVE CHUNKS
# ============================================================

def save_chunks(documents):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR / "structure_aware_chunks.txt"
    )

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        for number, document in enumerate(
            documents,
            1
        ):

            file.write(
                f"DOCUMENT {number}\n"
            )

            file.write(
                "=" * 80 + "\n"
            )

            file.write(
                document.page_content
                + "\n\n"
            )

            file.write(
                "METADATA:\n"
            )

            file.write(
                str(document.metadata)
                + "\n\n"
            )

    return output_file


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    documents = parse_structure_corpus()

    print()
    print(
        f"Structure chunks created: {len(documents)}"
    )

    unique_documents = remove_duplicates(documents)

    print(
        f"Duplicate chunks removed: "
        f"{len(documents) - len(unique_documents)}"
    )

    output_file = save_chunks(
        unique_documents
    )

    print(
        f"Saved to: {output_file}"
    )