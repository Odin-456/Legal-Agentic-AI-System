import re
from pathlib import Path
import xml.etree.ElementTree as ET

from langchain_core.documents import Document


XML_DIR = Path(
    "data/raw/australian_property_law"
)

def clean_text(text):
    if not text:
        return "" 

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_text(element):
    if element is None:
        return ""

    text = "".join(element.itertext())

    return clean_text(text)


def get_number(element):
    if element is None:
        return ""

    number = element.find("./no")

    if number is None:
        number = element.find("./head/no")

    if number is not None:
        return get_text(number)

    return ""


def parse_block_direct_text(block):
    """
    Extract direct text from a block.
    Nested lists are handled separately.
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


def get_section_number(level):

    section_id = level.get("id", "")

    match = re.match(
        r"sec\.([^.]+)",
        section_id
    )

    if match:
        return match.group(1)

    return ""


def get_section_heading(level):

    head = level.find("./head")

    if head is None:
        return ""

    heading = head.find("./heading")

    if heading is None:
        return ""

    return get_text(heading)


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

    metadata = {
        "act": act,
        "section": section,
        "heading": heading,
        "subsection": subsection,
        "paragraph": paragraph,
        "nested_paragraph": nested_paragraph,
        "hierarchy": hierarchy,
        "source_type": "NSW Property Law Legislation",
    }

    return Document(
        page_content=page_content,
        metadata=metadata
    )


def parse_list_items(
    list_element,
    act,
    section,
    heading,
    subsection,
    parent_paragraph=""
):

    documents = []

    for li in list_element.findall("./li"):

        paragraph_number = get_number(li)

        # Handle repealed provisions

        if li.get("status") == "repealed":

            if parent_paragraph:

                paragraph = parent_paragraph
                nested_paragraph = paragraph_number

                if subsection:
                    hierarchy = (
                        "section > subsection > "
                        "paragraph > nested_paragraph"
                    )
                else:
                    hierarchy = (
                        "section > paragraph > "
                        "nested_paragraph"
                    )

            else:

                paragraph = paragraph_number
                nested_paragraph = ""

                if subsection:
                    hierarchy = (
                        "section > subsection > paragraph"
                    )
                else:
                    hierarchy = (
                        "section > paragraph"
                    )

            document = create_document(
                act=act,
                section=section,
                heading=heading,
                text="[Repealed]",
                subsection=subsection,
                paragraph=paragraph,
                nested_paragraph=nested_paragraph,
                hierarchy=hierarchy
            )

            if document:
                documents.append(document)

            continue

        # Determine paragraph level

        if parent_paragraph:

            paragraph = parent_paragraph
            nested_paragraph = paragraph_number

        else:

            paragraph = paragraph_number
            nested_paragraph = ""

        # Extract direct text

        direct_texts = []

        for block in li.findall("./block"):

            text = parse_block_direct_text(block)

            if text:
                direct_texts.append(text)

        text = " ".join(direct_texts).strip()

        # Create document

        if text:

            if parent_paragraph:

                if subsection:
                    hierarchy = (
                        "section > subsection > "
                        "paragraph > nested_paragraph"
                    )
                else:
                    hierarchy = (
                        "section > paragraph > "
                        "nested_paragraph"
                    )

            else:

                if subsection:
                    hierarchy = (
                        "section > subsection > paragraph"
                    )
                else:
                    hierarchy = (
                        "section > paragraph"
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
                documents.append(document)

        # Find nested lists

        nested_lists = []

        for block in li.findall("./block"):
            nested_lists.extend(
                block.findall("./list")
            )

        nested_lists.extend(
            li.findall("./list")
        )

        # Recursively parse nested paragraphs

        for nested_list in nested_lists:

            nested_documents = parse_list_items(
                nested_list,
                act=act,
                section=section,
                heading=heading,
                subsection=subsection,
                parent_paragraph=paragraph
            )

            documents.extend(
                nested_documents
            )

    return documents


def parse_subsection(
    tier,
    act,
    section,
    heading
):

    documents = []

    subsection_number = get_number(tier)

    # Extract direct subsection text

    direct_texts = []

    for block in tier.findall("./block"):

        text = parse_block_direct_text(block)

        if text:
            direct_texts.append(text)

    direct_text = " ".join(
        direct_texts
    ).strip()

    # Create subsection document

    if direct_text:

        document = create_document(
            act=act,
            section=section,
            heading=heading,
            text=direct_text,
            subsection=subsection_number,
            hierarchy="section > subsection"
        )

        if document:
            documents.append(document)

    # Find lists inside subsection

    lists = []

    for block in tier.findall("./block"):

        lists.extend(
            block.findall("./list")
        )

    # Parse paragraphs

    for list_element in lists:

        paragraph_documents = parse_list_items(
            list_element,
            act=act,
            section=section,
            heading=heading,
            subsection=subsection_number
        )

        documents.extend(
            paragraph_documents
        )

    return documents


def parse_section(level, act):

    section = get_section_number(level)

    if not section:
        return []

    heading = get_section_heading(level)

    documents = []

    subsections = level.findall(
        "./tier[@type='subclause']"
    )

    if subsections:

        for subsection in subsections:

            subsection_documents = parse_subsection(
                subsection,
                act=act,
                section=section,
                heading=heading
            )

            documents.extend(
                subsection_documents
            )

    else:

        block = level.find("./block")

        if block is not None:

            # Extract direct section text

            direct_text = parse_block_direct_text(
                block
            )

            if direct_text:

                document = create_document(
                    act=act,
                    section=section,
                    heading=heading,
                    text=direct_text,
                    hierarchy="section"
                )

                if document:
                    documents.append(document)

            # Handle lists directly inside section

            lists = block.findall("./list")

            for list_element in lists:

                paragraph_documents = parse_list_items(
                    list_element,
                    act=act,
                    section=section,
                    heading=heading,
                    subsection=""
                )

                documents.extend(
                    paragraph_documents
                )

    return documents


def parse_act(xml_file):

    print(
        f"Parsing: {xml_file.name}"
    )

    tree = ET.parse(xml_file)

    root = tree.getroot()

    # Get Act name

    title = root.find(".//title")

    if title is not None:
        act_name = get_text(title)
    else:
        act_name = ""

    if not act_name:

        act_name = (
            xml_file.stem
            .replace("_", " ")
            .title()
        )

    documents = []

    section_count = 0

    for level in root.findall(
        ".//level[@type='clause']"
    ):

        section_id = level.get(
            "id",
            ""
        )

        if not section_id.startswith("sec."):
            continue

        section_count += 1

        section_documents = parse_section(
            level,
            act_name
        )

        documents.extend(
            section_documents
        )

    print(
        f"  Sections found: {section_count}"
    )

    return documents


def parse_structure_corpus():

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

        if key in seen:
            continue

        seen.add(key)

        unique.append(document)

    return unique
