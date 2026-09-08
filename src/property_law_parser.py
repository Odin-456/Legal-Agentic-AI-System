from pathlib import Path
import re
import xml.etree.ElementTree as ET

from langchain_core.documents import Document


XML_DIR = Path("data/raw/australian_property_law")


def clean_text(text):
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_text(element):
    if element is None:
        return ""

    return clean_text("".join(element.itertext()))


def get_number(element):
    """
    Get the numbering from an element.

    NSW XML stores subsection numbers like (1), (2), (3)
    inside:

        <head>
            <no>(1)</no>
        </head>
    """

    if element is None:
        return ""

    # First check directly under the element.
    no = element.find("./no")

    if no is not None:
        return get_text(no)

    # Then check inside <head>.
    no = element.find("./head/no")

    if no is not None:
        return get_text(no)

    return ""


def parse_txt(txt):
    return get_text(txt)


def parse_block_text(block):
    """
    Extract only direct <txt> content from a block.

    This prevents nested lists from being flattened into
    the paragraph text.
    """

    texts = []

    for child in block:
        if child.tag == "txt":
            text = parse_txt(child)

            if text:
                texts.append(text)

    return " ".join(texts).strip()


def parse_list(list_element, indent_level):
    """
    Parse a list while preserving numbering and indentation.
    """

    lines = []

    for li in list_element.findall("./li"):
        lines.extend(
            parse_list_item(
                li,
                indent_level
            )
        )

    return lines


def parse_list_item(li, indent_level):
    """
    Parse one list item.

    Example:

        (a) Main paragraph

            (i) Nested paragraph
    """

    lines = []

    number = get_number(li)

    indent = "    " * indent_level

    # Handle explicitly repealed items.
    if li.get("status") == "repealed":

        if number:
            lines.append(
                f"{indent}{number} [Repealed]"
            )

        return lines

    blocks = li.findall("./block")

    first_block = True

    for block in blocks:

        text = parse_block_text(block)

        if text:

            if first_block and number:
                lines.append(
                    f"{indent}{number} {text}"
                )
            else:
                lines.append(
                    f"{indent}{text}"
                )

            first_block = False

        # Handle lists nested inside a block.
        nested_lists = block.findall("./list")

        for nested_list in nested_lists:
            lines.extend(
                parse_list(
                    nested_list,
                    indent_level + 1
                )
            )

    # Handle lists directly inside the list item.
    nested_lists = li.findall("./list")

    for nested_list in nested_lists:
        lines.extend(
            parse_list(
                nested_list,
                indent_level + 1
            )
        )

    # If there was no block but the item has a number.
    if not blocks and number:
        lines.append(
            f"{indent}{number}"
        )

    return lines


def parse_subsection(tier):
    """
    Parse a subsection.

    NSW XML structure:

        <tier type="subclause">
            <head>
                <no>(1)</no>
            </head>
            <block>
                <txt>...</txt>
                <list>...</list>
            </block>
        </tier>

    The important part is that the subsection number is
    located at ./head/no.
    """

    lines = []

    # Correctly gets (1), (2), (3), etc.
    subsection_number = get_number(tier)

    blocks = tier.findall("./block")

    first_text_added = False

    for block in blocks:

        text = parse_block_text(block)

        if text:

            if not first_text_added:

                if subsection_number:
                    lines.append(
                        f"{subsection_number} {text}"
                    )
                else:
                    lines.append(text)

                first_text_added = True

            else:
                lines.append(
                    f"    {text}"
                )

        # Lists contained inside the block.
        nested_lists = block.findall("./list")

        for nested_list in nested_lists:
            lines.extend(
                parse_list(
                    nested_list,
                    1
                )
            )

    # If the subsection contains no text but still has
    # a subsection number, preserve that number.
    if not first_text_added and subsection_number:
        lines.append(
            subsection_number
        )

    return lines


def parse_section(level, act_name):
    """
    Parse one NSW legislation section.
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
        section_number = match.group(1)
    else:
        section_number = ""

    # Extract heading.
    heading = ""

    head = level.find("./head")

    if head is not None:

        heading_element = head.find(
            "./heading"
        )

        if heading_element is not None:
            heading = get_text(
                heading_element
            )

    # Only legal content goes into page_content.
    #
    # Section number and heading are stored separately
    # in metadata and printed by print_section().
    lines = []

    subsections = level.findall(
        "./tier[@type='subclause']"
    )

    for subsection in subsections:

        subsection_lines = parse_subsection(
            subsection
        )

        lines.extend(
            subsection_lines
        )

    text = "\n".join(lines).strip()

    return {
        "act": act_name,
        "section": section_number,
        "heading": heading,
        "text": text,
        "source_type": "NSW Property Law Legislation",
    }


def parse_act(xml_file):
    """
    Parse one NSW property-law Act.
    """

    print(
        f"Parsing: {xml_file.name}"
    )

    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract Act title.
    act_name = ""

    title = root.find(".//title")

    if title is not None:
        act_name = get_text(title)

    # Fallback if title is unavailable.
    if not act_name:
        act_name = (
            xml_file.stem
            .replace("_", " ")
            .title()
        )

    sections = []

    # NSW legislation uses:
    #
    # <level type="clause" id="sec.42">
    #
    # rather than <section>.
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

        section = parse_section(
            level,
            act_name
        )

        if section["section"]:
            sections.append(section)

    print(
        f"  Sections found: {len(sections)}"
    )

    return sections


def create_langchain_documents(sections):
    """
    Convert parsed sections into LangChain Documents.
    """

    documents = []

    for section in sections:

        document = Document(
            page_content=section["text"],
            metadata={
                "act": section["act"],
                "section": section["section"],
                "heading": section["heading"],
                "source_type": section["source_type"],
            }
        )

        documents.append(document)

    return documents


def parse_property_law_corpus():
    """
    Parse all XML files in the property-law corpus.
    """

    all_sections = []

    xml_files = sorted(
        XML_DIR.glob("*.xml")
    )

    if not xml_files:
        raise FileNotFoundError(
            f"No XML files found in {XML_DIR}"
        )

    for xml_file in xml_files:

        sections = parse_act(
            xml_file
        )

        all_sections.extend(
            sections
        )

    return all_sections


def print_section(
    documents,
    act_name,
    section_number
):
    """
    Print a specific section for testing.
    """

    for document in documents:

        if (
            document["act"] == act_name
            and document["section"] == section_number
        ):

            print()
            print(
                "=" * 80
            )

            print(
                f"{act_name} — Section {section_number}"
            )

            print(
                "=" * 80
            )

            print(
                f"Act: {document['act']}"
            )

            print(
                f"Section: {document['section']}"
            )

            print(
                f"Heading: {document['heading']}"
            )

            print()

            # Legal content only.
            print(
                document["text"]
            )

            print()

            return

    print(
        f"Section {section_number} not found in {act_name}"
    )


if __name__ == "__main__":

    documents = parse_property_law_corpus()

    print()
    print(
        "=" * 60
    )

    print(
        f"Documents created: {len(documents)}"
    )

    print(
        "=" * 60
    )

    print_section(
        documents,
        "Real Property Act 1900",
        "42"
    )

    langchain_documents = (
        create_langchain_documents(
            documents
        )
    )

    print(
        f"LangChain documents: "
        f"{len(langchain_documents)}"
    )