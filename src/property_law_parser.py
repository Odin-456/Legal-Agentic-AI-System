from pathlib import Path
import re
import xml.etree.ElementTree as ET

XML_DIR = Path("data/raw/australian_property_law")

# Cleans extra spaces, tabs, and line breaks from text
def clean_text(text):
    if not text:
        return ""

    return re.sub(r"\s+", " ", text).strip()

# Gets all text contained inside an XML element
def get_text(element):
    if element is None:
        return ""

    return clean_text("".join(element.itertext()))

# Gets the number of a subsection or paragraph, such as (1), (a), or (i)
def get_number(element):
    if element is None:
        return ""

    number = element.find("./no")

    if number is None:
        number = element.find("./head/no")

    return get_text(number)

# Gets only the direct text from an XML block
# This prevents nested lists from being mixed into the main text
def get_block_text(block):
    texts = []

    for child in block:
        if child.tag == "txt":
            text = get_text(child)

            if text:
                texts.append(text)

    return " ".join(texts)


# Parses legal paragraphs and nested paragraphs while keeping their indentation
def parse_list(list_element, indent=1):
    lines = []

    for item in list_element.findall("./li"):
        number = get_number(item)
        spaces = "    " * indent

        # Keep information about repealed paragraphs
        if item.get("status") == "repealed":
            if number:
                lines.append(f"{spaces}{number} [Repealed]")
            continue

        blocks = item.findall("./block")
        first_text = True

        for block in blocks:
            text = get_block_text(block)

            if text:
                if first_text and number:
                    lines.append(f"{spaces}{number} {text}")
                else:
                    lines.append(f"{spaces}{text}")

                first_text = False

            # Parse lists nested inside the block
            for nested in block.findall("./list"):
                lines.extend(parse_list(nested, indent + 1))

        # Parse lists directly inside the item
        for nested in item.findall("./list"):
            lines.extend(parse_list(nested, indent + 1))

        # Keep the paragraph number if there is no text
        if not blocks and number:
            lines.append(f"{spaces}{number}")

    return lines


# Parses a subsection such as (1), (2), or (3)
def parse_subsection(subsection):
    lines = []
    number = get_number(subsection)
    text_added = False

    for block in subsection.findall("./block"):
        text = get_block_text(block)
        if text:
            if not text_added and number:
                lines.append(f"{number} {text}")
            elif not text_added:
                lines.append(text)
            else:
                lines.append(f"    {text}")

            text_added = True
        # Parse paragraphs inside the subsection
        for nested in block.findall("./list"):
            lines.extend(parse_list(nested))

    # Keep the subsection number if it has no text
    if not text_added and number:
        lines.append(number)

    return lines


# Parses one section of an NSW property-law Act
def parse_section(section, act_name):
    section_id = section.get("id", "")

    # NSW XML uses IDs such as "sec.42"
    section_number = section_id.replace("sec.", "").split(".")[0]

    # Get the section heading
    heading = ""
    heading_element = section.find("./head/heading")

    if heading_element is not None:
        heading = get_text(heading_element)
    lines = []

    # Get all subsections in the section
    for subsection in section.findall("./tier[@type='subclause']"):
        lines.extend(parse_subsection(subsection))

    # Store the section information
    return {
        "act": act_name,
        "section": section_number,
        "heading": heading,
        "text": "\n".join(lines).strip(),
        "source_type": "NSW Property Law Legislation",
    }


# Reads one NSW property-law XML file and extracts all its sections
def parse_act(xml_file):
    print(f"Parsing: {xml_file.name}")

    # Read the XML file
    root = ET.parse(xml_file).getroot()

    # Get the Act name
    title = root.find(".//title")
    act_name = get_text(title)

    # Use the filename if no title is found
    if not act_name:
        act_name = xml_file.stem.replace("_", " ").title()

    sections = []

    # Find the sections stored as <level type="clause">
    for section in root.findall(".//level[@type='clause']"):

        # Ignore XML elements that are not sections
        if not section.get("id", "").startswith("sec."):
            continue

        sections.append(parse_section(section, act_name))

    print(f"  Sections found: {len(sections)}")

    return sections

# Reads and parses every XML Act in the NSW property-law corpus
def parse_property_law_corpus():
    xml_files = sorted(XML_DIR.glob("*.xml"))

    if not xml_files:
        raise FileNotFoundError(
            f"No XML files found in {XML_DIR}"
        )

    sections = []

    for xml_file in xml_files:
        sections.extend(parse_act(xml_file))

    return sections