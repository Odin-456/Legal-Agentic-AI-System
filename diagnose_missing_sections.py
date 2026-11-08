
import re
from pathlib import Path
import xml.etree.ElementTree as ET

from src.property_law_parser import parse_property_law_corpus
from src.structure_aware_chunker import parse_structure_corpus, XML_DIR


def get_text(element):
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def main():
    baseline = parse_property_law_corpus()
    structure_docs = parse_structure_corpus()

    baseline_keys = {
        (item["act"], str(item["section"]))
        for item in baseline
    }

    structure_keys = {
        (
            doc.metadata.get("act"),
            str(doc.metadata.get("section")),
        )
        for doc in structure_docs
    }

    missing = sorted(baseline_keys - structure_keys)

    print(f"\nMissing Act-section combinations: {len(missing)}")

    # Map each XML file to its Act title.
    xml_sections = {}

    for xml_file in sorted(XML_DIR.glob("*.xml")):
        root = ET.parse(xml_file).getroot()
        title_element = root.find(".//title")
        act_name = get_text(title_element)

        if not act_name:
            act_name = xml_file.stem.replace("_", " ").title()

        for level in root.findall(".//level[@type='clause']"):
            section_id = level.get("id", "")

            if not section_id.startswith("sec."):
                continue

            match = re.match(r"sec\.([^.]+)", section_id)
            if not match:
                continue

            section_number = match.group(1)
            xml_sections[(act_name, section_number)] = level

    for act, section in missing:
        print("\n" + "=" * 70)
        print(f"ACT: {act}")
        print(f"SECTION: {section}")

        element = xml_sections.get((act, section))

        if element is None:
            print("XML section element not found using the current matching rules.")
            continue

        print("XML ID:", element.get("id"))
        print("Attributes:", element.attrib)
        print("Direct child tags:", [child.tag for child in element])

        heading = element.find("./head")
        print("Heading:", get_text(heading))

        print("Text preview:", get_text(element)[:700])

        print("Direct blocks:", len(element.findall("./block")))
        print(
            "Direct subsections:",
            len(element.findall("./tier[@type='subclause']")),
        )
        print("Direct lists:", len(element.findall("./list")))

        print("First-level child structure:")
        for child in element:
            print(
                f"  <{child.tag}>",
                "attributes =", child.attrib,
                "text =", get_text(child)[:180],
            )


if __name__ == "__main__":
    main()
