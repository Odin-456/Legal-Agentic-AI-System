
from pathlib import Path
import xml.etree.ElementTree as ET

from src.property_law_parser import parse_property_law_corpus
from src.structure_aware_chunker import parse_structure_corpus


# --------------------------------------------------
# Configuration
# --------------------------------------------------

XML_DIR = Path("data/raw/australian_property_law")


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def clean_text(text):
    if not text:
        return ""
    return " ".join(text.split())


def get_element_text(element):
    if element is None:
        return ""
    return clean_text("".join(element.itertext()))


def get_section_statuses():
    """
    Return the status of each Act-section combination
    using the original XML legislation files.
    """
    statuses = {}

    if not XML_DIR.exists():
        print(f"WARNING: XML directory not found: {XML_DIR}")
        return statuses

    for xml_file in sorted(XML_DIR.glob("*.xml")):
        try:
            root = ET.parse(xml_file).getroot()
        except ET.ParseError as error:
            print(f"WARNING: Could not parse {xml_file.name}: {error}")
            continue

        title = get_element_text(root.find(".//title"))

        if not title:
            title = xml_file.stem.replace("_", " ").title()

        for section in root.findall(".//level[@type='clause']"):
            section_id = section.get("id", "")

            if not section_id.startswith("sec."):
                continue

            section_number = (
                section_id.replace("sec.", "").split(".")[0]
            )

            key = (title, section_number)
            status = section.get("status", "not specified")

            # Preserve the first occurrence if the same key
            # appears more than once in the XML.
            statuses.setdefault(key, status)

    return statuses


def print_section_list(title, sections):
    print(f"\n{title}: {len(sections)}")

    if not sections:
        print("None")
        return

    for act, section in sections:
        status = section_statuses.get(
            (act, section), "status not found"
        )
        print(f"- {act} | Section {section} | status={status}")


# --------------------------------------------------
# Main comparison
# --------------------------------------------------

def main():
    global section_statuses

    print("=" * 65)
    print("BASELINE VS STRUCTURE-AWARE CHUNKING COMPARISON")
    print("=" * 65)

    # 1. Run both existing pipelines.
    print("\n[1/5] Parsing baseline legislation...")
    baseline_sections = parse_property_law_corpus()

    print("\n[2/5] Parsing structure-aware legislation...")
    structure_documents = parse_structure_corpus()

    # 2. Reproduce the baseline create_chunks logic:
    # one Document per section with non-empty text.
    baseline_nonempty_sections = [
        section
        for section in baseline_sections
        if section.get("text", "").strip()
    ]

    baseline_empty_sections = [
        section
        for section in baseline_sections
        if not section.get("text", "").strip()
    ]

    baseline_chunk_count = len(baseline_nonempty_sections)
    structure_chunk_count = len(structure_documents)

    print("\n[3/5] Comparing document and chunk counts...")
    print("-" * 65)
    print(f"Baseline sections extracted: {len(baseline_sections)}")
    print(f"Baseline sections with text: {baseline_chunk_count}")
    print(f"Baseline empty sections skipped: {len(baseline_empty_sections)}")
    print(f"Baseline final chunks: {baseline_chunk_count}")
    print(f"Structure-aware final chunks: {structure_chunk_count}")
    print(
        "Difference in chunk count "
        "(structure-aware minus baseline): "
        f"{structure_chunk_count - baseline_chunk_count}"
    )

    # 3. Compare Act-section coverage using non-empty
    # baseline sections, not raw section counts.
    baseline_keys = {
        (section["act"], section["section"])
        for section in baseline_nonempty_sections
    }

    structure_keys = {
        (
            doc.metadata.get("act"),
            str(doc.metadata.get("section", "")),
        )
        for doc in structure_documents
    }

    print("\n[4/5] Comparing Act-section coverage...")
    print("-" * 65)
    print(f"Baseline non-empty Act-section combinations: {len(baseline_keys)}")
    print(f"Structure-aware Act-section combinations: {len(structure_keys)}")

    missing = sorted(baseline_keys - structure_keys)
    extra = sorted(structure_keys - baseline_keys)

    print(f"\nMissing from structure-aware output: {len(missing)}")
    print(f"Only in structure-aware output: {len(extra)}")

    # 4. Identify whether missing sections are marked repealed
    # in the original legislation XML.
    print("\n[5/5] Checking missing sections against XML status...")
    section_statuses = get_section_statuses()

    print_section_list(
        "Missing sections and their XML status",
        missing,
    )

    print_section_list(
        "Sections appearing only in structure-aware output",
        extra,
    )

    # Summarise the missing-section statuses.
    repealed_missing = [
        key for key in missing
        if section_statuses.get(key, "").lower() == "repealed"
    ]

    other_missing = [
        key for key in missing
        if section_statuses.get(key, "").lower() != "repealed"
    ]

    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)
    print(f"Baseline final chunks: {baseline_chunk_count}")
    print(f"Structure-aware final chunks: {structure_chunk_count}")
    print(f"Missing non-empty baseline sections: {len(missing)}")
    print(f"Missing sections marked repealed: {len(repealed_missing)}")
    print(f"Other missing sections to investigate: {len(other_missing)}")

    nested_documents = [
        doc for doc in structure_documents
        if doc.metadata.get("nested_paragraph")
    ]

    print(
        "Structure-aware nested-paragraph chunks: "
        f"{len(nested_documents)}"
    )

    if other_missing:
        print(
            "\nACTION REQUIRED: Inspect the missing sections above. "
            "Some may contain substantive text that the structure-aware "
            "parser should preserve."
        )
    elif missing:
        print(
            "\nThe remaining missing sections are marked repealed "
            "in the XML. Check their source content before deciding "
            "whether excluding them is appropriate."
        )
    else:
        print(
            "\nAll non-empty baseline Act-section combinations "
            "are represented in the structure-aware output."
        )

    print("\nComparison complete.")


if __name__ == "__main__":
    main()
