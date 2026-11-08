from src.property_law_parser import parse_property_law_corpus
from src.structure_aware_chunker import parse_structure_corpus

def main():
    # 1. Parse both corpora
    print("Parsing baseline corpus...")
    baseline_sections = parse_property_law_corpus()

    print("\nParsing structure-aware corpus...")
    structure_documents = parse_structure_corpus()

    # 2. Display the first 10 structure-aware documents
    print("\n=== STRUCTURE-AWARE DOCUMENT SAMPLE ===")
    print("Total documents:", len(structure_documents))

    for i, doc in enumerate(structure_documents[:10], start=1):
        print(f"\n--- Document {i} ---")
        print("Metadata:", doc.metadata)
        print("Content:", doc.page_content)

    # 3. Compare Act-section coverage
    baseline_keys = {
        (section["act"], section["section"])
        for section in baseline_sections
    }

    structure_keys = {
        (doc.metadata.get("act"), doc.metadata.get("section"))
        for doc in structure_documents
    }

    print("\n=== SECTION COVERAGE COMPARISON ===")
    print("Baseline Act-section combinations:", len(baseline_keys))
    print("Structure-aware Act-section combinations:", len(structure_keys))

    missing = sorted(baseline_keys - structure_keys)
    extra = sorted(structure_keys - baseline_keys)

    print("\nMissing from structure-aware output:", len(missing))
    for item in missing:
        print(item)

    print("\nOnly in structure-aware output:", len(extra))
    for item in extra:
        print(item)

    # 4. Inspect nested paragraph documents
    nested_documents = [
        doc for doc in structure_documents
        if doc.metadata.get("nested_paragraph")
    ]

    print("\n=== NESTED PARAGRAPH CHECK ===")
    print("Nested paragraph documents:", len(nested_documents))

    for doc in nested_documents[:10]:
        print("\nAct:", doc.metadata.get("act"))
        print("Section:", doc.metadata.get("section"))
        print("Subsection:", doc.metadata.get("subsection"))
        print("Paragraph:", doc.metadata.get("paragraph"))
        print("Nested paragraph:", doc.metadata.get("nested_paragraph"))
        print("Hierarchy:", doc.metadata.get("hierarchy"))
        print("Content:", doc.page_content)


if __name__ == "__main__":
    main()
