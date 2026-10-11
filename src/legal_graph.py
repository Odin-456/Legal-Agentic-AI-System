import re
import pickle
from pathlib import Path

import networkx as nx

from structure_aware_chunker import parse_structure_corpus


OUTPUT_FILE = Path(
    "data/evaluation/legal_graph.gpickle"
)


def extract_section_references(text):

    pattern = r"\bsections?\s+(\d+[A-Za-z]?)"

    matches = re.findall(
        pattern,
        text,
        flags=re.IGNORECASE
    )

    references = []

    for section in matches:

        section = section.lower()

        if section not in references:
            references.append(section)

    return references


def build_legal_graph():

    print("=" * 80)
    print("BUILDING LEGAL KNOWLEDGE GRAPH")
    print("=" * 80)

    print("\nLoading legal documents...")

    documents = parse_structure_corpus()

    print(
        f"Documents loaded: {len(documents)}"
    )

    graph = nx.DiGraph()

    section_nodes = {}

    print("\nBuilding hierarchy...")

    # --------------------------------------------------
    # PASS 1: Build legislative hierarchy
    # --------------------------------------------------

    for document in documents:

        metadata = document.metadata

        act = metadata.get("act", "")
        section = metadata.get("section", "")
        subsection = metadata.get("subsection", "")
        paragraph = metadata.get("paragraph", "")
        nested_paragraph = metadata.get(
            "nested_paragraph",
            ""
        )

        # ----------------------------------------------
        # Section
        # ----------------------------------------------

        section_id = (
            f"{act}|section|{section}"
        )

        graph.add_node(
            section_id,
            node_type="section",
            act=act,
            section=section,
            heading=metadata.get(
                "heading",
                ""
            )
        )

        section_nodes[
            (act.lower(), section.lower())
        ] = section_id

        # ----------------------------------------------
        # Subsection
        # ----------------------------------------------

        subsection_id = None

        if subsection:

            subsection_id = (
                f"{act}|subsection|"
                f"{section}|{subsection}"
            )

            graph.add_node(
                subsection_id,
                node_type="subsection",
                act=act,
                section=section,
                subsection=subsection
            )

            graph.add_edge(
                section_id,
                subsection_id,
                relationship="contains"
            )

        # ----------------------------------------------
        # Paragraph
        # ----------------------------------------------

        paragraph_id = None

        if paragraph:

            paragraph_id = (
                f"{act}|paragraph|"
                f"{section}|{subsection}|"
                f"{paragraph}"
            )

            graph.add_node(
                paragraph_id,
                node_type="paragraph",
                act=act,
                section=section,
                subsection=subsection,
                paragraph=paragraph
            )

            parent_id = (
                subsection_id
                if subsection_id
                else section_id
            )

            graph.add_edge(
                parent_id,
                paragraph_id,
                relationship="contains"
            )

        # ----------------------------------------------
        # Nested paragraph
        # ----------------------------------------------

        if nested_paragraph:

            nested_id = (
                f"{act}|nested|"
                f"{section}|{subsection}|"
                f"{paragraph}|"
                f"{nested_paragraph}"
            )

            graph.add_node(
                nested_id,
                node_type="nested_paragraph",
                act=act,
                section=section,
                subsection=subsection,
                paragraph=paragraph,
                nested_paragraph=nested_paragraph
            )

            parent_id = (
                paragraph_id
                if paragraph_id
                else (
                    subsection_id
                    if subsection_id
                    else section_id
                )
            )

            graph.add_edge(
                parent_id,
                nested_id,
                relationship="contains"
            )

    print(
        f"Hierarchy nodes: {graph.number_of_nodes()}"
    )

    print(
        f"Hierarchy edges: {graph.number_of_edges()}"
    )

    # --------------------------------------------------
    # PASS 2: Add cross-reference edges
    # --------------------------------------------------

    print("\nFinding legal cross-references...")

    reference_count = 0

    for document in documents:

        metadata = document.metadata

        act = metadata.get("act", "")
        section = metadata.get("section", "")

        source_id = section_nodes.get(
            (
                act.lower(),
                section.lower()
            )
        )

        if not source_id:
            continue

        references = extract_section_references(
            document.page_content
        )

        for referenced_section in references:

            target_id = section_nodes.get(
                (
                    act.lower(),
                    referenced_section
                )
            )

            if not target_id:
                continue

            if source_id == target_id:
                continue

            graph.add_edge(
                source_id,
                target_id,
                relationship="references"
            )

            reference_count += 1

    print(
        f"Cross-reference edges added: "
        f"{reference_count}"
    )

    # --------------------------------------------------
    # Save graph
    # --------------------------------------------------

    print("\nSaving graph...")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "wb"
    ) as file:

        pickle.dump(
            graph,
            file,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print(
        f"\nFinal nodes: "
        f"{graph.number_of_nodes()}"
    )

    print(
        f"Final edges: "
        f"{graph.number_of_edges()}"
    )

    print(
        f"\nGraph saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    build_legal_graph()

