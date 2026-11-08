
from pathlib import Path
import sys
import os
import json
import re
import csv
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

XML_DIR = BASE_DIR / "data" / "raw" / "australian_property_law"
QUESTION_FILE = BASE_DIR / "data" / "evaluation" / "qa_evaluation_questions.json"
OUTPUT_DIR = BASE_DIR / "data" / "exploration_figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The original parser uses relative paths such as data/raw/...
# Change to the project root so those paths work.
os.chdir(BASE_DIR)

# Allow this script to import structure_aware_chunker.py from src.
sys.path.insert(0, str(BASE_DIR / "src"))

from structure_aware_chunker import parse_structure_corpus


# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

def normalise_section(section):
    """Normalise section identifiers for comparison."""
    if section is None:
        return ""

    section = str(section).strip().lower()
    section = re.sub(r"^sec[.\s]*", "", section)
    section = section.replace("section", "").strip()
    section = re.sub(r"\s+", "", section)

    return section


def get_act_name(root, file_path):
    """Get the Act name from the XML title."""
    title = root.find(".//title")

    if title is not None:
        act_name = " ".join("".join(title.itertext()).split())

        if act_name:
            return act_name

    # Fallback if the XML title cannot be found.
    return file_path.stem.replace("_", " ").title()


def get_xml_sections():
    """Count unique sections in each XML legislation file."""
    sections_by_act = {}

    xml_files = sorted(XML_DIR.glob("*.xml"))

    if not xml_files:
        raise FileNotFoundError(
            f"No XML files were found in: {XML_DIR}"
        )

    for file_path in xml_files:
        tree = ET.parse(file_path)
        root = tree.getroot()

        act_name = get_act_name(root, file_path)

        sections = set()

        for element in root.findall(".//level[@type='clause']"):
            section_id = element.get("id", "")

            if section_id.startswith("sec."):
                section_number = normalise_section(
                    section_id.replace("sec.", "", 1)
                )

                if section_number:
                    sections.add(section_number)

        sections_by_act[act_name] = sections

    return sections_by_act


def save_bar_chart(
    labels,
    values,
    title,
    x_label,
    y_label,
    filename,
    rotation=0
):
    """Create a bar chart with exact values above each bar."""

    fig, ax = plt.subplots(figsize=(11, 6))

    bars = ax.bar(
        labels,
        values,
        width=0.65
    )

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)

    ax.tick_params(axis="x", labelrotation=rotation)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Display exact values above the bars.
    ax.bar_label(
        bars,
        labels=[f"{value:,}" for value in values],
        padding=4,
        fontsize=9
    )

    # Leave space above the tallest bar for its label.
    if values:
        maximum = max(values)
        ax.set_ylim(0, maximum * 1.15 if maximum > 0 else 1)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    print(f"Saved: {filename}")


# --------------------------------------------------
# 3. EXPLORE THE XML LEGISLATION DATASET
# --------------------------------------------------

print("\n========== XML DATASET EXPLORATION ==========\n")

sections_by_act = get_xml_sections()

section_counts = {
    act: len(sections)
    for act, sections in sections_by_act.items()
}

total_sections = sum(section_counts.values())

print("Number of XML legislation files:", len(sections_by_act))
print("Total unique sections:", f"{total_sections:,}")

print("\nSections in each Act:")

for act, count in section_counts.items():
    print(f"{act}: {count:,}")


# Figure 1: Number of sections in each Act

save_bar_chart(
    labels=list(section_counts.keys()),
    values=list(section_counts.values()),
    title="Number of Sections in Each NSW Property Law Act",
    x_label="Legislation",
    y_label="Number of Sections",
    filename="figure_1_sections_by_act.png",
    rotation=20
)


# --------------------------------------------------
# 4. EXPLORE THE PROCESSED STRUCTURE-AWARE DOCUMENTS
# --------------------------------------------------

print("\n========== STRUCTURE-AWARE DOCUMENT EXPLORATION ==========\n")

documents = parse_structure_corpus()

if not documents:
    raise ValueError(
        "The XML parser did not generate any documents. "
        "Check structure_aware_chunker.py and the XML folder."
    )

print("Total documents generated:", f"{len(documents):,}")

chunks_by_act = Counter()
hierarchy_counts = Counter()
unique_sections_by_act = defaultdict(set)

for document in documents:
    metadata = document.metadata

    act = metadata.get("act", "Unknown")
    section = metadata.get("section", "")
    hierarchy = metadata.get("hierarchy", "Unknown")

    chunks_by_act[act] += 1
    hierarchy_counts[hierarchy] += 1

    if section:
        unique_sections_by_act[act].add(
            normalise_section(section)
        )


print("\nGenerated documents in each Act:")

for act, count in chunks_by_act.items():
    print(f"{act}: {count:,}")

print("\nDocument hierarchy distribution:")

for hierarchy, count in hierarchy_counts.items():
    print(f"{hierarchy}: {count:,}")

print("\nUnique Act-section combinations in generated documents:")

for act, sections in unique_sections_by_act.items():
    print(f"{act}: {len(sections):,}")

total_unique_act_sections = sum(
    len(sections)
    for sections in unique_sections_by_act.values()
)

print(
    "Total unique Act-section combinations:",
    f"{total_unique_act_sections:,}"
)


# Figure 2: Number of generated documents in each Act

ordered_acts = list(section_counts.keys())

chunks_labels = [
    act for act in ordered_acts
    if act in chunks_by_act
]

chunks_values = [
    chunks_by_act[act] for act in chunks_labels
]

save_bar_chart(
    labels=chunks_labels,
    values=chunks_values,
    title="Number of Generated Documents in Each Act",
    x_label="Legislation",
    y_label="Number of Generated Documents",
    filename="figure_2_documents_by_act.png",
    rotation=20
)


# Figure 3: Document hierarchy distribution

hierarchy_labels = list(hierarchy_counts.keys())
hierarchy_values = list(hierarchy_counts.values())

save_bar_chart(
    labels=hierarchy_labels,
    values=hierarchy_values,
    title="Distribution of Legal Document Hierarchy",
    x_label="Document Hierarchy",
    y_label="Number of Generated Documents",
    filename="figure_3_document_hierarchy.png",
    rotation=20
)


# --------------------------------------------------
# 5. EXPLORE THE EVALUATION QUESTION DATASET
# --------------------------------------------------

print("\n========== EVALUATION DATASET EXPLORATION ==========\n")

if not QUESTION_FILE.exists():
    raise FileNotFoundError(
        f"Evaluation question file not found: {QUESTION_FILE}"
    )

with open(QUESTION_FILE, "r", encoding="utf-8") as file:
    question_data = json.load(file)

questions = question_data.get("questions", [])

if not questions:
    raise ValueError(
        "No questions were found in qa_evaluation_questions.json."
    )

print("Total evaluation questions:", f"{len(questions):,}")

questions_by_act = Counter()
question_sections_by_act = defaultdict(set)

for question in questions:
    act = question.get("act", "Unknown")
    section = question.get("section", "")

    questions_by_act[act] += 1

    if section:
        question_sections_by_act[act].add(
            normalise_section(section)
        )

print("\nEvaluation questions in each Act:")

for act, count in questions_by_act.items():
    print(f"{act}: {count:,}")

print("\nUnique target sections in the evaluation questions:")

for act, sections in question_sections_by_act.items():
    print(f"{act}: {len(sections):,}")


# Figure 4: Evaluation question distribution by Act

question_labels = [
    act for act in ordered_acts
    if act in questions_by_act
]

question_values = [
    questions_by_act[act] for act in question_labels
]

save_bar_chart(
    labels=question_labels,
    values=question_values,
    title="Distribution of Evaluation Questions Across Acts",
    x_label="Legislation",
    y_label="Number of Questions",
    filename="figure_4_evaluation_questions_by_act.png",
    rotation=20
)


# --------------------------------------------------
# 6. EVALUATION SECTION COVERAGE
# --------------------------------------------------

print("\n========== EVALUATION SECTION COVERAGE ==========\n")

coverage_labels = []
coverage_values = []
coverage_percentages = []

for act in ordered_acts:
    total_act_sections = len(sections_by_act.get(act, set()))

    target_sections = question_sections_by_act.get(act, set())

    # Only count target sections that also exist in the XML section set.
    matched_sections = (
        target_sections & sections_by_act.get(act, set())
    )

    matched_count = len(matched_sections)

    if total_act_sections > 0:
        coverage_percentage = (
            matched_count / total_act_sections
        ) * 100
    else:
        coverage_percentage = 0

    coverage_labels.append(act)
    coverage_values.append(coverage_percentage)
    coverage_percentages.append(
        (matched_count, total_act_sections)
    )

    print(
        f"{act}: {matched_count:,} / "
        f"{total_act_sections:,} sections "
        f"({coverage_percentage:.2f}%)"
    )

fig, ax = plt.subplots(figsize=(11, 6))

bars = ax.bar(
    coverage_labels,
    coverage_values,
    width=0.65
)

ax.set_title(
    "Evaluation Question Coverage of Legislation Sections",
    fontsize=14,
    pad=15
)

ax.set_xlabel("Legislation", fontsize=11)
ax.set_ylabel("Section Coverage (%)", fontsize=11)

ax.set_ylim(0, 110)
ax.tick_params(axis="x", labelrotation=20)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

# Show exact section counts and percentages above each bar.
coverage_text = [
    f"{matched:,}/{total:,} ({percentage:.1f}%)"
    for (matched, total), percentage
    in zip(coverage_percentages, coverage_values)
]

ax.bar_label(
    bars,
    labels=coverage_text,
    padding=4,
    fontsize=8
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_5_evaluation_section_coverage.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved: figure_5_evaluation_section_coverage.png")


# --------------------------------------------------
# 7. SAVE SUMMARY STATISTICS
# --------------------------------------------------

summary_file = OUTPUT_DIR / "dataset_exploration_summary.csv"

with open(
    summary_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Act",
        "XML Sections",
        "Generated Documents",
        "Unique Generated Sections",
        "Evaluation Questions",
        "Evaluation Target Sections",
        "Matched Evaluation Sections",
        "Section Coverage Percentage"
    ])

    for act in ordered_acts:
        xml_section_count = len(
            sections_by_act.get(act, set())
        )

        generated_document_count = chunks_by_act.get(act, 0)

        generated_section_count = len(
            unique_sections_by_act.get(act, set())
        )

        evaluation_question_count = questions_by_act.get(act, 0)

        target_section_count = len(
            question_sections_by_act.get(act, set())
        )

        matched_section_count = len(
            question_sections_by_act.get(act, set())
            & sections_by_act.get(act, set())
        )

        if xml_section_count > 0:
            coverage = (
                matched_section_count / xml_section_count
            ) * 100
        else:
            coverage = 0

        writer.writerow([
            act,
            xml_section_count,
            generated_document_count,
            generated_section_count,
            evaluation_question_count,
            target_section_count,
            matched_section_count,
            f"{coverage:.2f}"
        ])


# --------------------------------------------------
# 8. FINAL SUMMARY
# --------------------------------------------------

print("\n========== FINAL SUMMARY ==========\n")

print("XML legislation files:", len(sections_by_act))
print("Total XML sections:", f"{total_sections:,}")
print("Total generated documents:", f"{len(documents):,}")
print("Total unique generated Act-section combinations:",
      f"{total_unique_act_sections:,}")
print("Total evaluation questions:", f"{len(questions):,}")

print("\nFigures and summary saved in:")
print(OUTPUT_DIR)

print("\nGenerated files:")
for output_file in sorted(OUTPUT_DIR.iterdir()):
    print(output_file.name)
