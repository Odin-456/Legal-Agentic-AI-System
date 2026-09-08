from pathlib import Path
import json

from property_law_parser import parse_property_law_corpus


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("data/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "evaluation_questions_60.json"


# ============================================================
# QUESTION TEMPLATES
# ============================================================

QUESTION_TEMPLATES = [
    "What is the purpose of {heading}?",
    "What rules apply under {heading}?",
    "What requirements apply under {heading}?",
    "What does {heading} provide for?",
    "When does {heading} apply?",
    "What obligations arise under {heading}?",
    "What powers are provided under {heading}?",
    "What conditions apply under {heading}?",
    "What happens under {heading}?",
    "Who is affected by {heading}?",
    "What protections are provided under {heading}?",
    "What restrictions apply under {heading}?",
]


# ============================================================
# SECTION SELECTION
# ============================================================

# We deliberately select sections from different parts of
# each Act rather than only selecting the first sections.
#
# The section numbers below are taken from the actual corpus
# after validation.
#
# If a section does not exist, it is skipped automatically.

TARGET_SECTIONS = {
    "Real Property Act 1900": [
        "1",
        "14",
        "22",
        "32",
        "41",
        "42",
        "43",
        "45C",
        "45D",
        "46C",
        "74F",
        "118",
    ],

    "Conveyancing Act 1919": [
        "23B",
        "38",
        "41",
        "54",
        "54A",
        "55",
        "66G",
        "73",
        "75",
        "109",
        "111",
        "113",
    ],

    "Residential Tenancies Act 2010": [
        "21",
        "22",
        "24",
        "41",
        "51",
        "52",
        "55",
        "63",
        "64",
        "87",
        "100",
        "187",
    ],

    "Strata Schemes Management Act 2015": [
        "9",
        "10",
        "11",
        "30",
        "31",
        "32",
        "44",
        "50",
        "106",
        "135",
        "139",
        "154",
    ],

    "Community Land Management Act 2021": [
        "5",
        "6",
        "8",
        "9",
        "25",
        "30",
        "31",
        "32",
        "34",
        "40",
        "50",
        "100",
    ],
}


# ============================================================
# NORMALISE SECTION NUMBER
# ============================================================

def normalise_section(value):
    if value is None:
        return ""

    return str(value).strip()


# ============================================================
# BUILD CORPUS INDEX
# ============================================================

def build_index(sections):

    index = {}

    for section in sections:

        act = section["act"]
        section_number = normalise_section(section["section"])

        if act not in index:
            index[act] = {}

        index[act][section_number] = section

    return index


# ============================================================
# CREATE QUESTION
# ============================================================

def create_question(section, template_index):

    heading = section["heading"]

    template = QUESTION_TEMPLATES[template_index]

    question = template.format(
        heading=heading
    )

    return question


# ============================================================
# MAIN QUESTION GENERATION
# ============================================================

def generate_questions(corpus_index):

    questions = []

    question_id = 1

    for act, target_sections in TARGET_SECTIONS.items():

        if act not in corpus_index:
            print(f"WARNING: Act not found: {act}")
            continue

        act_index = corpus_index[act]

        print()
        print("-" * 80)
        print(act)
        print("-" * 80)

        template_index = 0

        for section_number in target_sections:

            section_number = normalise_section(section_number)

            if section_number not in act_index:

                print(
                    f"  WARNING: Section {section_number} "
                    f"not found"
                )

                continue

            section = act_index[section_number]

            question = create_question(
                section,
                template_index
            )

            item = {
                "id": question_id,
                "question": question,
                "expected_act": act,
                "expected_section": section_number,
                "expected_heading": section["heading"],
            }

            questions.append(item)

            print(
                f"  [{question_id}] "
                f"Section {section_number} - "
                f"{section['heading']}"
            )

            question_id += 1

            template_index += 1

            if template_index >= len(QUESTION_TEMPLATES):
                template_index = 0

    return questions


# ============================================================
# SAVE
# ============================================================

def save_questions(questions):

    output = {
        "total_questions": len(questions),
        "questions": questions,
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=4,
            ensure_ascii=False
        )

    return OUTPUT_FILE


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("CREATING PROPERTY LAW EVALUATION QUESTIONS")
    print("=" * 80)

    print()
    print("Parsing NSW property-law corpus...")

    sections = parse_property_law_corpus()

    print()
    print(f"Sections parsed: {len(sections)}")

    print()
    print("Building corpus index...")

    corpus_index = build_index(sections)

    print(
        f"Acts indexed: {len(corpus_index)}"
    )

    questions = generate_questions(
        corpus_index
    )

    output_file = save_questions(
        questions
    )

    print()
    print("=" * 80)
    print("QUESTION GENERATION SUMMARY")
    print("=" * 80)

    print(
        f"Total questions created: {len(questions)}"
    )

    for act in TARGET_SECTIONS:

        count = sum(
            1
            for q in questions
            if q["expected_act"] == act
        )

        print(
            f"  {act}: {count}"
        )

    print()
    print("Saved to:")
    print(f"  {output_file}")

    print()
    print("Question generation complete.")


if __name__ == "__main__":
    main()