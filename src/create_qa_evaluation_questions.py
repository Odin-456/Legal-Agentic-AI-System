"""
Create a natural-language QA evaluation dataset for NSW property law.

The target Act and section numbers are fixed from the validated retrieval
evaluation set. An LLM generates natural questions based on the actual
section heading and legal text.

Output:
    data/evaluation/qa_evaluation_questions.json
"""

import json
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from property_law_parser import parse_property_law_corpus


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_FILE = Path(
    "data/evaluation/qa_evaluation_questions.json"
)

MODEL_NAME = "gpt-5-mini"


# These are the same target sections used in the previous
# 59-question retrieval evaluation.
TARGET_SECTIONS = {
    "Real Property Act 1900": [
        "1",
        "14",
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
# HELPERS
# ============================================================

def normalise_section(section_number):
    """
    Normalise section numbers so comparisons are reliable.
    """
    if section_number is None:
        return ""

    return str(section_number).strip()


def extract_json(text):
    """
    Extract JSON from an LLM response.

    Handles:
    - normal JSON
    - JSON inside ```json ... ```
    - extra explanatory text
    """

    text = text.strip()

    # Remove markdown code fences.
    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.strip()

    # First try direct JSON parsing.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array.
    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1:
        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Could not extract valid JSON from LLM response."
    )


def clean_question(question):
    """
    Clean generated question text.
    """

    question = str(question).strip()

    # Remove accidental numbering.
    question = re.sub(
        r"^\s*\d+[\.\)]\s*",
        "",
        question,
    )

    # Remove surrounding quotation marks.
    question = question.strip('"').strip("'").strip()

    # Make sure the question ends with ?.
    if question and not question.endswith("?"):
        question += "?"

    return question


# ============================================================
# LOAD CORPUS
# ============================================================

def load_target_sections():
    """
    Parse the actual NSW legislation corpus and return the
    requested sections.
    """

    print("=" * 80)
    print("LOADING NSW PROPERTY LAW CORPUS")
    print("=" * 80)

    sections = parse_property_law_corpus()

    print(f"\nTotal sections in corpus: {len(sections)}")

    lookup = {}

    for section in sections:

        act = section.get("act", "").strip()

        section_number = normalise_section(
            section.get("section", "")
        )

        key = (act, section_number)

        lookup[key] = section

    selected = []

    missing = []

    for act, target_numbers in TARGET_SECTIONS.items():

        for section_number in target_numbers:

            key = (act, normalise_section(section_number))

            if key not in lookup:
                missing.append(
                    f"{act} Section {section_number}"
                )
                continue

            section = lookup[key]

            selected.append(section)

    print(f"Target sections found: {len(selected)}")
    print(f"Missing sections: {len(missing)}")

    if missing:
        print("\nMissing sections:")

        for item in missing:
            print(f"  - {item}")

    return selected


# ============================================================
# GENERATE QUESTIONS
# ============================================================

def generate_questions_for_act(
    llm,
    act_name,
    sections,
):
    """
    Generate natural-language questions for all selected
    sections belonging to one Act.
    """

    section_information = []

    for section in sections:

        section_number = normalise_section(
            section.get("section", "")
        )

        heading = section.get("heading", "").strip()

        text = section.get("text", "").strip()

        # Limit extremely long sections to keep prompts manageable.
        if len(text) > 12000:
            text = text[:12000]

        section_information.append(
            {
                "section": section_number,
                "heading": heading,
                "text": text,
            }
        )

    prompt = f"""
You are creating a high-quality evaluation dataset for a
research project on legal question answering.

The system being evaluated retrieves NSW property-law legislation
and then answers questions using retrieved legal text.

Generate exactly ONE natural-language legal question for EACH
section provided below.

ACT:
{act_name}

IMPORTANT REQUIREMENTS:

1. Create exactly one question for every supplied section.
2. Do NOT change the section numbers.
3. Do NOT combine multiple sections into one question.
4. The question must be answerable primarily from the supplied
   section text.
5. The question must be natural, as if asked by a person seeking
   legal information.
6. Do NOT copy the section heading directly into the question.
7. Paraphrase the legal issue rather than repeating the heading.
8. Do NOT mention the section number in the question.
9. Do NOT mention "the section", "Section X", "according to the
   Act", or similar retrieval clues.
10. Avoid questions whose answer is simply the name of the section.
11. Prefer questions about:
      - requirements
      - rights
      - duties
      - restrictions
      - exceptions
      - conditions
      - procedures
      - circumstances
      - powers
      - legal consequences
12. Keep each question between approximately 10 and 30 words.
13. Do not invent facts that are not supported by the supplied text.
14. Preserve the legal meaning of the supplied section.
15. Questions should be sufficiently different from the section
    heading that they test semantic retrieval rather than keyword
    matching.
16. Return ONLY valid JSON.

Return this exact JSON structure:

[
  {{
    "section": "SECTION_NUMBER",
    "question": "Natural language question?"
  }}
]

Sections:

{json.dumps(section_information, indent=2)}
"""

    print(f"\nGenerating questions for: {act_name}")
    print(f"Sections: {len(sections)}")

    response = llm.invoke(prompt)

    generated = extract_json(response.content)

    if not isinstance(generated, list):
        raise ValueError(
            f"Expected a JSON list for {act_name}."
        )

    return generated


# ============================================================
# VALIDATE GENERATED QUESTIONS
# ============================================================

def validate_questions(
    questions,
    selected_sections,
):
    """
    Validate that:
    - every target section has exactly one question
    - no target section is missing
    - no unexpected section appears
    - questions are not empty
    """

    expected = {
        (
            section["act"],
            normalise_section(section["section"]),
        )
        for section in selected_sections
    }

    actual = set()

    validated = []

    errors = []

    for item in questions:

        if not isinstance(item, dict):
            errors.append(
                "Generated item is not a JSON object."
            )
            continue

        act = str(item.get("act", "")).strip()
        section = normalise_section(
            item.get("section", "")
        )

        question = clean_question(
            item.get("question", "")
        )

        key = (act, section)

        if key in actual:
            errors.append(
                f"Duplicate question for {act} Section {section}."
            )
            continue

        if key not in expected:
            errors.append(
                f"Unexpected section: {act} Section {section}."
            )
            continue

        if len(question) < 10:
            errors.append(
                f"Question too short: {act} Section {section}."
            )
            continue

        actual.add(key)

        validated.append(
            {
                "id": len(validated) + 1,
                "act": act,
                "section": section,
                "question": question,
            }
        )

    missing = expected - actual

    for act, section in sorted(missing):
        errors.append(
            f"Missing question for {act} Section {section}."
        )

    return validated, errors


# ============================================================
# MAIN
# ============================================================

def main():

    load_dotenv()

    print("\n")
    print("=" * 80)
    print("NATURAL-LANGUAGE PROPERTY LAW QA DATASET")
    print("=" * 80)

    # --------------------------------------------------------
    # Load actual corpus
    # --------------------------------------------------------

    selected_sections = load_target_sections()

    if not selected_sections:
        raise RuntimeError(
            "No target sections were found."
        )

    # --------------------------------------------------------
    # Create LLM
    # --------------------------------------------------------

    print("\nCreating question-generation model...")

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.2,
    )

    # --------------------------------------------------------
    # Generate questions
    # --------------------------------------------------------

    all_questions = []

    # Group sections by Act.
    sections_by_act = {}

    for section in selected_sections:

        act = section["act"]

        if act not in sections_by_act:
            sections_by_act[act] = []

        sections_by_act[act].append(section)

    for act, sections in sections_by_act.items():

        generated = generate_questions_for_act(
            llm=llm,
            act_name=act,
            sections=sections,
        )

        # Add Act information to each generated item.
        for item in generated:

            item["act"] = act

            item["question"] = clean_question(
                item.get("question", "")
            )

        all_questions.extend(generated)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("VALIDATING GENERATED QUESTIONS")
    print("=" * 80)

    validated, errors = validate_questions(
        all_questions,
        selected_sections,
    )

    if errors:

        print("\nValidation warnings/errors:")

        for error in errors:
            print(f"  - {error}")

    print(
        f"\nValid questions: {len(validated)}"
    )

    # --------------------------------------------------------
    # Sort questions by Act and section target order.
    # --------------------------------------------------------

    target_order = {}

    for act, sections in TARGET_SECTIONS.items():

        for index, section in enumerate(sections):

            target_order[
                (act, section)
            ] = index

    validated.sort(
        key=lambda x: (
            list(TARGET_SECTIONS.keys()).index(x["act"]),
            target_order.get(
                (x["act"], x["section"]),
                999,
            ),
        )
    )

    # Reassign IDs after sorting.
    for index, item in enumerate(validated, start=1):
        item["id"] = index

    # --------------------------------------------------------
    # Add dataset metadata
    # --------------------------------------------------------

    dataset = {
        "dataset_name": (
            "NSW Property Law Natural-Language "
            "QA Evaluation Dataset"
        ),
        "description": (
            "Natural-language legal questions targeting "
            "validated sections from five NSW property-law Acts."
        ),
        "question_count": len(validated),
        "acts": list(TARGET_SECTIONS.keys()),
        "questions": validated,
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            dataset,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("QA DATASET CREATED")
    print("=" * 80)

    print(
        f"\nTotal questions: {len(validated)}"
    )

    for act in TARGET_SECTIONS:

        count = sum(
            1
            for question in validated
            if question["act"] == act
        )

        print(
            f"  {act}: {count}"
        )

    print(
        f"\nSaved to:\n  {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Print questions for manual inspection
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("GENERATED QUESTIONS")
    print("=" * 80)

    for item in validated:

        print(
            f"\n[{item['id']}] "
            f"{item['act']} "
            f"Section {item['section']}"
        )

        print(
            f"    {item['question']}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()