from pathlib import Path
import json
import re

from property_law_parser import parse_property_law_corpus


# ============================================================
# PATHS
# ============================================================

OUTPUT_DIR = Path("data/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "validated_evaluation_questions.json"


# ============================================================
# YOUR CURRENT 35 EVALUATION QUESTIONS
# ============================================================

QUESTIONS = [
    {
        "id": 1,
        "question": "What exceptions apply to the title of a registered proprietor?",
        "expected_act": "Real Property Act 1900",
        "expected_section": "42",
    },
    {
        "id": 2,
        "question": "What protection is given to a registered proprietor against proceedings?",
        "expected_act": "Real Property Act 1900",
        "expected_section": "118",
    },
    {
        "id": 3,
        "question": "When can a person apply for title by possession?",
        "expected_act": "Real Property Act 1900",
        "expected_section": "45D",
    },
    {
        "id": 4,
        "question": "When can a person acquire possessory title to land?",
        "expected_act": "Real Property Act 1900",
        "expected_section": "45C",
    },
    {
        "id": 5,
        "question": "Can a purchaser from a registered proprietor be affected by notice of another interest?",
        "expected_act": "Real Property Act 1900",
        "expected_section": "43",
    },
    {
        "id": 6,
        "question": "When may the Registrar-General register a person as proprietor because of a statutory entitlement?",
        "expected_act": "Real Property Act 1900",
        "expected_section": "46C",
    },
    {
        "id": 7,
        "question": "What happens when an estate or interest is recorded in a folio of the Register?",
        "expected_act": "Real Property Act 1900",
        "expected_section": "41",
    },
    {
        "id": 8,
        "question": "What types of interests may remain protected despite registration?",
        "expected_act": "Real Property Act 1900",
        "expected_section": "42",
    },
    {
        "id": 9,
        "question": "What restrictions apply to acquiring title by adverse possession?",
        "expected_act": "Real Property Act 1900",
        "expected_section": "45D",
    },
    {
        "id": 10,
        "question": "What conditions apply to a possessory application?",
        "expected_act": "Real Property Act 1900",
        "expected_section": "45D",
    },
    {
        "id": 11,
        "question": "What is required for a contract for the sale of land?",
        "expected_act": "Conveyancing Act 1919",
        "expected_section": "54A",
    },
    {
        "id": 12,
        "question": "When must a contract for the sale of land be in writing?",
        "expected_act": "Conveyancing Act 1919",
        "expected_section": "54A",
    },
    {
        "id": 13,
        "question": "What rules apply to the execution of deeds?",
        "expected_act": "Conveyancing Act 1919",
        "expected_section": "38",
    },
    {
        "id": 14,
        "question": "What powers does a mortgagee have in relation to mortgaged property?",
        "expected_act": "Conveyancing Act 1919",
        "expected_section": "111",
    },
    {
        "id": 15,
        "question": "What is the effect of a contract for the sale of land?",
        "expected_act": "Conveyancing Act 1919",
        "expected_section": "54",
    },
    {
        "id": 16,
        "question": "What requirements apply to a conveyance of land?",
        "expected_act": "Conveyancing Act 1919",
        "expected_section": "23B",
    },
    {
        "id": 17,
        "question": "What obligations does a landlord have under a residential tenancy agreement?",
        "expected_act": "Residential Tenancies Act 2010",
        "expected_section": "52",
    },
    {
        "id": 18,
        "question": "What obligations does a tenant have under a residential tenancy agreement?",
        "expected_act": "Residential Tenancies Act 2010",
        "expected_section": "51",
    },
    {
        "id": 19,
        "question": "When can a landlord enter residential premises?",
        "expected_act": "Residential Tenancies Act 2010",
        "expected_section": "55",
    },
    {
        "id": 20,
        "question": "What rules apply to rent increases?",
        "expected_act": "Residential Tenancies Act 2010",
        "expected_section": "41",
    },
    {
        "id": 21,
        "question": "When can a residential tenancy agreement be terminated?",
        "expected_act": "Residential Tenancies Act 2010",
        "expected_section": "100",
    },
    {
        "id": 22,
        "question": "What repairs is a landlord responsible for?",
        "expected_act": "Residential Tenancies Act 2010",
        "expected_section": "63",
    },
    {
        "id": 23,
        "question": "What rights does a tenant have concerning repairs?",
        "expected_act": "Residential Tenancies Act 2010",
        "expected_section": "64",
    },
    {
        "id": 24,
        "question": "What notice is required to terminate a residential tenancy?",
        "expected_act": "Residential Tenancies Act 2010",
        "expected_section": "100",
    },
    {
        "id": 25,
        "question": "What are the duties of an owners corporation?",
        "expected_act": "Strata Schemes Management Act 2015",
        "expected_section": "9",
    },
    {
        "id": 26,
        "question": "Who is responsible for maintaining common property?",
        "expected_act": "Strata Schemes Management Act 2015",
        "expected_section": "106",
    },
    {
        "id": 27,
        "question": "What rules apply to by-laws in a strata scheme?",
        "expected_act": "Strata Schemes Management Act 2015",
        "expected_section": "135",
    },
    {
        "id": 28,
        "question": "What is a strata committee?",
        "expected_act": "Strata Schemes Management Act 2015",
        "expected_section": "30",
    },
    {
        "id": 29,
        "question": "What powers does an owners corporation have?",
        "expected_act": "Strata Schemes Management Act 2015",
        "expected_section": "9",
    },
    {
        "id": 30,
        "question": "What obligations apply to owners corporations regarding common property?",
        "expected_act": "Strata Schemes Management Act 2015",
        "expected_section": "106",
    },
    {
        "id": 31,
        "question": "What is a community association?",
        "expected_act": "Community Land Management Act 2021",
        "expected_section": "6",
    },
    {
        "id": 32,
        "question": "What functions does a community association have?",
        "expected_act": "Community Land Management Act 2021",
        "expected_section": "8",
    },
    {
        "id": 33,
        "question": "What duties does a community association have?",
        "expected_act": "Community Land Management Act 2021",
        "expected_section": "9",
    },
    {
        "id": 34,
        "question": "What rules apply to association property?",
        "expected_act": "Community Land Management Act 2021",
        "expected_section": "25",
    },
    {
        "id": 35,
        "question": "What is the role of the association committee?",
        "expected_act": "Community Land Management Act 2021",
        "expected_section": "30",
    },
]


# ============================================================
# NORMALISE SECTION NUMBERS
# ============================================================

def normalise_section(value):
    """
    Normalise section numbers so values such as:
        42
        42A
        42(1)
    can be compared consistently.
    """
    if value is None:
        return ""

    value = str(value).strip()

    match = re.match(r"^([0-9]+[A-Za-z]*)", value)

    if match:
        return match.group(1)

    return value


# ============================================================
# BUILD CORPUS INDEX
# ============================================================

def build_corpus_index(sections):
    """
    Create an index:

        Act -> section -> section information
    """

    index = {}

    for section in sections:
        act = section["act"]
        section_number = normalise_section(section["section"])

        if act not in index:
            index[act] = {}

        index[act][section_number] = {
            "section": section_number,
            "heading": section["heading"],
            "text": section["text"],
        }

    return index


# ============================================================
# FIND POSSIBLE SECTION MATCHES
# ============================================================

def find_similar_sections(act_index, expected_section):
    """
    If the expected section does not exist, show nearby
    section numbers to help identify the correct section.
    """

    numbers = list(act_index.keys())

    numeric_expected = None

    match = re.match(r"^(\d+)", expected_section)

    if match:
        numeric_expected = int(match.group(1))

    if numeric_expected is None:
        return numbers[:10]

    scored = []

    for number in numbers:
        match = re.match(r"^(\d+)", number)

        if not match:
            continue

        numeric_number = int(match.group(1))
        difference = abs(numeric_number - numeric_expected)

        scored.append((difference, number))

    scored.sort()

    return [number for _, number in scored[:10]]


# ============================================================
# VALIDATE QUESTIONS
# ============================================================

def validate_questions(questions, corpus_index):

    results = []

    valid_count = 0
    invalid_count = 0
    missing_act_count = 0

    for item in questions:

        question_id = item["id"]
        question = item["question"]
        expected_act = item["expected_act"]
        expected_section = normalise_section(item["expected_section"])

        result = {
            "id": question_id,
            "question": question,
            "expected_act": expected_act,
            "expected_section": expected_section,
            "status": None,
            "actual_heading": None,
            "suggested_sections": [],
        }

        # ----------------------------------------------------
        # Check Act
        # ----------------------------------------------------

        if expected_act not in corpus_index:

            result["status"] = "INVALID_ACT"
            result["suggested_sections"] = list(corpus_index.keys())

            missing_act_count += 1
            invalid_count += 1

            results.append(result)
            continue

        act_index = corpus_index[expected_act]

        # ----------------------------------------------------
        # Check section
        # ----------------------------------------------------

        if expected_section in act_index:

            section_info = act_index[expected_section]

            result["status"] = "VALID"
            result["actual_heading"] = section_info["heading"]

            valid_count += 1

        else:

            result["status"] = "INVALID_SECTION"

            result["suggested_sections"] = find_similar_sections(
                act_index,
                expected_section,
            )

            invalid_count += 1

        results.append(result)

    return results, valid_count, invalid_count, missing_act_count


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(results):

    print()
    print("=" * 80)
    print("EVALUATION QUESTION VALIDATION")
    print("=" * 80)

    for result in results:

        print()
        print(f"[{result['id']}] {result['question']}")
        print(f"    Act: {result['expected_act']}")
        print(f"    Expected section: {result['expected_section']}")
        print(f"    Status: {result['status']}")

        if result["actual_heading"]:
            print(f"    Actual heading: {result['actual_heading']}")

        if result["suggested_sections"]:
            print(
                "    Suggested section numbers: "
                + ", ".join(result["suggested_sections"])
            )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    valid_questions = []

    for result in results:

        if result["status"] != "VALID":
            continue

        valid_questions.append(
            {
                "id": result["id"],
                "question": result["question"],
                "expected_act": result["expected_act"],
                "expected_section": result["expected_section"],
            }
        )

    output = {
        "total_questions": len(results),
        "valid_questions": len(valid_questions),
        "invalid_questions": len(results) - len(valid_questions),
        "questions": valid_questions,
        "validation_results": results,
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    return OUTPUT_FILE


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("VALIDATING PROPERTY LAW EVALUATION QUESTIONS")
    print("=" * 80)

    print()
    print("Parsing NSW property-law corpus...")

    sections = parse_property_law_corpus()

    print()
    print(f"Sections parsed: {len(sections)}")

    print()
    print("Building corpus index...")

    corpus_index = build_corpus_index(sections)

    print(f"Acts indexed: {len(corpus_index)}")

    for act, act_sections in corpus_index.items():
        print(f"  {act}: {len(act_sections)} sections")

    print()
    print(f"Evaluation questions: {len(QUESTIONS)}")

    results, valid_count, invalid_count, missing_act_count = (
        validate_questions(
            QUESTIONS,
            corpus_index,
        )
    )

    print_results(results)

    output_file = save_results(results)

    print()
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    print(f"Total questions: {len(QUESTIONS)}")
    print(f"Valid questions: {valid_count}")
    print(f"Invalid questions: {invalid_count}")
    print(f"Missing Acts: {missing_act_count}")

    print()
    print(f"Saved validation results to:")
    print(f"  {output_file}")

    print()
    print("Validation complete.")


if __name__ == "__main__":
    main()