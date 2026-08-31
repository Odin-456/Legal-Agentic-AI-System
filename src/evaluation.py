import json
from pathlib import Path

from retrieval import LegalRetriever
from structure_retrieval import StructureAwareRetriever

# ============================================================
# CONFIGURATION
# ============================================================

QUESTIONS_PATH = Path(
    "data/evaluation_questions.json"
)

TOP_K = 5


# ============================================================
# LOAD QUESTIONS
# ============================================================

def load_questions():

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# CHECK EXPECTED CITATION
# ============================================================

def contains_expected(
    results,
    expected
):

    for result in results:

        metadata = result.get(
            "metadata",
            {}
        )

        citation = metadata.get(
            "citation",
            ""
        )

        paragraph = metadata.get(
            "paragraph"
        )

        subparagraph = metadata.get(
            "subparagraph"
        )

        location = citation

        if paragraph:
            location += f" {paragraph}"

        if subparagraph:
            location += f" {subparagraph}"

        if location in expected:
            return True

    return False


# ============================================================
# MAIN
# ============================================================

def main():

    questions = load_questions()

    baseline = LegalRetriever()

    structure = StructureAwareRetriever()

    baseline_correct = 0
    structure_correct = 0

    print()
    print("=" * 80)
    print("RETRIEVAL EVALUATION")
    print("=" * 80)

    for number, item in enumerate(
        questions,
        start=1
    ):

        question = item["question"]
        expected = item["expected"]

        print()
        print("-" * 80)
        print(f"Question {number}")
        print(f"Question: {question}")
        print(f"Expected: {expected}")

        # ----------------------------------------------------
        # Baseline
        # ----------------------------------------------------

        baseline_results = baseline.retrieve(
            question,
            top_k=TOP_K
        )

        baseline_found = contains_expected(
            baseline_results,
            expected
        )

        # ----------------------------------------------------
        # Structure-aware
        # ----------------------------------------------------

        structure_results = structure.retrieve(
            question,
            top_k=TOP_K,
            max_related=5
        )

        structure_found = contains_expected(
            structure_results,
            expected
        )

        if baseline_found:
            baseline_correct += 1

        if structure_found:
            structure_correct += 1

        print(
            "Baseline:",
            "FOUND" if baseline_found else "NOT FOUND"
        )

        print(
            "Structure-aware:",
            "FOUND" if structure_found else "NOT FOUND"
        )

        # ----------------------------------------------------
        # Top results
        # ----------------------------------------------------

        if baseline_results:

            metadata = baseline_results[0].get(
                "metadata",
                {}
            )

            print()
            print("Baseline top result:")
            print(
                metadata.get(
                    "citation",
                    "Unknown"
                )
            )

        if structure_results:

            metadata = structure_results[0].get(
                "metadata",
                {}
            )

            print()
            print("Structure-aware top result:")
            print(
                metadata.get(
                    "citation",
                    "Unknown"
                )
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    total = len(questions)

    baseline_recall = (
        baseline_correct / total
    )

    structure_recall = (
        structure_correct / total
    )

    improvement = (
        structure_recall
        -
        baseline_recall
    )

    print()
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print(
        f"Total questions: {total}"
    )

    print(
        f"Baseline correct retrievals: "
        f"{baseline_correct}"
    )

    print(
        f"Structure-aware correct retrievals: "
        f"{structure_correct}"
    )

    print(
        f"Baseline Recall@{TOP_K}: "
        f"{baseline_recall * 100:.2f}%"
    )

    print(
        f"Structure-aware Recall@{TOP_K}: "
        f"{structure_recall * 100:.2f}%"
    )

    print(
        f"Improvement: "
        f"{improvement * 100:.2f}%"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()