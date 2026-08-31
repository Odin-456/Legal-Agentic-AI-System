from structure_retrieval import StructureAwareRetriever


QUESTIONS = [
    {
        "question": "What must a foreign person do when land subsequently becomes agricultural land?",
        "expected": "7 CFR 781.3 (d)",
    },
    {
        "question": "When must a foreign person report an interest in agricultural land?",
        "expected": "7 CFR 781.3 (b)",
    },
    {
        "question": "What happens when agricultural land later ceases to be agricultural?",
        "expected": "7 CFR 781.3 (i)",
    },
    {
        "question": "Where must reports concerning agricultural land be filed?",
        "expected": "7 CFR 781.3 (a)",
    },
    {
        "question": "What information must be included about agricultural land in the report?",
        "expected": "7 CFR 781.3 (e)",
    },
]


def citation_matches(result, expected):
    metadata = result.get("metadata", {})

    citation = metadata.get("citation", "")
    paragraph = metadata.get("paragraph")

    if paragraph:
        full_citation = f"{citation} {paragraph}"
    else:
        full_citation = citation

    return full_citation == expected


def evaluate():

    print("=" * 80)
    print("FINAL RETRIEVAL EVALUATION")
    print("=" * 80)

    retriever = StructureAwareRetriever()

    baseline_correct = 0
    structure_correct = 0

    for number, item in enumerate(QUESTIONS, start=1):

        question = item["question"]
        expected = item["expected"]

        print()
        print("-" * 80)
        print(f"Question {number}")
        print(f"Question: {question}")
        print(f"Expected: {expected}")
        print("-" * 80)

        # Baseline
        baseline_results = retriever.semantic_retrieve(
            question,
            top_k=5
        )

        baseline_found = any(
            citation_matches(result, expected)
            for result in baseline_results
        )

        # Structure-aware
        structure_results = retriever.retrieve(
            question,
            top_k=5,
            max_related=10
        )

        structure_found = any(
            citation_matches(result, expected)
            for result in structure_results
        )

        print(
            f"Baseline: "
            f"{'FOUND' if baseline_found else 'NOT FOUND'}"
        )

        print(
            f"Structure-aware: "
            f"{'FOUND' if structure_found else 'NOT FOUND'}"
        )

        if baseline_found:
            baseline_correct += 1

        if structure_found:
            structure_correct += 1

    total = len(QUESTIONS)

    baseline_recall = (
        baseline_correct / total
    ) * 100

    structure_recall = (
        structure_correct / total
    ) * 100

    improvement = (
        structure_recall
        -
        baseline_recall
    )

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"Total questions: {total}")

    print(
        f"Baseline Recall@5: "
        f"{baseline_recall:.2f}%"
    )

    print(
        f"Structure-aware Recall@5: "
        f"{structure_recall:.2f}%"
    )

    print(
        f"Improvement: "
        f"{improvement:+.2f}%"
    )


if __name__ == "__main__":
    evaluate()