import json
import re
from pathlib import Path


INPUT_PATH = Path(
    "data/evaluation/agent_results.json"
)


def normalize_citation(citation):
    if not citation:
        return ""

    citation = str(citation).strip()

    citation = re.sub(
        r"[.,;:]+$",
        "",
        citation
    )

    return citation


def citation_found(answer, expected_citations):
    """
    Check whether the answer contains at least
    one expected CFR citation.
    """

    if not answer:
        return False

    answer = answer.lower()

    for citation in expected_citations:

        citation = normalize_citation(
            citation
        ).lower()

        if citation in answer:
            return True

    return False


def main():

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        results = json.load(file)

    total = len(results)

    citation_correct = 0

    print("=" * 60)
    print("AGENT ANSWER EVALUATION")
    print("=" * 60)

    for result in results:

        answer = result.get(
            "answer",
            ""
        )

        expected = result.get(
            "expected_citations",
            []
        )

        correct = citation_found(
            answer,
            expected
        )

        if correct:
            citation_correct += 1

        status = "PASS" if correct else "FAIL"

        print(
            f"{result['id']:2d}. "
            f"{status} | "
            f"{result['query']}"
        )

    citation_accuracy = (
        citation_correct / total
        if total
        else 0
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(
        f"Total questions: {total}"
    )

    print(
        f"Citation correct: "
        f"{citation_correct}"
    )

    print(
        f"Citation accuracy: "
        f"{citation_accuracy:.4f}"
    )

    print(
        f"Citation accuracy: "
        f"{citation_accuracy * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
