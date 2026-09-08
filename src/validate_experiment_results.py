import json
from pathlib import Path


RESULTS_FILE = Path("data/evaluation/final_experiment_results.json")


def main():
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 80)
    print("EXPERIMENT RESULT VALIDATION")
    print("=" * 80)

    print(f"\nTotal questions: {data['total_questions']}")

    baseline_reasoning_correct = 0
    structure_reasoning_correct = 0

    baseline_retrieval_correct = 0
    structure_retrieval_correct = 0

    print("\n" + "-" * 80)
    print("QUESTIONS WHERE RETRIEVAL AND ANSWER QUALITY DIFFER")
    print("-" * 80)

    for result in data["results"]:

        q = result["question_number"]
        question = result["question"]

        baseline = result["baseline"]
        structure = result["structure_aware"]

        baseline_retrieval = baseline["retrieval"]["hit_at_5"]
        structure_retrieval = structure["retrieval"]["hit_at_5"]

        baseline_reasoning = baseline["qa_evaluation"]["reasoning_correct"]
        structure_reasoning = structure["qa_evaluation"]["reasoning_correct"]

        if baseline_retrieval:
            baseline_retrieval_correct += 1

        if structure_retrieval:
            structure_retrieval_correct += 1

        if baseline_reasoning:
            baseline_reasoning_correct += 1

        if structure_reasoning:
            structure_reasoning_correct += 1

        # Show cases where retrieval failed but answer was judged correct
        if not baseline_retrieval and baseline_reasoning:
            print(
                f"\nQ{q} BASELINE:"
                f"\nQuestion: {question}"
                f"\nRetrieval@5: FAIL"
                f"\nReasoning: CORRECT"
                f"\nAnswer: {baseline['answer'][:300]}"
            )

        if not structure_retrieval and structure_reasoning:
            print(
                f"\nQ{q} STRUCTURE:"
                f"\nQuestion: {question}"
                f"\nRetrieval@5: FAIL"
                f"\nReasoning: CORRECT"
                f"\nAnswer: {structure['answer'][:300]}"
            )

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    print(
        f"\nBaseline Recall@5: "
        f"{baseline_retrieval_correct}/{data['total_questions']}"
    )

    print(
        f"Structure Recall@5: "
        f"{structure_retrieval_correct}/{data['total_questions']}"
    )

    print(
        f"\nBaseline reasoning-correct: "
        f"{baseline_reasoning_correct}/{data['total_questions']}"
    )

    print(
        f"Structure reasoning-correct: "
        f"{structure_reasoning_correct}/{data['total_questions']}"
    )


if __name__ == "__main__":
    main()