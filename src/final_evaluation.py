import json
from pathlib import Path

from structure_retrieval import StructureAwareRetriever


QUESTIONS_FILE = Path(
    "data/evaluation/qa_evaluation_questions.json"
)


def load_questions():
    """
    Load the NSW property-law evaluation questions.
    """

    if not QUESTIONS_FILE.exists():

        raise FileNotFoundError(
            f"Evaluation questions file not found: "
            f"{QUESTIONS_FILE}"
        )

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        for key in [
            "questions",
            "evaluation_questions",
            "data",
        ]:

            if key in data:
                return data[key]

    raise ValueError(
        "Could not determine the evaluation-question "
        "list in the JSON file."
    )


def get_expected_act(question):
    """
    Extract the expected Act from different possible
    evaluation-question JSON formats.
    """

    for key in [
        "act",
        "expected_act",
        "expectedAct",
        "source_act",
    ]:

        if key in question:
            return str(question[key]).strip()

    expected = question.get("expected")

    if isinstance(expected, dict):

        for key in [
            "act",
            "expected_act",
        ]:

            if key in expected:
                return str(expected[key]).strip()

    return ""


def get_expected_section(question):
    """
    Extract the expected section from different possible
    evaluation-question JSON formats.
    """

    for key in [
        "section",
        "expected_section",
        "expectedSection",
        "source_section",
    ]:

        if key in question:
            return str(question[key]).strip()

    expected = question.get("expected")

    if isinstance(expected, dict):

        for key in [
            "section",
            "expected_section",
        ]:

            if key in expected:
                return str(expected[key]).strip()

    return ""


def get_question_text(question):
    """
    Extract the actual question text.
    """

    for key in [
        "question",
        "query",
        "text",
    ]:

        if key in question:
            return str(question[key]).strip()

    return ""


def normalise_section(section):
    """
    Normalise section strings for comparison.
    """

    return (
        str(section)
        .strip()
        .replace("§", "")
        .strip()
    )


def result_matches(result, expected_act, expected_section):
    """
    Check whether a retrieval result belongs to the
    expected Act and section.
    """

    metadata = result.get(
        "metadata",
        {}
    )

    actual_act = str(
        metadata.get(
            "act",
            ""
        )
    ).strip()

    actual_section = normalise_section(
        metadata.get(
            "section",
            ""
        )
    )

    expected_section = normalise_section(
        expected_section
    )

    act_matches = (
        not expected_act
        or actual_act.lower() == expected_act.lower()
    )

    section_matches = (
        not expected_section
        or actual_section == expected_section
    )

    return act_matches and section_matches


def find_rank(results, expected_act, expected_section):
    """
    Return the first relevant rank, or None.
    """

    for rank, result in enumerate(
        results,
        start=1,
    ):

        if result_matches(
            result,
            expected_act,
            expected_section,
        ):

            return rank

    return None


def reciprocal_rank(rank):
    """
    Calculate reciprocal rank.
    """

    if rank is None:
        return 0.0

    return 1.0 / rank


def main():

    print("=" * 80)
    print("FINAL NSW PROPERTY-LAW RETRIEVAL EVALUATION")
    print("=" * 80)

    questions = load_questions()

    print()
    print(
        f"Evaluation questions: "
        f"{len(questions)}"
    )

    retriever = StructureAwareRetriever()

    baseline_hits = 0
    structure_hits = 0

    baseline_rr_total = 0.0
    structure_rr_total = 0.0

    baseline_top5_hits = 0
    structure_top5_hits = 0

    valid_questions = 0

    results_output = []

    print()
    print("=" * 80)
    print("RUNNING EVALUATION")
    print("=" * 80)

    for i, question_data in enumerate(
        questions,
        start=1,
    ):

        question = get_question_text(
            question_data
        )

        expected_act = get_expected_act(
            question_data
        )

        expected_section = get_expected_section(
            question_data
        )

        if not question:
            print(
                f"[{i}/{len(questions)}] "
                "Skipped: missing question text"
            )
            continue

        if not expected_act or not expected_section:

            print(
                f"[{i}/{len(questions)}] "
                f"{question}"
            )

            print(
                "  WARNING: missing expected Act "
                "or section"
            )

            continue

        valid_questions += 1

        baseline_results = (
            retriever.semantic_retrieve(
                question,
                top_k=5,
            )
        )

        structure_results = retriever.retrieve(
            question,
            top_k=5,
            max_related=10,
        )

        baseline_rank = find_rank(
            baseline_results,
            expected_act,
            expected_section,
        )

        structure_rank = find_rank(
            structure_results,
            expected_act,
            expected_section,
        )

        if baseline_rank == 1:
            baseline_hits += 1

        if structure_rank == 1:
            structure_hits += 1

        if baseline_rank is not None:
            baseline_rr_total += reciprocal_rank(
                baseline_rank
            )

        if structure_rank is not None:
            structure_rr_total += reciprocal_rank(
                structure_rank
            )

        if (
            baseline_rank is not None
            and baseline_rank <= 5
        ):
            baseline_top5_hits += 1

        if (
            structure_rank is not None
            and structure_rank <= 5
        ):
            structure_top5_hits += 1

        print()
        print(
            f"[{i}/{len(questions)}] "
            f"{question}"
        )

        print(
            f"  Expected: "
            f"{expected_act} § {expected_section}"
        )

        print(
            f"  Baseline rank: "
            f"{baseline_rank if baseline_rank else 'N/A'}"
        )

        print(
            f"  Structure-aware rank: "
            f"{structure_rank if structure_rank else 'N/A'}"
        )

        results_output.append(
            {
                "question": question,
                "expected_act": expected_act,
                "expected_section": expected_section,
                "baseline_rank": baseline_rank,
                "structure_aware_rank": structure_rank,
            }
        )

    if valid_questions == 0:

        print()
        print(
            "No valid evaluation questions were found."
        )
        return

    baseline_top1 = (
        baseline_hits / valid_questions
    )

    structure_top1 = (
        structure_hits / valid_questions
    )

    baseline_top5 = (
        baseline_top5_hits / valid_questions
    )

    structure_top5 = (
        structure_top5_hits / valid_questions
    )

    baseline_mrr = (
        baseline_rr_total / valid_questions
    )

    structure_mrr = (
        structure_rr_total / valid_questions
    )

    print()
    print("=" * 80)
    print("FINAL RETRIEVAL EVALUATION RESULTS")
    print("=" * 80)

    print()
    print(
        f"Valid questions: "
        f"{valid_questions}"
    )

    print()
    print("TOP-1 SECTION RETRIEVAL")

    print(
        f"Baseline: "
        f"{baseline_top1:.4f} "
        f"({baseline_top1 * 100:.2f}%)"
    )

    print(
        f"Structure-aware: "
        f"{structure_top1:.4f} "
        f"({structure_top1 * 100:.2f}%)"
    )

    print(
        f"Difference: "
        f"{structure_top1 - baseline_top1:+.4f} "
        f"({(structure_top1 - baseline_top1) * 100:+.2f}%)"
    )

    print()
    print("TOP-5 SECTION RETRIEVAL")

    print(
        f"Baseline: "
        f"{baseline_top5:.4f} "
        f"({baseline_top5 * 100:.2f}%)"
    )

    print(
        f"Structure-aware: "
        f"{structure_top5:.4f} "
        f"({structure_top5 * 100:.2f}%)"
    )

    print(
        f"Difference: "
        f"{structure_top5 - baseline_top5:+.4f} "
        f"({(structure_top5 - baseline_top5) * 100:+.2f}%)"
    )

    print()
    print("MEAN RECIPROCAL RANK (MRR)")

    print(
        f"Baseline: "
        f"{baseline_mrr:.4f}"
    )

    print(
        f"Structure-aware: "
        f"{structure_mrr:.4f}"
    )

    print(
        f"Difference: "
        f"{structure_mrr - baseline_mrr:+.4f}"
    )

    output_path = Path(
        "data/evaluation/"
        "final_retrieval_evaluation_results.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_output = {
        "total_questions": len(questions),
        "valid_questions": valid_questions,
        "baseline": {
            "top1": baseline_top1,
            "top5": baseline_top5,
            "mrr": baseline_mrr,
        },
        "structure_aware": {
            "top1": structure_top1,
            "top5": structure_top5,
            "mrr": structure_mrr,
        },
        "difference": {
            "top1": structure_top1 - baseline_top1,
            "top5": structure_top5 - baseline_top5,
            "mrr": structure_mrr - baseline_mrr,
        },
        "questions": results_output,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"Results saved to:\n"
        f"  {output_path}"
    )

    print()
    print("Evaluation complete.")


if __name__ == "__main__":
    main()