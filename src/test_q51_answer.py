import json

from final_experiment_evaluation import (
    retrieve_structure,
    generate_answer,
    evaluate_answer,
    build_ground_truth_lookup,
)


QUESTION_ID = 51


with open(
    "data/evaluation/qa_evaluation_questions.json",
    "r",
    encoding="utf-8"
) as f:
    dataset = json.load(f)


question_item = next(
    item
    for item in dataset["questions"]
    if item["id"] == QUESTION_ID
)


question = question_item["question"]
expected_act = question_item["act"]
expected_section = question_item["section"]


print("=" * 80)
print(f"TESTING Q{QUESTION_ID}")
print("=" * 80)

print()
print("Question:")
print(question)

print()
print("Expected:")
print(f"Act: {expected_act}")
print(f"Section: {expected_section}")


# ------------------------------------------------------------
# Ground truth
# ------------------------------------------------------------

ground_truth_lookup = build_ground_truth_lookup()

ground_truth_document = ground_truth_lookup.get(
    (
        expected_act.lower(),
        str(expected_section).lower()
    )
)


# ------------------------------------------------------------
# Structure-aware retrieval
# ------------------------------------------------------------

print()
print("Running structure-aware retrieval...")

structure_documents, structure_context_documents = (
    retrieve_structure(question)
)


print()
print("Retrieved target sections:")

for document in structure_documents:
    metadata = document.metadata
    print(
        f"- {metadata.get('act')} "
        f"s {metadata.get('section')} "
        f"| {metadata.get('heading')}"
    )


# ------------------------------------------------------------
# Generate answer
# ------------------------------------------------------------

print()
print("Generating answer...")

answer = generate_answer(
    question,
    structure_context_documents
)


print()
print("=" * 80)
print("GENERATED ANSWER")
print("=" * 80)
print()
print(answer)


# ------------------------------------------------------------
# Evaluate answer
# ------------------------------------------------------------

print()
print("Evaluating answer...")

evaluation = evaluate_answer(
    question,
    expected_act,
    expected_section,
    answer,
    structure_context_documents,
    ground_truth_document
)


print()
print("=" * 80)
print("Q51 EVALUATION")
print("=" * 80)

print(
    f"Answer correctness: "
    f"{evaluation['answer_correctness']}"
)

print(
    f"Groundedness: "
    f"{evaluation['groundedness']}"
)

print(
    f"Completeness: "
    f"{evaluation['completeness']}"
)

print(
    f"Citation/source accuracy: "
    f"{evaluation['citation_source_accuracy']}"
)

print(
    f"Reasoning score: "
    f"{evaluation['reasoning_score']}"
)

print(
    f"Answers question: "
    f"{evaluation['answers_question']}"
)

print(
    f"Reasoning correct: "
    f"{evaluation['reasoning_correct']}"
)

print(
    f"Target section retrieved: "
    f"{evaluation['target_section_retrieved']}"
)

print(
    f"Hallucination rate: "
    f"{evaluation['hallucination_rate']}"
)

print()
print("Explanation:")
print(evaluation["explanation"])
