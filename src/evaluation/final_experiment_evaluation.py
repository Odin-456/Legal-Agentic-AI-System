import json
import sys

from requests.exceptions import Timeout

from experiment_config import (
    ANSWER_MODEL, BASELINE_VECTORSTORE, EMBEDDING_MODEL, EVALUATOR_MODEL,
    LOCAL_MODELS, OUTPUT_FILE, QUESTIONS_FILE, STRUCTURE_CANDIDATES, TOP_K,
    QWEN_MODEL, LLAMA_MODEL, baseline_vectorstore, structure_retriever,
)
from answer_generation import generate_answer, generate_local_answer
from answer_evaluation import (
    add_metric_values, document_is_relevant, evaluate_one_answer,
    new_metric_store, normalize_act, normalize_section, summarise_metrics,
)
from answer_generation import get_metadata, get_page_content


def deduplicate_documents(documents):
    unique_documents = []
    seen = set()
    for document in documents:
        metadata = get_metadata(document)
        key = (
            metadata.get("act", ""),
            metadata.get("section", ""),
            metadata.get("subsection", ""),
            metadata.get("paragraph", ""),
            metadata.get("nested_paragraph", ""),
            get_page_content(document),
        )
        if key not in seen:
            seen.add(key)
            unique_documents.append(document)
    return unique_documents


def retrieve_baseline(question):
    return baseline_vectorstore.similarity_search(question, k=TOP_K)


def retrieve_structure(question, retriever):
    results = retriever.retrieve(question, top_k=TOP_K)
    documents = []
    context_documents = []
    for result in results:
        document = result.get("document")
        if document is not None:
            documents.append(document)
        context_documents.extend(result.get("related_documents", []))
    documents = deduplicate_documents(documents)
    context_documents = deduplicate_documents(context_documents)
    context_documents = deduplicate_documents(documents + context_documents)
    return documents, context_documents


def build_ground_truth_lookup():
    collection_data = baseline_vectorstore.get(
        include=["documents", "metadatas"]
    )
    raw_documents = collection_data.get("documents", [])
    raw_metadatas = collection_data.get("metadatas", [])
    lookup = {}

    for text, metadata in zip(raw_documents, raw_metadatas):
        document = {"page_content": text, "metadata": metadata or {}}
        metadata = get_metadata(document)
        act = normalize_act(metadata.get("act"))
        section = normalize_section(metadata.get("section"))

        if not act or not section:
            continue

        key = (act, section)
        if key not in lookup:
            lookup[key] = document

    return lookup


def format_metric(value):
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def save_results(
    questions,
    results,
    metric_values,
    ground_truth_lookup,
    evaluator_failures,
    evaluator_retries,
    completed,
    summaries=None,
    structure_difference=None,
    model_comparison=None,
):
    current_summaries = {
        system_name: summarise_metrics(values)
        for system_name, values in metric_values.items()
    }

    if structure_difference is None:
        structure_difference = {}
        for metric in current_summaries["baseline"]:
            baseline_value = current_summaries["baseline"][metric]
            structure_value = current_summaries["structure_aware"][metric]
            structure_difference[metric] = (
                None
                if baseline_value is None or structure_value is None
                else structure_value - baseline_value
            )

    if summaries is not None:
        current_summaries = summaries

    if model_comparison is None:
        model_comparison = {
            "GPT-5-mini": current_summaries["structure_aware"],
            "Qwen3-8B": current_summaries["structure_aware_qwen3_8b"],
            "Llama-3.1-8B": current_summaries[
                "structure_aware_llama3_1_8b"
            ],
        }

    final_results = {
        "experiment": "Baseline RAG, Structure-Aware RAG, and LLM Comparison",
        "evaluation_method": (
            "Baseline-versus-structure-aware retrieval comparison and "
            "answer-generation model comparison using a shared "
            "structure-aware context for GPT-5-mini, Qwen3 8B, and "
            "Llama 3.1 8B."
        ),
        "completed": completed,
        "total_questions": len(questions),
        "completed_questions": len(results),
        "embedding_model": EMBEDDING_MODEL,
        "answer_model": ANSWER_MODEL,
        "evaluator_model": EVALUATOR_MODEL,
        "local_models": {
            "qwen3_8b": QWEN_MODEL,
            "llama3_1_8b": LLAMA_MODEL,
        },
        "experiments": {
            "baseline": "Standard semantic retrieval with GPT-5-mini",
            "structure_aware": (
                "Structure-aware retrieval with GPT-5-mini"
            ),
            "structure_aware_qwen3_8b": (
                "Structure-aware retrieval with Qwen3 8B"
            ),
            "structure_aware_llama3_1_8b": (
                "Structure-aware retrieval with Llama 3.1 8B"
            ),
        },
        "query_expansion": False,
        "query_expansion_note": (
            "No separate LLM-based query-expansion experiment. The "
            "structure retriever retains its existing legal-term matching "
            "and fuzzy keyword correction."
        ),
        "structure_retrieval": {
            "top_k_sections": TOP_K,
            "candidate_k": STRUCTURE_CANDIDATES,
            "retriever": "StructureAwareRetriever",
        },
        "reasoning_correct_threshold": 4,
        "precision_formula": (
            "Relevant documents among returned top-k documents divided "
            "by the number of documents actually returned."
        ),
        "recall_formula": (
            "1 if the target Act and section is retrieved within the "
            "top-k results; otherwise 0."
        ),
        "retrieval_definition": (
            "A retrieved document is relevant when its Act and section "
            "match the expected Act and section."
        ),
        "recall_definition": (
            "Each evaluation question has one target Act and section. "
            "Recall@k measures whether that target appears in the top-k "
            "results."
        ),
        "answer_correctness_definition": (
            "Score from 0 to 100 representing how accurately the generated "
            "answer answers the legal question."
        ),
        "groundedness_definition": (
            "Score from 0 to 100 representing how well the answer is "
            "supported by retrieved legislative context."
        ),
        "completeness_definition": (
            "Score from 0 to 100 representing how completely the answer "
            "provides the important information required."
        ),
        "citation_source_accuracy_definition": (
            "Score from 0 to 100 representing the accuracy of the cited "
            "Act and section, with subsection considered when applicable."
        ),
        "reasoning_accuracy_definition": (
            "Proportion of answers that receive a reasoning score of at "
            "least 4, answer the question, and retrieve the target legal "
            "section."
        ),
        "hallucination_formula": (
            "Unsupported substantive claims divided by total substantive "
            "claims. Questions with zero claims or unparseable evaluator "
            "results are excluded."
        ),
        "ground_truth_provisions_available": len(ground_truth_lookup),
        "evaluator_parse_failures": evaluator_failures,
        "evaluator_retries": evaluator_retries,
        "summary": {
            "baseline": current_summaries["baseline"],
            "structure_aware": current_summaries["structure_aware"],
            "difference_structure_minus_baseline": structure_difference,
            "model_comparison": model_comparison,
        },
        "results": results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(final_results, file, indent=2, ensure_ascii=False)


def run_experiment():
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation questions not found: {QUESTIONS_FILE}"
        )

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    questions = dataset["questions"]

    print("=" * 80)
    print("FINAL EXPERIMENT EVALUATION")
    print("=" * 80)
    print(f"\nQuestions: {len(questions)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"GPT answer model: {ANSWER_MODEL}")
    print(f"Evaluator model: {EVALUATOR_MODEL}")
    print("\nExperiments:")
    print("  1. Baseline RAG + GPT-5-mini")
    print("  2. Structure-aware RAG + GPT-5-mini")
    print("  3. Structure-aware RAG + Qwen3 8B")
    print("  4. Structure-aware RAG + Llama 3.1 8B")
    print("\nStructure-aware retrieval:")
    print(f"  Top sections: {TOP_K}")
    print(f"  Candidate chunks: {STRUCTURE_CANDIDATES}")

    print("\nBuilding ground-truth lookup...")
    ground_truth_lookup = build_ground_truth_lookup()
    print(
        f"Ground-truth provisions found: {len(ground_truth_lookup)}"
    )

    system_names = [
        "baseline",
        "structure_aware",
        "structure_aware_qwen3_8b",
        "structure_aware_llama3_1_8b",
    ]

    metric_values = {
        name: new_metric_store() for name in system_names
    }
    evaluator_failures = {name: 0 for name in system_names}
    evaluator_retries = {name: 0 for name in system_names}
    results = []

    # Restore saved results and metrics so completed questions are not rerun.
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
            saved_data = json.load(file)

        if saved_data.get("completed"):
            raise RuntimeError(
                "The saved experiment is already marked complete; "
                "refusing to overwrite it."
            )

        saved_results = saved_data.get("results", [])

        if len(saved_results) > len(questions):
            raise RuntimeError("Saved results exceed the dataset size.")

        # Confirm saved results match the dataset's initial questions.
        for saved, item in zip(saved_results, questions):
            if saved.get("question_number") != item["id"]:
                raise RuntimeError(
                    "Saved question order does not match the dataset. "
                    "No experiment was started."
                )

            if not all(name in saved for name in system_names):
                raise RuntimeError(
                    f"Incomplete saved result for Q{item['id']}. "
                    "No experiment was started."
                )

        results = saved_results

        for saved in results:
            for system_name in system_names:
                system_result = saved[system_name]
                evaluation = system_result["qa_evaluation"]

                add_metric_values(
                    metric_values[system_name],
                    system_result["retrieval"],
                    evaluation,
                )

                if evaluation.get("evaluator_parse_failed"):
                    evaluator_failures[system_name] += 1

                if evaluation.get("evaluator_retry_used"):
                    evaluator_retries[system_name] += 1

        print(f"Restored {len(results)} completed questions.")

        if results:
            print(
                f"Resuming from question {len(results) + 1} "
                f"of {len(questions)}."
            )

    for index, item in enumerate(questions, start=1):
        if index <= len(results):
            continue

        question_id = item["id"]
        question = item["question"]
        expected_act = item["act"]
        expected_section = item["section"]

        print(f"\n[{index}/{len(questions)}] Q{question_id}: {question}")

        ground_truth_key = (
            normalize_act(expected_act),
            normalize_section(expected_section),
        )
        ground_truth_document = ground_truth_lookup.get(ground_truth_key)

        question_result = {
            "question_number": question_id,
            "question": question,
            "expected_act": expected_act,
            "expected_section": expected_section,
        }

        baseline_documents = retrieve_baseline(question)
        baseline_context = baseline_documents

        structure_documents, structure_context = retrieve_structure(
            question, structure_retriever
        )

        print("  Running baseline RAG + GPT-5-mini...")
        baseline_answer = generate_answer(question, baseline_context)

        baseline_result = evaluate_one_answer(
            question,
            expected_act,
            expected_section,
            ground_truth_document,
            baseline_documents,
            baseline_context,
            baseline_answer,
        )

        question_result["baseline"] = baseline_result

        add_metric_values(
            metric_values["baseline"],
            baseline_result["retrieval"],
            baseline_result["qa_evaluation"],
        )

        print("  Running structure-aware RAG + GPT-5-mini...")
        structure_answer = generate_answer(question, structure_context)

        structure_result = evaluate_one_answer(
            question,
            expected_act,
            expected_section,
            ground_truth_document,
            structure_documents,
            structure_context,
            structure_answer,
        )

        question_result["structure_aware"] = structure_result

        add_metric_values(
            metric_values["structure_aware"],
            structure_result["retrieval"],
            structure_result["qa_evaluation"],
        )

        for system_name, model_name in LOCAL_MODELS.items():
            result_key = f"structure_aware_{system_name}"

            print(
                f"  Running structure-aware RAG + {model_name}..."
            )

            try:
                local_answer = generate_local_answer(
                    model_name, question, structure_context
                )

            except Timeout as exc:
                # Record the failure explicitly. Do not treat this as
                # a successful model generation.
                print(
                    f"  WARNING: {model_name} timed out: {exc}",
                    flush=True,
                )

                question_result[result_key] = {
                    "generation_failed": True,
                    "generation_error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "retrieval": {
                        "precision_at_1": None,
                        "precision_at_5": None,
                        "recall_at_1": None,
                        "recall_at_5": None,
                    },
                    "qa_evaluation": {
                        "evaluator_parse_failed": True,
                        "evaluator_retry_used": False,
                        "generation_failed": True,
                    },
                }

                # Do not add fabricated evaluation metrics for this model.
                # Continue to the other local model.
                continue

            local_result = evaluate_one_answer(
                question,
                expected_act,
                expected_section,
                ground_truth_document,
                structure_documents,
                structure_context,
                local_answer,
            )

            local_result["generation_failed"] = False
            local_result["generation_error"] = None

            question_result[result_key] = local_result

            add_metric_values(
                metric_values[result_key],
                local_result["retrieval"],
                local_result["qa_evaluation"],
            )

        for system_name in system_names:
            system_result = question_result[system_name]
            evaluation = system_result["qa_evaluation"]

            if evaluation.get("evaluator_parse_failed"):
                evaluator_failures[system_name] += 1

            if evaluation.get("evaluator_retry_used"):
                evaluator_retries[system_name] += 1

        results.append(question_result)

        print(f"  Completed question {question_id}.")

        save_results(
            questions,
            results,
            metric_values,
            ground_truth_lookup,
            evaluator_failures,
            evaluator_retries,
            completed=False,
        )

    summaries = {
        name: summarise_metrics(metric_values[name])
        for name in system_names
    }

    structure_difference = {}

    for metric in summaries["baseline"]:
        baseline_value = summaries["baseline"][metric]
        structure_value = summaries["structure_aware"][metric]

        structure_difference[metric] = (
            None
            if baseline_value is None or structure_value is None
            else structure_value - baseline_value
        )

    model_comparison = {
        "GPT-5-mini": summaries["structure_aware"],
        "Qwen3-8B": summaries["structure_aware_qwen3_8b"],
        "Llama-3.1-8B": summaries["structure_aware_llama3_1_8b"],
    }

    save_results(
        questions,
        results,
        metric_values,
        ground_truth_lookup,
        evaluator_failures,
        evaluator_retries,
        completed=True,
        summaries=summaries,
        structure_difference=structure_difference,
        model_comparison=model_comparison,
    )

    print("\n" + "=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)

    print("\nRetrieval Metrics")
    print(f"{'Metric':30s}{'Baseline':15s}{'Structure':15s}")
    print("-" * 65)

    for metric in (
        "precision_at_1",
        "precision_at_5",
        "recall_at_1",
        "recall_at_5",
    ):
        print(
            f"{metric:30s}"
            f"{format_metric(summaries['baseline'][metric]):15s}"
            f"{format_metric(summaries['structure_aware'][metric]):15s}"
        )

    print("\nAnswer Evaluation Metrics")
    print(f"{'Metric':30s}{'Baseline':15s}{'Structure':15s}")
    print("-" * 65)

    for metric in (
        "answer_correctness",
        "groundedness",
        "completeness",
        "citation_source_accuracy",
    ):
        print(
            f"{metric:30s}"
            f"{format_metric(summaries['baseline'][metric]):15s}"
            f"{format_metric(summaries['structure_aware'][metric]):15s}"
        )

    print("\nLanguage Model Comparison")
    print("All three models use the same structure-aware retrieval results.")

    for metric in (
        "answer_correctness",
        "groundedness",
        "completeness",
        "citation_source_accuracy",
        "reasoning_accuracy",
        "hallucination_rate",
    ):
        print(f"\n{metric}")

        for model_name, model_summary in model_comparison.items():
            print(
                f"  {model_name:20s}: "
                f"{format_metric(model_summary[metric])}"
            )

    print("\nDifference: Structure-aware GPT - Baseline GPT")

    for metric, value in structure_difference.items():
        print(f"  {metric:35s}{format_metric(value)}")

    print("\nEvaluator parse failures:")

    for name, failures in evaluator_failures.items():
        print(f"  {name}: {failures}")

    print("\nEvaluator retries:")

    for name, retries in evaluator_retries.items():
        print(f"  {name}: {retries}")

    print("\nResults saved to:")
    print(OUTPUT_FILE)
    print("\nExperiment complete.")


if __name__ == "__main__":
    run_experiment()
