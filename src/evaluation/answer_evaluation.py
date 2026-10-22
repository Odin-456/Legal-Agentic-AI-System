"""Legal answer evaluation, JSON parsing, retrieval metrics, and metric aggregation."""
import json
import re

from experiment_config import (
    REASONING_CORRECT_THRESHOLD,
    evaluator_llm,
)
from answer_generation import build_context, get_metadata, get_page_content


def normalize_act(act):
    if act is None:
        return ""
    act = str(act).strip().lower()
    aliases = {
        "real property act": "real property act 1900",
        "real property act 1900": "real property act 1900",
        "conveyancing act": "conveyancing act 1919",
        "conveyancing act 1919": "conveyancing act 1919",
        "residential tenancies act": "residential tenancies act 2010",
        "residential tenancies act 2010": "residential tenancies act 2010",
        "strata schemes management act": "strata schemes management act 2015",
        "strata schemes management act 2015": "strata schemes management act 2015",
        "community land management act": "community land management act 2021",
        "community land management act 2021": "community land management act 2021",
    }
    return aliases.get(act, act)


def normalize_section(section):
    if section is None:
        return ""
    return re.sub(r"\s+", "", str(section).strip()).lower()


def document_is_relevant(document, expected_act, expected_section):
    metadata = get_metadata(document)
    act = normalize_act(metadata.get("act"))
    section = normalize_section(metadata.get("section"))
    return act == normalize_act(expected_act) and section == normalize_section(expected_section)


def extract_json_object(raw_text):
    if not raw_text:
        return None
    text = str(raw_text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:index + 1])
                except json.JSONDecodeError:
                    return None
    return None


def build_evaluator_prompt(question, expected_act, expected_section, ground_truth,
                           retrieved_context, answer, target_retrieved):
    return f"""
You are evaluating a NSW property-law question-answering system.

Evaluate the answer against the ground-truth legal provision
and the retrieved legislative context.

QUESTION:
{question}

EXPECTED ACT:
{expected_act}

EXPECTED SECTION:
{expected_section}

GROUND-TRUTH LEGAL PROVISION:
{ground_truth}

RETRIEVED CONTEXT:
{retrieved_context}

GENERATED ANSWER:
{answer}

TARGET SECTION RETRIEVED:
{target_retrieved}

Score the following metrics from 0 to 100.

1. answer_correctness:
How correctly does the answer answer the question?

2. groundedness:
How well are the answer's substantive claims supported
by the retrieved legal context?

3. completeness:
How completely does the answer include the important
information required by the question?

4. citation_source_accuracy:
How accurately does the answer identify the relevant Act
and section, and subsection when applicable?

The evaluation dataset does not provide subsection-level
ground truth for every question. Only evaluate subsection
accuracy when it can be determined from the supplied information.

Also evaluate:

reasoning_score:
5 = Completely correct and directly answers the question.
4 = Mostly correct with only minor omissions.
3 = Partially correct but missing important information.
2 = Mostly incorrect or materially incomplete.
1 = Incorrect or does not answer the question.

answers_question:
True only if the answer actually answers the question.

reasoning_correct:
True only when reasoning_score is at least 4,
answers_question is true, and the target section was retrieved.

total_claims:
Count substantive claims in the answer.

unsupported_claims:
Count substantive claims that cannot be supported by
the retrieved legislative context.

A refusal saying "the context is insufficient" is not correct
if the ground-truth provision contains enough information
to answer the question.

If the target section was not retrieved,
reasoning_correct MUST be false.

Return valid JSON only:

{{
    "answer_correctness": 0,
    "groundedness": 0,
    "completeness": 0,
    "citation_source_accuracy": 0,
    "reasoning_score": 1,
    "answers_question": false,
    "reasoning_correct": false,
    "total_claims": 0,
    "unsupported_claims": 0,
    "explanation": "brief explanation"
}}
"""


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return False


def safe_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def evaluate_answer(question, expected_act, expected_section, answer,
                    retrieved_documents, ground_truth_document):
    target_retrieved = any(
        document_is_relevant(document, expected_act, expected_section)
        for document in retrieved_documents
    )
    ground_truth = (
        get_page_content(ground_truth_document)
        if ground_truth_document is not None else ""
    )
    retrieved_context = build_context(retrieved_documents)
    prompt = build_evaluator_prompt(
        question=question,
        expected_act=expected_act,
        expected_section=expected_section,
        ground_truth=ground_truth,
        retrieved_context=retrieved_context,
        answer=answer,
        target_retrieved=target_retrieved,
    )
    evaluator_retry_used = False
    evaluator_parse_failed = False
    response = evaluator_llm.invoke(prompt)
    raw_response = response.content
    if isinstance(raw_response, list):
        raw_response = "".join(
            item.get("text", "") for item in raw_response if isinstance(item, dict)
        )
    raw_response = str(raw_response).strip()
    evaluation = extract_json_object(raw_response)

    if evaluation is None:
        evaluator_retry_used = True
        retry_prompt = prompt + """

IMPORTANT: Your previous response could not be parsed.
Return ONLY the requested valid JSON object.
Do not include Markdown fences or additional commentary.
"""
        retry_response = evaluator_llm.invoke(retry_prompt)
        retry_raw = retry_response.content
        if isinstance(retry_raw, list):
            retry_raw = "".join(
                item.get("text", "") for item in retry_raw if isinstance(item, dict)
            )
        raw_response = str(retry_raw).strip()
        evaluation = extract_json_object(raw_response)

    if evaluation is None:
        evaluator_parse_failed = True
        answer_correctness = groundedness = completeness = None
        citation_source_accuracy = reasoning_score = None
        answers_question = reasoning_correct = None
        total_claims = unsupported_claims = None
        explanation = "Evaluator response could not be parsed after retry."
        hallucination_rate = None
    else:
        answer_correctness = safe_int(evaluation.get("answer_correctness"))
        groundedness = safe_int(evaluation.get("groundedness"))
        completeness = safe_int(evaluation.get("completeness"))
        citation_source_accuracy = safe_int(evaluation.get("citation_source_accuracy"))
        reasoning_score = safe_int(evaluation.get("reasoning_score"))
        total_claims = safe_int(evaluation.get("total_claims"))
        unsupported_claims = safe_int(evaluation.get("unsupported_claims"))
        answers_question = parse_bool(evaluation.get("answers_question"))
        reasoning_correct = parse_bool(evaluation.get("reasoning_correct"))
        explanation = str(evaluation.get("explanation", ""))

        score_fields = [answer_correctness, groundedness, completeness, citation_source_accuracy]
        score_fields = [clamp(value, 0, 100) if value is not None else None for value in score_fields]
        answer_correctness, groundedness, completeness, citation_source_accuracy = score_fields

        if reasoning_score is not None:
            reasoning_score = clamp(reasoning_score, 1, 5)
        if total_claims is not None:
            total_claims = max(0, total_claims)
        if unsupported_claims is not None:
            unsupported_claims = max(0, unsupported_claims)
        if total_claims is not None and unsupported_claims is not None:
            unsupported_claims = min(unsupported_claims, total_claims)

        if (reasoning_score is None or reasoning_score < REASONING_CORRECT_THRESHOLD
                or not answers_question or not target_retrieved):
            reasoning_correct = False

        if total_claims is not None and total_claims > 0 and unsupported_claims is not None:
            hallucination_rate = unsupported_claims / total_claims
        else:
            hallucination_rate = None

    return {
        "answer_correctness": answer_correctness,
        "groundedness": groundedness,
        "completeness": completeness,
        "citation_source_accuracy": citation_source_accuracy,
        "reasoning_score": reasoning_score,
        "answers_question": answers_question,
        "reasoning_correct": reasoning_correct,
        "total_claims": total_claims,
        "unsupported_claims": unsupported_claims,
        "hallucination_rate": hallucination_rate,
        "target_section_retrieved": target_retrieved,
        "evaluator_parse_failed": evaluator_parse_failed,
        "evaluator_retry_used": evaluator_retry_used,
        "explanation": explanation,
        "raw_evaluator_response": raw_response,
    }


def precision_at_k(documents, expected_act, expected_section, k):
    top_documents = documents[:k]
    if not top_documents:
        return 0.0
    relevant = sum(
        1 for document in top_documents
        if document_is_relevant(document, expected_act, expected_section)
    )
    return relevant / len(top_documents)


def recall_at_k(documents, expected_act, expected_section, k):
    for document in documents[:k]:
        if document_is_relevant(document, expected_act, expected_section):
            return 1.0
    return 0.0


def average(values):
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return None
    return sum(valid_values) / len(valid_values)


def new_metric_store():
    return {
        "precision_at_1": [], "precision_at_5": [],
        "recall_at_1": [], "recall_at_5": [],
        "answer_correctness": [], "groundedness": [],
        "completeness": [], "citation_source_accuracy": [],
        "reasoning_accuracy": [], "hallucination_rate": [],
    }


def add_metric_values(metric_values, retrieval, evaluation):
    for metric in ("precision_at_1", "precision_at_5", "recall_at_1", "recall_at_5"):
        metric_values[metric].append(retrieval[metric])
    for metric in ("answer_correctness", "groundedness", "completeness", "citation_source_accuracy"):
        value = evaluation.get(metric)
        if value is not None:
            metric_values[metric].append(value)
    reasoning_correct = evaluation.get("reasoning_correct")
    if reasoning_correct is not None:
        metric_values["reasoning_accuracy"].append(int(reasoning_correct))
    hallucination_rate = evaluation.get("hallucination_rate")
    if hallucination_rate is not None:
        metric_values["hallucination_rate"].append(hallucination_rate)


def summarise_metrics(metric_values):
    return {metric: average(values) for metric, values in metric_values.items()}


def evaluate_one_answer(question, expected_act, expected_section, ground_truth_document,
                        documents_for_metrics, context_documents, answer):
    p1 = precision_at_k(documents_for_metrics, expected_act, expected_section, 1)
    p5 = precision_at_k(documents_for_metrics, expected_act, expected_section, 5)
    r1 = recall_at_k(documents_for_metrics, expected_act, expected_section, 1)
    r5 = recall_at_k(documents_for_metrics, expected_act, expected_section, 5)
    retrieval_metrics = {
        "precision_at_1": p1,
        "precision_at_5": p5,
        "recall_at_1": r1,
        "recall_at_5": r5,
        "hit_at_5": r5 == 1.0,
    }
    evaluation = evaluate_answer(
        question=question,
        expected_act=expected_act,
        expected_section=expected_section,
        answer=answer,
        retrieved_documents=context_documents,
        ground_truth_document=ground_truth_document,
    )
    return {"retrieval": retrieval_metrics, "answer": answer, "qa_evaluation": evaluation}
