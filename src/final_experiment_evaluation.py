import json
import re
import requests
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ============================================================
# PROJECT PATH
# ============================================================

SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from structure_retrieval import StructureAwareRetriever


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

QUESTIONS_FILE = Path(
    "data/evaluation/qa_evaluation_questions.json"
)

OUTPUT_FILE = Path(
    "data/evaluation/final_experiment_results.json"
)

BASELINE_VECTORSTORE = Path(
    "data/vectorstore/property_law_baseline"
)

EMBEDDING_MODEL = "text-embedding-3-large"

ANSWER_MODEL = "gpt-5-mini"
EVALUATOR_MODEL = "gpt-5-mini"

QWEN_MODEL = "qwen3:8b"
LLAMA_MODEL = "llama3.1:8b"

OLLAMA_URL = "http://localhost:11434/api/chat"

TOP_K = 5
STRUCTURE_CANDIDATES = 50

REASONING_CORRECT_THRESHOLD = 4

OLLAMA_TIMEOUT = 300

LOCAL_MODELS = {
    "qwen3_8b": QWEN_MODEL,
    "llama3_1_8b": LLAMA_MODEL,
}


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a NSW property law question-answering system.

Answer the user's question using ONLY the NSW property-law
legislation provided in the context.

Rules:
- Use only the supplied legislative context.
- Do not use outside legal knowledge.
- Do not invent Acts, sections, subsections, or legal rules.
- Identify the relevant Act and section when possible.
- If the supplied context is insufficient, clearly say so.
- Do not provide personalised legal advice.
- Answer concisely and accurately.
"""


# ============================================================
# MODELS
# ============================================================

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL
)

# Used for baseline and structure-aware answer generation.
answer_llm = ChatOpenAI(
    model=ANSWER_MODEL,
    temperature=0,
)

# Used consistently to evaluate answers from every model.
evaluator_llm = ChatOpenAI(
    model=EVALUATOR_MODEL,
    temperature=0,
)


# ============================================================
# BASELINE VECTOR STORE
# ============================================================

if not BASELINE_VECTORSTORE.exists():
    raise FileNotFoundError(
        f"Baseline vectorstore not found: "
        f"{BASELINE_VECTORSTORE}"
    )

baseline_vectorstore = Chroma(
    collection_name="property_law_baseline",
    persist_directory=str(BASELINE_VECTORSTORE),
    embedding_function=embeddings,
)


# ============================================================
# STRUCTURE-AWARE RETRIEVER
# ============================================================

# Query expansion is not used as a separate experiment.
# This constructor matches structure_retrieval.py.

structure_retriever = StructureAwareRetriever(
    candidate_k=STRUCTURE_CANDIDATES
)


# ============================================================
# NORMALISATION
# ============================================================

def normalize_act(act):

    if act is None:
        return ""

    act = str(act).strip().lower()

    aliases = {
        "real property act":
            "real property act 1900",

        "real property act 1900":
            "real property act 1900",

        "conveyancing act":
            "conveyancing act 1919",

        "conveyancing act 1919":
            "conveyancing act 1919",

        "residential tenancies act":
            "residential tenancies act 2010",

        "residential tenancies act 2010":
            "residential tenancies act 2010",

        "strata schemes management act":
            "strata schemes management act 2015",

        "strata schemes management act 2015":
            "strata schemes management act 2015",

        "community land management act":
            "community land management act 2021",

        "community land management act 2021":
            "community land management act 2021",
    }

    return aliases.get(act, act)


def normalize_section(section):

    if section is None:
        return ""

    section = str(section).strip()

    section = re.sub(
        r"\s+",
        "",
        section,
    )

    return section.lower()


# ============================================================
# DOCUMENT HELPERS
# ============================================================

def get_metadata(document):

    if hasattr(document, "metadata"):
        return document.metadata or {}

    if isinstance(document, dict):
        return document.get("metadata", document) or {}

    return {}


def get_page_content(document):

    if hasattr(document, "page_content"):
        return document.page_content or ""

    if isinstance(document, dict):
        return document.get(
            "page_content",
            document.get("text", ""),
        ) or ""

    return str(document)


def document_is_relevant(
    document,
    expected_act,
    expected_section,
):

    metadata = get_metadata(document)

    act = normalize_act(
        metadata.get("act")
    )

    section = normalize_section(
        metadata.get("section")
    )

    return (
        act == normalize_act(expected_act)
        and section == normalize_section(expected_section)
    )


def deduplicate_documents(documents):

    unique_documents = []
    seen = set()

    for document in documents:

        metadata = get_metadata(document)

        # Use document content and identifying metadata to avoid
        # adding the same chunk to the context multiple times.
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


# ============================================================
# BASELINE RETRIEVAL
# ============================================================

def retrieve_baseline(question):

    return baseline_vectorstore.similarity_search(
        question,
        k=TOP_K,
    )


# ============================================================
# STRUCTURE-AWARE RETRIEVAL
# ============================================================

def retrieve_structure(
    question,
    retriever,
):

    results = retriever.retrieve(
        question,
        top_k=TOP_K,
    )

    documents = []
    context_documents = []

    for result in results:

        document = result.get("document")

        if document is not None:
            documents.append(document)

        related_documents = result.get(
            "related_documents",
            [],
        )

        context_documents.extend(related_documents)

    documents = deduplicate_documents(documents)

    context_documents = deduplicate_documents(
        context_documents
    )

    # Ensure the representative retrieved sections are available
    # in the generated answer's context as well.
    context_documents = deduplicate_documents(
        documents + context_documents
    )

    return documents, context_documents


# ============================================================
# CONTEXT
# ============================================================

def build_context(documents):

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        metadata = get_metadata(document)

        act = metadata.get("act", "")
        section = metadata.get("section", "")
        heading = metadata.get("heading", "")
        subsection = metadata.get("subsection", "")
        paragraph = metadata.get("paragraph", "")
        nested_paragraph = metadata.get(
            "nested_paragraph",
            "",
        )

        text = get_page_content(document)

        context_parts.append(
            f"""
[SOURCE {index}]
Act: {act}
Section: {section}
Heading: {heading}
Subsection: {subsection}
Paragraph: {paragraph}
Nested paragraph: {nested_paragraph}

Legal text:
{text}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# ANSWER GENERATION: GPT-5-MINI
# ============================================================

def generate_answer(
    question,
    documents,
):

    context = build_context(documents)

    prompt = f"""
{SYSTEM_PROMPT}

QUESTION:
{question}

RETRIEVED LEGAL CONTEXT:
{context}

Provide a concise and accurate answer.
"""

    response = answer_llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        content = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        )

    return str(content).strip()


# ============================================================
# ANSWER GENERATION: LOCAL OLLAMA MODELS
# ============================================================

def generate_local_answer(
    model_name,
    question,
    documents,
):

    context = build_context(documents)

    user_prompt = f"""
Use the following NSW property-law legislative context
to answer the question.

CONTEXT:
{context}

QUESTION:
{question}

Answer concisely and accurately using only the context.
"""

    payload = {
        "model": model_name,

        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],

        "stream": False,

        "think": False,

        "options": {
            "temperature": 0,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    message = data.get("message", {})

    content = message.get("content", "")

    if not content:
        raise ValueError(
            f"Ollama returned an empty answer for {model_name}."
        )

    return content.strip()


# ============================================================
# ROBUST JSON PARSER
# ============================================================

def extract_json_object(raw_text):

    if not raw_text:
        return None

    text = str(raw_text).strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    text = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.replace(
        "```",
        "",
    ).strip()

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

                candidate = text[start:index + 1]

                try:
                    return json.loads(candidate)

                except json.JSONDecodeError:
                    return None

    return None


# ============================================================
# EVALUATOR PROMPT
# ============================================================

def build_evaluator_prompt(
    question,
    expected_act,
    expected_section,
    ground_truth,
    retrieved_context,
    answer,
    target_retrieved,
):

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


# ============================================================
# EVALUATION HELPERS
# ============================================================

def parse_bool(value):

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value != 0

    if isinstance(value, str):

        return value.strip().lower() in {
            "true",
            "yes",
            "1",
        }

    return False


def safe_int(value, default=None):

    try:
        return int(value)

    except (TypeError, ValueError):
        return default


def clamp(value, minimum, maximum):

    return max(
        minimum,
        min(maximum, value),
    )


# ============================================================
# ANSWER EVALUATION
# ============================================================

def evaluate_answer(
    question,
    expected_act,
    expected_section,
    answer,
    retrieved_documents,
    ground_truth_document,
):

    target_retrieved = any(
        document_is_relevant(
            document,
            expected_act,
            expected_section,
        )
        for document in retrieved_documents
    )

    ground_truth = ""

    if ground_truth_document is not None:
        ground_truth = get_page_content(
            ground_truth_document
        )

    retrieved_context = build_context(
        retrieved_documents
    )

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
            item.get("text", "")
            for item in raw_response
            if isinstance(item, dict)
        )

    raw_response = str(raw_response).strip()

    evaluation = extract_json_object(
        raw_response
    )

    # Retry once if the evaluator did not return valid JSON.
    if evaluation is None:

        evaluator_retry_used = True

        retry_prompt = prompt + """

IMPORTANT: Your previous response could not be parsed.
Return ONLY the requested valid JSON object.
Do not include Markdown fences or additional commentary.
"""

        retry_response = evaluator_llm.invoke(
            retry_prompt
        )

        retry_raw = retry_response.content

        if isinstance(retry_raw, list):
            retry_raw = "".join(
                item.get("text", "")
                for item in retry_raw
                if isinstance(item, dict)
            )

        raw_response = str(retry_raw).strip()

        evaluation = extract_json_object(
            raw_response
        )

    if evaluation is None:

        evaluator_parse_failed = True

        answer_correctness = None
        groundedness = None
        completeness = None
        citation_source_accuracy = None
        reasoning_score = None
        answers_question = None
        reasoning_correct = None
        total_claims = None
        unsupported_claims = None

        explanation = (
            "Evaluator response could not be parsed after retry."
        )

        hallucination_rate = None

    else:

        answer_correctness = safe_int(
            evaluation.get("answer_correctness")
        )

        groundedness = safe_int(
            evaluation.get("groundedness")
        )

        completeness = safe_int(
            evaluation.get("completeness")
        )

        citation_source_accuracy = safe_int(
            evaluation.get("citation_source_accuracy")
        )

        reasoning_score = safe_int(
            evaluation.get("reasoning_score")
        )

        total_claims = safe_int(
            evaluation.get("total_claims")
        )

        unsupported_claims = safe_int(
            evaluation.get("unsupported_claims")
        )

        answers_question = parse_bool(
            evaluation.get("answers_question")
        )

        reasoning_correct = parse_bool(
            evaluation.get("reasoning_correct")
        )

        explanation = str(
            evaluation.get("explanation", "")
        )

        score_fields = [
            answer_correctness,
            groundedness,
            completeness,
            citation_source_accuracy,
        ]

        score_fields = [
            clamp(value, 0, 100)
            if value is not None
            else None
            for value in score_fields
        ]

        (
            answer_correctness,
            groundedness,
            completeness,
            citation_source_accuracy,
        ) = score_fields

        if reasoning_score is not None:
            reasoning_score = clamp(
                reasoning_score,
                1,
                5,
            )

        if total_claims is not None:
            total_claims = max(0, total_claims)

        if unsupported_claims is not None:
            unsupported_claims = max(
                0,
                unsupported_claims,
            )

        if (
            total_claims is not None
            and unsupported_claims is not None
        ):
            unsupported_claims = min(
                unsupported_claims,
                total_claims,
            )

        # Apply the same reasoning rule to every model.
        if (
            reasoning_score is None
            or reasoning_score < REASONING_CORRECT_THRESHOLD
            or not answers_question
            or not target_retrieved
        ):
            reasoning_correct = False

        if (
            total_claims is not None
            and total_claims > 0
            and unsupported_claims is not None
        ):
            hallucination_rate = (
                unsupported_claims / total_claims
            )

        else:
            hallucination_rate = None

    return {
        "answer_correctness":
            answer_correctness,

        "groundedness":
            groundedness,

        "completeness":
            completeness,

        "citation_source_accuracy":
            citation_source_accuracy,

        "reasoning_score":
            reasoning_score,

        "answers_question":
            answers_question,

        "reasoning_correct":
            reasoning_correct,

        "total_claims":
            total_claims,

        "unsupported_claims":
            unsupported_claims,

        "hallucination_rate":
            hallucination_rate,

        "target_section_retrieved":
            target_retrieved,

        "evaluator_parse_failed":
            evaluator_parse_failed,

        "evaluator_retry_used":
            evaluator_retry_used,

        "explanation":
            explanation,

        "raw_evaluator_response":
            raw_response,
    }


# ============================================================
# GROUND TRUTH
# ============================================================

def build_ground_truth_lookup():

    collection_data = baseline_vectorstore.get(
        include=[
            "documents",
            "metadatas",
        ]
    )

    raw_documents = collection_data.get(
        "documents",
        [],
    )

    raw_metadatas = collection_data.get(
        "metadatas",
        [],
    )

    lookup = {}

    for text, metadata in zip(
        raw_documents,
        raw_metadatas,
    ):

        document = {
            "page_content": text,
            "metadata": metadata or {},
        }

        metadata = get_metadata(document)

        act = normalize_act(
            metadata.get("act")
        )

        section = normalize_section(
            metadata.get("section")
        )

        if not act or not section:
            continue

        key = (
            act,
            section,
        )

        if key not in lookup:
            lookup[key] = document

    return lookup


# ============================================================
# RETRIEVAL METRICS
# ============================================================

def precision_at_k(
    documents,
    expected_act,
    expected_section,
    k,
):

    top_documents = documents[:k]

    if not top_documents:
        return 0.0

    relevant = sum(
        1
        for document in top_documents
        if document_is_relevant(
            document,
            expected_act,
            expected_section,
        )
    )

    # Preserves your original convention:
    # divide by the number of documents actually returned.
    return relevant / len(top_documents)


def recall_at_k(
    documents,
    expected_act,
    expected_section,
    k,
):

    top_documents = documents[:k]

    for document in top_documents:

        if document_is_relevant(
            document,
            expected_act,
            expected_section,
        ):
            return 1.0

    # Each evaluation question has one target Act and section.
    return 0.0


def average(values):

    valid_values = [
        value
        for value in values
        if value is not None
    ]

    if not valid_values:
        return None

    return sum(valid_values) / len(valid_values)


# ============================================================
# METRIC STORAGE
# ============================================================

def new_metric_store():

    return {
        "precision_at_1": [],
        "precision_at_5": [],
        "recall_at_1": [],
        "recall_at_5": [],
        "answer_correctness": [],
        "groundedness": [],
        "completeness": [],
        "citation_source_accuracy": [],
        "reasoning_accuracy": [],
        "hallucination_rate": [],
    }


def add_metric_values(
    metric_values,
    retrieval,
    evaluation,
):

    metric_values["precision_at_1"].append(
        retrieval["precision_at_1"]
    )

    metric_values["precision_at_5"].append(
        retrieval["precision_at_5"]
    )

    metric_values["recall_at_1"].append(
        retrieval["recall_at_1"]
    )

    metric_values["recall_at_5"].append(
        retrieval["recall_at_5"]
    )

    for metric in (
        "answer_correctness",
        "groundedness",
        "completeness",
        "citation_source_accuracy",
    ):

        value = evaluation.get(metric)

        if value is not None:
            metric_values[metric].append(value)

    reasoning_correct = evaluation.get(
        "reasoning_correct"
    )

    if reasoning_correct is not None:
        metric_values["reasoning_accuracy"].append(
            int(reasoning_correct)
        )

    hallucination_rate = evaluation.get(
        "hallucination_rate"
    )

    if hallucination_rate is not None:
        metric_values["hallucination_rate"].append(
            hallucination_rate
        )


def summarise_metrics(metric_values):

    return {
        metric: average(values)
        for metric, values in metric_values.items()
    }


# ============================================================
# EVALUATE ONE ANSWER
# ============================================================

def evaluate_one_answer(
    question,
    expected_act,
    expected_section,
    ground_truth_document,
    documents_for_metrics,
    context_documents,
    answer,
):

    p1 = precision_at_k(
        documents_for_metrics,
        expected_act,
        expected_section,
        1,
    )

    p5 = precision_at_k(
        documents_for_metrics,
        expected_act,
        expected_section,
        5,
    )

    r1 = recall_at_k(
        documents_for_metrics,
        expected_act,
        expected_section,
        1,
    )

    r5 = recall_at_k(
        documents_for_metrics,
        expected_act,
        expected_section,
        5,
    )

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

    return {
        "retrieval": retrieval_metrics,
        "answer": answer,
        "qa_evaluation": evaluation,
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():

    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation questions not found: {QUESTIONS_FILE}"
        )

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
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
        f"Ground-truth provisions found: "
        f"{len(ground_truth_lookup)}"
    )

    system_names = [
        "baseline",
        "structure_aware",
        "structure_aware_qwen3_8b",
        "structure_aware_llama3_1_8b",
    ]

    metric_values = {
        system_name: new_metric_store()
        for system_name in system_names
    }

    evaluator_failures = {
        system_name: 0
        for system_name in system_names
    }

    evaluator_retries = {
        system_name: 0
        for system_name in system_names
    }

    results = []

    # ========================================================
    # PROCESS EACH QUESTION
    # ========================================================

    for index, item in enumerate(
        questions,
        start=1,
    ):

        question_id = item["id"]
        question = item["question"]
        expected_act = item["act"]
        expected_section = item["section"]

        print(
            f"\n[{index}/{len(questions)}] "
            f"Q{question_id}: {question}"
        )

        ground_truth_key = (
            normalize_act(expected_act),
            normalize_section(expected_section),
        )

        ground_truth_document = ground_truth_lookup.get(
            ground_truth_key
        )

        question_result = {
            "question_number": question_id,
            "question": question,
            "expected_act": expected_act,
            "expected_section": expected_section,
        }

        # ----------------------------------------------------
        # RETRIEVE ONCE PER SYSTEM
        # ----------------------------------------------------

        # Baseline retrieval happens once. All baseline metrics
        # and the GPT answer use this same retrieval result.
        baseline_documents = retrieve_baseline(question)

        baseline_context = baseline_documents

        # Structure-aware retrieval happens once. GPT, Qwen,
        # and Llama all receive the same structure-aware context.
        (
            structure_documents,
            structure_context,
        ) = retrieve_structure(
            question,
            structure_retriever,
        )

        # ----------------------------------------------------
        # BASELINE RAG + GPT-5-MINI
        # ----------------------------------------------------

        print("  Running baseline RAG + GPT-5-mini...")

        baseline_answer = generate_answer(
            question,
            baseline_context,
        )

        baseline_result = evaluate_one_answer(
            question=question,
            expected_act=expected_act,
            expected_section=expected_section,
            ground_truth_document=ground_truth_document,
            documents_for_metrics=baseline_documents,
            context_documents=baseline_context,
            answer=baseline_answer,
        )

        question_result["baseline"] = baseline_result

        add_metric_values(
            metric_values["baseline"],
            baseline_result["retrieval"],
            baseline_result["qa_evaluation"],
        )

        # ----------------------------------------------------
        # STRUCTURE-AWARE RAG + GPT-5-MINI
        # ----------------------------------------------------

        print("  Running structure-aware RAG + GPT-5-mini...")

        structure_answer = generate_answer(
            question,
            structure_context,
        )

        structure_result = evaluate_one_answer(
            question=question,
            expected_act=expected_act,
            expected_section=expected_section,
            ground_truth_document=ground_truth_document,
            documents_for_metrics=structure_documents,
            context_documents=structure_context,
            answer=structure_answer,
        )

        question_result["structure_aware"] = structure_result

        add_metric_values(
            metric_values["structure_aware"],
            structure_result["retrieval"],
            structure_result["qa_evaluation"],
        )

        # ----------------------------------------------------
        # STRUCTURE-AWARE RAG + LOCAL MODELS
        # ----------------------------------------------------

        for system_name, model_name in LOCAL_MODELS.items():

            result_key = f"structure_aware_{system_name}"

            print(
                f"  Running structure-aware RAG + {model_name}..."
            )

            # All local models receive the exact same retrieved
            # context as structure-aware GPT-5-mini.
            local_answer = generate_local_answer(
                model_name=model_name,
                question=question,
                documents=structure_context,
            )

            local_result = evaluate_one_answer(
                question=question,
                expected_act=expected_act,
                expected_section=expected_section,
                ground_truth_document=ground_truth_document,
                documents_for_metrics=structure_documents,
                context_documents=structure_context,
                answer=local_answer,
            )

            question_result[result_key] = local_result

            add_metric_values(
                metric_values[result_key],
                local_result["retrieval"],
                local_result["qa_evaluation"],
            )

        # ----------------------------------------------------
        # FAILURE AND RETRY COUNTS
        # ----------------------------------------------------

        for system_name in system_names:

            evaluation = question_result[
                system_name
            ]["qa_evaluation"]

            if evaluation["evaluator_parse_failed"]:
                evaluator_failures[system_name] += 1

            if evaluation["evaluator_retry_used"]:
                evaluator_retries[system_name] += 1

        results.append(question_result)

        print(
            "  Completed question "
            f"{question_id}."
        )

        # Save progress after each completed question.
        # If a later question fails, completed results remain.
        save_results(
            questions=questions,
            results=results,
            metric_values=metric_values,
            ground_truth_lookup=ground_truth_lookup,
            evaluator_failures=evaluator_failures,
            evaluator_retries=evaluator_retries,
            completed=False,
        )

    # ========================================================
    # FINAL SUMMARIES
    # ========================================================

    summaries = {
        system_name: summarise_metrics(
            metric_values[system_name]
        )
        for system_name in system_names
    }

    # Preserve the original baseline-versus-structure
    # difference calculation.
    structure_difference = {}

    for metric in summaries["baseline"]:

        baseline_value = summaries["baseline"][metric]
        structure_value = summaries["structure_aware"][metric]

        if baseline_value is None or structure_value is None:
            structure_difference[metric] = None

        else:
            structure_difference[metric] = (
                structure_value - baseline_value
            )

    # Compare answer-generation models with the same
    # structure-aware retrieval.
    model_comparison = {
        "GPT-5-mini": summaries["structure_aware"],
        "Qwen3-8B": summaries["structure_aware_qwen3_8b"],
        "Llama-3.1-8B": summaries["structure_aware_llama3_1_8b"],
    }

    # ========================================================
    # SAVE FINAL RESULTS
    # ========================================================

    save_results(
        questions=questions,
        results=results,
        metric_values=metric_values,
        ground_truth_lookup=ground_truth_lookup,
        evaluator_failures=evaluator_failures,
        evaluator_retries=evaluator_retries,
        completed=True,
        summaries=summaries,
        structure_difference=structure_difference,
        model_comparison=model_comparison,
    )

    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print("\n")
    print("=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)

    print("\nRetrieval Metrics")

    print(
        f"{'Metric':30s}"
        f"{'Baseline':15s}"
        f"{'Structure':15s}"
    )

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

    print(
        f"{'Metric':30s}"
        f"{'Baseline':15s}"
        f"{'Structure':15s}"
    )

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
    print(
        "All three models use the same structure-aware "
        "retrieval results."
    )

    comparison_metrics = (
        "answer_correctness",
        "groundedness",
        "completeness",
        "citation_source_accuracy",
        "reasoning_accuracy",
        "hallucination_rate",
    )

    for metric in comparison_metrics:

        print(f"\n{metric}")

        for model_name, model_summary in model_comparison.items():

            print(
                f"  {model_name:20s}: "
                f"{format_metric(model_summary[metric])}"
            )

    print("\nDifference: Structure-aware GPT - Baseline GPT")

    for metric, value in structure_difference.items():

        print(
            f"  {metric:35s}"
            f"{format_metric(value)}"
        )

    print("\nEvaluator parse failures:")

    for system_name, failures in evaluator_failures.items():

        print(f"  {system_name}: {failures}")

    print("\nEvaluator retries:")

    for system_name, retries in evaluator_retries.items():

        print(f"  {system_name}: {retries}")

    print("\nResults saved to:")
    print(OUTPUT_FILE)

    print("\nExperiment complete.")


# ============================================================
# RESULT FORMATTING
# ============================================================

def format_metric(value):

    if value is None:
        return "N/A"

    return f"{value:.4f}"


# ============================================================
# SAVE RESULTS
# ============================================================

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

            structure_value = current_summaries[
                "structure_aware"
            ][metric]

            if baseline_value is None or structure_value is None:
                structure_difference[metric] = None

            else:
                structure_difference[metric] = (
                    structure_value - baseline_value
                )

    if summaries is not None:
        current_summaries = summaries

    if model_comparison is None:

        model_comparison = {
            "GPT-5-mini": current_summaries["structure_aware"],
            "Qwen3-8B": current_summaries[
                "structure_aware_qwen3_8b"
            ],
            "Llama-3.1-8B": current_summaries[
                "structure_aware_llama3_1_8b"
            ],
        }

    final_results = {
        "experiment":
            "Baseline RAG, Structure-Aware RAG, and LLM Comparison",

        "evaluation_method":
            "Baseline-versus-structure-aware retrieval comparison "
            "and answer-generation model comparison using a shared "
            "structure-aware context for GPT-5-mini, Qwen3 8B, "
            "and Llama 3.1 8B.",

        "completed":
            completed,

        "total_questions":
            len(questions),

        "completed_questions":
            len(results),

        "embedding_model":
            EMBEDDING_MODEL,

        "answer_model":
            ANSWER_MODEL,

        "evaluator_model":
            EVALUATOR_MODEL,

        "local_models": {
            "qwen3_8b": QWEN_MODEL,
            "llama3_1_8b": LLAMA_MODEL,
        },

        "experiments": {
            "baseline":
                "Standard semantic retrieval with GPT-5-mini",

            "structure_aware":
                "Structure-aware retrieval with GPT-5-mini",

            "structure_aware_qwen3_8b":
                "Structure-aware retrieval with Qwen3 8B",

            "structure_aware_llama3_1_8b":
                "Structure-aware retrieval with Llama 3.1 8B",
        },

        "query_expansion":
            False,

        "query_expansion_note":
            "No separate LLM-based query-expansion experiment. "
            "The structure retriever retains its existing legal-term "
            "matching and fuzzy keyword correction.",

        "structure_retrieval": {
            "top_k_sections": TOP_K,
            "candidate_k": STRUCTURE_CANDIDATES,
            "retriever": "StructureAwareRetriever",
        },

        "reasoning_correct_threshold":
            REASONING_CORRECT_THRESHOLD,

        "precision_formula":
            "Relevant documents among returned top-k documents "
            "divided by the number of documents actually returned.",

        "recall_formula":
            "1 if the target Act and section is retrieved within "
            "the top-k results; otherwise 0.",

        "retrieval_definition":
            "A retrieved document is relevant when its Act and "
            "section match the expected Act and section.",

        "recall_definition":
            "Each evaluation question has one target Act and section. "
            "Recall@k measures whether that target appears in the "
            "top-k results.",

        "answer_correctness_definition":
            "Score from 0 to 100 representing how accurately the "
            "generated answer answers the legal question.",

        "groundedness_definition":
            "Score from 0 to 100 representing how well the answer "
            "is supported by retrieved legislative context.",

        "completeness_definition":
            "Score from 0 to 100 representing how completely the "
            "answer provides the important information required.",

        "citation_source_accuracy_definition":
            "Score from 0 to 100 representing the accuracy of "
            "the cited Act and section, with subsection considered "
            "when applicable.",

        "reasoning_accuracy_definition":
            "Proportion of answers that receive a reasoning score "
            "of at least 4, answer the question, and retrieve the "
            "target legal section.",

        "hallucination_formula":
            "Unsupported substantive claims divided by total "
            "substantive claims. Questions with zero claims or "
            "unparseable evaluator results are excluded.",

        "ground_truth_provisions_available":
            len(ground_truth_lookup),

        "evaluator_parse_failures":
            evaluator_failures,

        "evaluator_retries":
            evaluator_retries,

        "summary": {
            "baseline":
                current_summaries["baseline"],

            "structure_aware":
                current_summaries["structure_aware"],

            "difference_structure_minus_baseline":
                structure_difference,

            "model_comparison":
                model_comparison,
        },

        "results":
            results,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            final_results,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_experiment()
