import json
import re
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

TOP_K = 5
STRUCTURE_CANDIDATES = 50

REASONING_CORRECT_THRESHOLD = 4


# ============================================================
# MODELS
# ============================================================

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL
)

llm = ChatOpenAI(
    model=ANSWER_MODEL,
    temperature=0
)


# ============================================================
# BASELINE VECTOR STORE
# ============================================================

baseline_vectorstore = Chroma(
    collection_name="property_law_baseline",
    persist_directory=str(BASELINE_VECTORSTORE),
    embedding_function=embeddings,
)


# ============================================================
# STRUCTURE-AWARE RETRIEVERS
# ============================================================

structure_retriever = StructureAwareRetriever(
    candidate_k=STRUCTURE_CANDIDATES,
    use_query_expansion=False
)

query_expansion_retriever = StructureAwareRetriever(
    candidate_k=STRUCTURE_CANDIDATES,
    use_query_expansion=True
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
        section
    )

    return section.lower()


# ============================================================
# DOCUMENT HELPERS
# ============================================================

def get_metadata(document):

    if hasattr(document, "metadata"):
        return document.metadata

    if isinstance(document, dict):
        return document.get(
            "metadata",
            document
        )

    return {}


def get_page_content(document):

    if hasattr(document, "page_content"):
        return document.page_content

    if isinstance(document, dict):
        return document.get(
            "page_content",
            document.get("text", "")
        )

    return str(document)


def document_is_relevant(
    document,
    expected_act,
    expected_section
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
        and
        section == normalize_section(expected_section)
    )


# ============================================================
# BASELINE RETRIEVAL
# ============================================================

def retrieve_baseline(question):

    return baseline_vectorstore.similarity_search(
        question,
        k=TOP_K
    )


# ============================================================
# STRUCTURE-AWARE RETRIEVAL
# ============================================================

def retrieve_structure(
    question,
    retriever
):

    results = retriever.retrieve(
        question
    )

    documents = []
    context_documents = []

    for result in results:

        document = result.get("document")

        if document is not None:
            documents.append(document)

        related_documents = result.get(
            "related_documents",
            []
        )

        for related_document in related_documents:

            if related_document not in context_documents:
                context_documents.append(
                    related_document
                )

    return (
        documents,
        context_documents
    )


# ============================================================
# CONTEXT
# ============================================================

def build_context(documents):

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        metadata = get_metadata(document)

        act = metadata.get("act", "")
        section = metadata.get("section", "")
        heading = metadata.get("heading", "")
        subsection = metadata.get("subsection", "")
        paragraph = metadata.get("paragraph", "")
        nested_paragraph = metadata.get(
            "nested_paragraph",
            ""
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

{text}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(
    question,
    documents
):

    context = build_context(documents)

    prompt = f"""
You are a NSW property-law question answering system.

Answer the user's question using ONLY the supplied
NSW property-law legislation.

Do not use outside legal knowledge.

If the supplied context does not contain enough information
to answer the question, clearly say that the context is
insufficient.

When possible, identify the relevant Act and section.

If the question specifies a section, subsection, paragraph,
or nested paragraph, focus on that specific provision.

Do not provide unrelated legal provisions unless they are
necessary to answer the question.

QUESTION:
{question}

RETRIEVED LEGAL CONTEXT:
{context}

Provide a concise and accurate answer.
"""

    response = llm.invoke(prompt)

    return response.content.strip()


# ============================================================
# ROBUST JSON PARSER
# ============================================================

def extract_json_object(raw_text):

    if not raw_text:
        return None

    text = raw_text.strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    text = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.replace(
        "```",
        ""
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

    for index in range(
        start,
        len(text)
    ):

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

                candidate = text[
                    start:index + 1
                ]

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
    target_retrieved
):

    return f"""
You are evaluating a legal question-answering system.

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

Evaluate the generated answer against the ground-truth
legal provision and the retrieved legal context.

Evaluate the following four metrics using scores from
0 to 100.

1. ANSWER CORRECTNESS

How correctly does the answer answer the legal question?

100 = Fully correct.
80-99 = Mostly correct with minor issues.
60-79 = Partially correct with some important errors
        or omissions.
40-59 = Significant errors or omissions.
1-39 = Mostly incorrect.
0 = Completely incorrect or does not answer the question.

2. GROUNDEDNESS

How well are the answer's claims supported by the
retrieved legal context?

100 = All substantive claims are supported.
80-99 = Almost all claims are supported.
60-79 = Some important claims are unsupported.
40-59 = Many claims are unsupported.
1-39 = Most claims are unsupported.
0 = The answer is not grounded in the retrieved context.

3. COMPLETENESS

How completely does the answer provide the important
information required to answer the question?

100 = All important information is included.
80-99 = Almost all important information is included.
60-79 = Main answer is provided but some important
        information is missing.
40-59 = Substantial information is missing.
1-39 = Very little required information is provided.
0 = Required information is not provided.

4. CITATION/SOURCE ACCURACY

How accurately does the answer identify the relevant
legal source?

100 = Correct Act and section, and correct subsection
      when applicable.
80-99 = Mostly correct with a minor issue.
60-79 = Partly correct but with an important omission.
40-59 = Significant source errors.
1-39 = Mostly incorrect.
0 = Incorrect source or required source is not identified.

The evaluation dataset provides expected Act and section
information. Subsection-level ground truth is not available
for every question, so only evaluate subsection accuracy
when it can be determined from the supplied information.

Also evaluate:

REASONING SCORE:

5 = Completely correct and directly answers the question.
4 = Mostly correct with only minor omissions.
3 = Partially correct but missing important information.
2 = Mostly incorrect or materially incomplete.
1 = Incorrect or does not answer the question.

A claim is unsupported if it cannot be supported by the
retrieved legal context.

A refusal such as "the context is insufficient" is not a
correct answer when the ground-truth provision contains
enough information to answer the question.

If the target section was not retrieved,
reasoning_correct MUST be false.

Return ONLY this JSON object:

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
# ANSWER EVALUATION
# ============================================================

def evaluate_answer(
    question,
    expected_act,
    expected_section,
    answer,
    retrieved_documents,
    ground_truth_document
):

    target_retrieved = False

    for document in retrieved_documents:

        if document_is_relevant(
            document,
            expected_act,
            expected_section
        ):
            target_retrieved = True
            break

    ground_truth = ""

    if ground_truth_document is not None:
        ground_truth = get_page_content(
            ground_truth_document
        )

    retrieved_context = build_context(
        retrieved_documents
    )

    prompt = build_evaluator_prompt(
        question,
        expected_act,
        expected_section,
        ground_truth,
        retrieved_context,
        answer,
        target_retrieved
    )

    evaluator_parse_failed = False
    evaluator_retry_used = False

    response = llm.invoke(prompt)

    raw_response = response.content.strip()

    evaluation = extract_json_object(
        raw_response
    )

    if evaluation is None:

        evaluator_retry_used = True

        retry_prompt = f"""
Evaluate this legal QA result and return ONLY valid JSON.

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

Give four scores from 0 to 100:

answer_correctness:
How correctly the answer answers the question.

groundedness:
How well the answer is supported by the retrieved legal context.

completeness:
How completely the answer contains the important information
required by the question.

citation_source_accuracy:
How accurately the answer identifies the correct Act and section,
and subsection when applicable.

Also give reasoning_score from 1 to 5.

Return exactly:

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

If the target section was not retrieved,
reasoning_correct MUST be false.
"""

        retry_response = llm.invoke(
            retry_prompt
        )

        retry_raw = retry_response.content.strip()

        evaluation = extract_json_object(
            retry_raw
        )

        if evaluation is not None:
            raw_response = retry_raw

        else:
            evaluator_parse_failed = True
            raw_response = retry_raw

    if evaluation is not None:

        try:
            answer_correctness = int(
                evaluation.get(
                    "answer_correctness",
                    0
                )
            )
        except (TypeError, ValueError):
            answer_correctness = 0

        try:
            groundedness = int(
                evaluation.get(
                    "groundedness",
                    0
                )
            )
        except (TypeError, ValueError):
            groundedness = 0

        try:
            completeness = int(
                evaluation.get(
                    "completeness",
                    0
                )
            )
        except (TypeError, ValueError):
            completeness = 0

        try:
            citation_source_accuracy = int(
                evaluation.get(
                    "citation_source_accuracy",
                    0
                )
            )
        except (TypeError, ValueError):
            citation_source_accuracy = 0

        try:
            reasoning_score = int(
                evaluation.get(
                    "reasoning_score",
                    1
                )
            )
        except (TypeError, ValueError):
            reasoning_score = 1

        answers_question = bool(
            evaluation.get(
                "answers_question",
                False
            )
        )

        reasoning_correct = bool(
            evaluation.get(
                "reasoning_correct",
                False
            )
        )

        try:
            total_claims = int(
                evaluation.get(
                    "total_claims",
                    0
                )
            )
        except (TypeError, ValueError):
            total_claims = 0

        try:
            unsupported_claims = int(
                evaluation.get(
                    "unsupported_claims",
                    0
                )
            )
        except (TypeError, ValueError):
            unsupported_claims = 0

        explanation = str(
            evaluation.get(
                "explanation",
                ""
            )
        )

        answer_correctness = max(
            0,
            min(100, answer_correctness)
        )

        groundedness = max(
            0,
            min(100, groundedness)
        )

        completeness = max(
            0,
            min(100, completeness)
        )

        citation_source_accuracy = max(
            0,
            min(100, citation_source_accuracy)
        )

        reasoning_score = max(
            1,
            min(5, reasoning_score)
        )

        total_claims = max(
            0,
            total_claims
        )

        unsupported_claims = max(
            0,
            min(
                unsupported_claims,
                total_claims
            )
        )

    else:

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
            "Evaluator response could not be parsed "
            "after retry."
        )

    if not evaluator_parse_failed:

        if reasoning_score < REASONING_CORRECT_THRESHOLD:
            reasoning_correct = False

        if not answers_question:
            reasoning_correct = False

        if not target_retrieved:
            reasoning_correct = False

    if (
        not evaluator_parse_failed
        and total_claims is not None
        and total_claims > 0
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
            "metadatas"
        ]
    )

    documents = []

    raw_documents = collection_data.get(
        "documents",
        []
    )

    raw_metadatas = collection_data.get(
        "metadatas",
        []
    )

    for text, metadata in zip(
        raw_documents,
        raw_metadatas
    ):

        documents.append(
            {
                "page_content": text,
                "metadata": metadata or {}
            }
        )

    lookup = {}

    for document in documents:

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
            section
        )

        if key not in lookup:
            lookup[key] = document

    return lookup


# ============================================================
# METRICS
# ============================================================

def precision_at_k(
    documents,
    expected_act,
    expected_section,
    k
):

    top_documents = documents[:k]

    if not top_documents:
        return 0.0

    relevant = 0

    for document in top_documents:

        if document_is_relevant(
            document,
            expected_act,
            expected_section
        ):
            relevant += 1

    return relevant / len(top_documents)


def recall_at_k(
    documents,
    expected_act,
    expected_section,
    k
):

    top_documents = documents[:k]

    for document in top_documents:

        if document_is_relevant(
            document,
            expected_act,
            expected_section
        ):
            return 1.0

    return 0.0


def average(values):

    valid_values = []

    for value in values:

        if value is not None:
            valid_values.append(value)

    if not valid_values:
        return 0.0

    return sum(valid_values) / len(valid_values)


# ============================================================
# EVALUATE ONE SYSTEM
# ============================================================

def evaluate_system(
    question,
    expected_act,
    expected_section,
    ground_truth_document,
    retriever=None
):

    if retriever is None:

        documents = retrieve_baseline(
            question
        )

        context_documents = documents

    else:

        (
            documents,
            context_documents
        ) = retrieve_structure(
            question,
            retriever
        )

    p1 = precision_at_k(
        documents,
        expected_act,
        expected_section,
        1
    )

    p5 = precision_at_k(
        documents,
        expected_act,
        expected_section,
        5
    )

    r1 = recall_at_k(
        documents,
        expected_act,
        expected_section,
        1
    )

    r5 = recall_at_k(
        documents,
        expected_act,
        expected_section,
        5
    )

    answer = generate_answer(
        question,
        context_documents
    )

    evaluation = evaluate_answer(
        question,
        expected_act,
        expected_section,
        answer,
        context_documents,
        ground_truth_document
    )

    return {
        "retrieval": {
            "precision_at_1": p1,
            "precision_at_5": p5,
            "recall_at_1": r1,
            "recall_at_5": r5,
            "hit_at_5": r5 == 1.0,
        },

        "answer": answer,

        "qa_evaluation": evaluation,
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        dataset = json.load(file)

    questions = dataset["questions"]

    print("=" * 80)
    print("FINAL EXPERIMENT EVALUATION")
    print("=" * 80)

    print(
        f"\nQuestions: {len(questions)}"
    )

    print(
        f"Embedding model: {EMBEDDING_MODEL}"
    )

    print(
        f"Answer model: {ANSWER_MODEL}"
    )

    print("\nExperiments:")
    print("  1. Baseline RAG")
    print("  2. Structure-aware RAG")
    print("  3. Structure-aware RAG + Query Expansion")

    print("\nStructure-aware retrieval:")
    print(f"  Top sections: {TOP_K}")
    print(
        f"  Candidate chunks: "
        f"{STRUCTURE_CANDIDATES}"
    )

    print(
        "\nBuilding ground-truth lookup..."
    )

    ground_truth_lookup = (
        build_ground_truth_lookup()
    )

    print(
        f"Ground-truth provisions found: "
        f"{len(ground_truth_lookup)}"
    )

    systems = {
        "baseline": None,
        "structure_aware": structure_retriever,
        "structure_aware_query_expansion":
            query_expansion_retriever,
    }

    metric_values = {}

    for system_name in systems:

        metric_values[system_name] = {
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

    evaluator_failures = {}

    for system_name in systems:
        evaluator_failures[system_name] = 0

    results = []

    for index, item in enumerate(
        questions,
        start=1
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
            normalize_section(expected_section)
        )

        ground_truth_document = (
            ground_truth_lookup.get(
                ground_truth_key
            )
        )

        question_result = {
            "question_number": question_id,
            "question": question,
            "expected_act": expected_act,
            "expected_section": expected_section,
        }

        for system_name in systems:

            retriever = systems[system_name]

            system_result = evaluate_system(
                question,
                expected_act,
                expected_section,
                ground_truth_document,
                retriever
            )

            question_result[system_name] = system_result

            retrieval = system_result[
                "retrieval"
            ]

            evaluation = system_result[
                "qa_evaluation"
            ]

            metric_values[system_name][
                "precision_at_1"
            ].append(
                retrieval["precision_at_1"]
            )

            metric_values[system_name][
                "precision_at_5"
            ].append(
                retrieval["precision_at_5"]
            )

            metric_values[system_name][
                "recall_at_1"
            ].append(
                retrieval["recall_at_1"]
            )

            metric_values[system_name][
                "recall_at_5"
            ].append(
                retrieval["recall_at_5"]
            )

            if evaluation[
                "answer_correctness"
            ] is not None:

                metric_values[system_name][
                    "answer_correctness"
                ].append(
                    evaluation[
                        "answer_correctness"
                    ]
                )

            if evaluation[
                "groundedness"
            ] is not None:

                metric_values[system_name][
                    "groundedness"
                ].append(
                    evaluation[
                        "groundedness"
                    ]
                )

            if evaluation[
                "completeness"
            ] is not None:

                metric_values[system_name][
                    "completeness"
                ].append(
                    evaluation[
                        "completeness"
                    ]
                )

            if evaluation[
                "citation_source_accuracy"
            ] is not None:

                metric_values[system_name][
                    "citation_source_accuracy"
                ].append(
                    evaluation[
                        "citation_source_accuracy"
                    ]
                )

            if evaluation[
                "reasoning_correct"
            ] is not None:

                metric_values[system_name][
                    "reasoning_accuracy"
                ].append(
                    int(
                        evaluation[
                            "reasoning_correct"
                        ]
                    )
                )

            if evaluation[
                "hallucination_rate"
            ] is not None:

                metric_values[system_name][
                    "hallucination_rate"
                ].append(
                    evaluation[
                        "hallucination_rate"
                    ]
                )

            if evaluation[
                "evaluator_parse_failed"
            ]:

                evaluator_failures[
                    system_name
                ] += 1

            print(
                f"  {system_name}: "
                f"R@5={retrieval['recall_at_5']:.0f}, "
                f"correctness="
                f"{evaluation['answer_correctness']}"
            )

        results.append(question_result)

    # ========================================================
    # FINAL SUMMARIES
    # ========================================================

    summaries = {}

    for system_name in systems:

        values = metric_values[system_name]

        summaries[system_name] = {
            "precision_at_1":
                average(
                    values["precision_at_1"]
                ),

            "precision_at_5":
                average(
                    values["precision_at_5"]
                ),

            "recall_at_1":
                average(
                    values["recall_at_1"]
                ),

            "recall_at_5":
                average(
                    values["recall_at_5"]
                ),

            "answer_correctness":
                average(
                    values["answer_correctness"]
                ),

            "groundedness":
                average(
                    values["groundedness"]
                ),

            "completeness":
                average(
                    values["completeness"]
                ),

            "citation_source_accuracy":
                average(
                    values["citation_source_accuracy"]
                ),

            "reasoning_accuracy":
                average(
                    values["reasoning_accuracy"]
                ),

            "hallucination_rate":
                average(
                    values["hallucination_rate"]
                ),
        }

    # ========================================================
    # DIFFERENCES
    # ========================================================

    structure_difference = {}

    query_expansion_difference = {}

    for metric in summaries["baseline"]:

        structure_difference[metric] = (
            summaries["structure_aware"][metric]
            -
            summaries["baseline"][metric]
        )

        query_expansion_difference[metric] = (
            summaries[
                "structure_aware_query_expansion"
            ][metric]
            -
            summaries["structure_aware"][metric]
        )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    final_results = {
        "experiment":
            "Baseline RAG vs Structure-Aware RAG vs "
            "Structure-Aware RAG with Query Expansion",

        "evaluation_method":
            "Controlled comparison using the same NSW "
            "property-law corpus, evaluation questions, "
            "embedding model and answer model.",

        "total_questions":
            len(questions),

        "embedding_model":
            EMBEDDING_MODEL,

        "answer_model":
            ANSWER_MODEL,

        "experiments":
            {
                "baseline":
                    "Standard semantic retrieval",

                "structure_aware":
                    "Structure-aware retrieval without query expansion",

                "structure_aware_query_expansion":
                    "Structure-aware retrieval with LLM-based query expansion",
            },

        "structure_retrieval":
            {
                "top_k_sections":
                    TOP_K,

                "candidate_k":
                    STRUCTURE_CANDIDATES,

                "retriever":
                    "StructureAwareRetriever",
            },

        "query_expansion":
            {
                "enabled_for":
                    "structure_aware_query_expansion",

                "model":
                    ANSWER_MODEL,
            },

        "reasoning_correct_threshold":
            REASONING_CORRECT_THRESHOLD,

        "precision_formula":
            "relevant retrieved documents in top-k / k",

        "recall_formula":
            "target section retrieved within top-k / target section exists",

        "retrieval_definition":
            "A retrieved document is relevant when its Act "
            "and section match the expected Act and section.",

        "recall_definition":
            "Each question has one target Act and section. "
            "Recall@k is 1 when the target section appears "
            "within the top-k retrieved sections.",

        "answer_correctness_definition":
            "Score from 0 to 100 representing how accurately "
            "the generated answer answers the legal question.",

        "groundedness_definition":
            "Score from 0 to 100 representing how well the "
            "generated answer is supported by retrieved legal context.",

        "completeness_definition":
            "Score from 0 to 100 representing how completely "
            "the generated answer contains the important information "
            "required by the question.",

        "citation_source_accuracy_definition":
            "Score from 0 to 100 representing how accurately "
            "the generated answer identifies the relevant Act "
            "and section, with subsection considered where applicable.",

        "reasoning_accuracy_definition":
            "A generated answer is reasoning-correct only "
            "when it answers the question, has a reasoning "
            "score of 4 or 5, and the target legal section "
            "was retrieved.",

        "hallucination_formula":
            "unsupported_claims / total_claims",

        "ground_truth_provisions_available":
            len(ground_truth_lookup),

        "evaluator_parse_failures":
            evaluator_failures,

        "summary":
            {
                "baseline":
                    summaries["baseline"],

                "structure_aware":
                    summaries["structure_aware"],

                "structure_aware_query_expansion":
                    summaries[
                        "structure_aware_query_expansion"
                    ],

                "difference_structure_minus_baseline":
                    structure_difference,

                "difference_query_expansion_minus_structure":
                    query_expansion_difference,
            },

        "results":
            results,
    }

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            final_results,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # PRINT RESULTS
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
        f"{'Structure + QE':15s}"
    )

    print("-" * 80)

    print(
        f"{'Precision@1':30s}"
        f"{summaries['baseline']['precision_at_1']:.4f}"
        f"{'':8s}"
        f"{summaries['structure_aware']['precision_at_1']:.4f}"
        f"{'':8s}"
        f"{summaries['structure_aware_query_expansion']['precision_at_1']:.4f}"
    )

    print(
        f"{'Precision@5':30s}"
        f"{summaries['baseline']['precision_at_5']:.4f}"
        f"{'':8s}"
        f"{summaries['structure_aware']['precision_at_5']:.4f}"
        f"{'':8s}"
        f"{summaries['structure_aware_query_expansion']['precision_at_5']:.4f}"
    )

    print(
        f"{'Recall@1':30s}"
        f"{summaries['baseline']['recall_at_1']:.4f}"
        f"{'':8s}"
        f"{summaries['structure_aware']['recall_at_1']:.4f}"
        f"{'':8s}"
        f"{summaries['structure_aware_query_expansion']['recall_at_1']:.4f}"
    )

    print(
        f"{'Recall@5':30s}"
        f"{summaries['baseline']['recall_at_5']:.4f}"
        f"{'':8s}"
        f"{summaries['structure_aware']['recall_at_5']:.4f}"
        f"{'':8s}"
        f"{summaries['structure_aware_query_expansion']['recall_at_5']:.4f}"
    )

    print("\nAnswer Evaluation Metrics")

    print(
        f"{'Metric':30s}"
        f"{'Baseline':15s}"
        f"{'Structure':15s}"
        f"{'Structure + QE':15s}"
    )

    print("-" * 80)

    print(
        f"{'Answer Correctness (%)':30s}"
        f"{summaries['baseline']['answer_correctness']:.2f}"
        f"{'':8s}"
        f"{summaries['structure_aware']['answer_correctness']:.2f}"
        f"{'':8s}"
        f"{summaries['structure_aware_query_expansion']['answer_correctness']:.2f}"
    )

    print(
        f"{'Groundedness (%)':30s}"
        f"{summaries['baseline']['groundedness']:.2f}"
        f"{'':8s}"
        f"{summaries['structure_aware']['groundedness']:.2f}"
        f"{'':8s}"
        f"{summaries['structure_aware_query_expansion']['groundedness']:.2f}"
    )

    print(
        f"{'Completeness (%)':30s}"
        f"{summaries['baseline']['completeness']:.2f}"
        f"{'':8s}"
        f"{summaries['structure_aware']['completeness']:.2f}"
        f"{'':8s}"
        f"{summaries['structure_aware_query_expansion']['completeness']:.2f}"
    )

    print(
        f"{'Citation/Source Accuracy (%)':30s}"
        f"{summaries['baseline']['citation_source_accuracy']:.2f}"
        f"{'':8s}"
        f"{summaries['structure_aware']['citation_source_accuracy']:.2f}"
        f"{'':8s}"
        f"{summaries['structure_aware_query_expansion']['citation_source_accuracy']:.2f}"
    )

    print("\nSupporting Metrics")

    print(
        f"{'Metric':30s}"
        f"{'Baseline':15s}"
        f"{'Structure':15s}"
        f"{'Structure + QE':15s}"
    )

    print("-" * 80)

    print(
        f"{'Reasoning Accuracy':30s}"
        f"{summaries['baseline']['reasoning_accuracy']:.4f}"
        f"{'':8s}"
        f"{summaries['structure_aware']['reasoning_accuracy']:.4f}"
        f"{'':8s}"
        f"{summaries['structure_aware_query_expansion']['reasoning_accuracy']:.4f}"
    )

    print(
        f"{'Hallucination Rate':30s}"
        f"{summaries['baseline']['hallucination_rate']:.4f}"
        f"{'':8s}"
        f"{summaries['structure_aware']['hallucination_rate']:.4f}"
        f"{'':8s}"
        f"{summaries['structure_aware_query_expansion']['hallucination_rate']:.4f}"
    )

    print("\nDifference: Structure - Baseline")

    print("-" * 80)

    for metric, value in structure_difference.items():

        print(
            f"{metric:35s}"
            f"{value:+.4f}"
        )

    print("\nDifference: Structure + QE - Structure")

    print("-" * 80)

    for metric, value in query_expansion_difference.items():

        print(
            f"{metric:35s}"
            f"{value:+.4f}"
        )

    print("\nEvaluator parse failures:")

    for system_name, failures in evaluator_failures.items():

        print(
            f"  {system_name}: {failures}"
        )

    print("\nResults saved to:")
    print(OUTPUT_FILE)

    print("\nExperiment complete.")


if __name__ == "__main__":
    run_experiment()