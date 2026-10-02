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
# SETTINGS
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
# BASELINE VECTORSTORE
# ============================================================

baseline_vectorstore = Chroma(
    collection_name="property_law_baseline",
    persist_directory=str(BASELINE_VECTORSTORE),
    embedding_function=embeddings,
)


# ============================================================
# STRUCTURE-AWARE RETRIEVER
# ============================================================

structure_retriever = StructureAwareRetriever(
    candidate_k=STRUCTURE_CANDIDATES
)


# ============================================================
# NORMALISE ACT NAMES
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


# ============================================================
# NORMALISE SECTION NUMBERS
# ============================================================

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

def retrieve_structure(question):

    results = structure_retriever.retrieve(
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

        for document in related_documents:

            if document not in context_documents:
                context_documents.append(document)

    return documents, context_documents


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(documents):

    context = []

    for number, document in enumerate(
        documents,
        start=1
    ):

        metadata = get_metadata(
            document
        )

        act = metadata.get(
            "act",
            ""
        )

        section = metadata.get(
            "section",
            ""
        )

        heading = metadata.get(
            "heading",
            ""
        )

        subsection = metadata.get(
            "subsection",
            ""
        )

        paragraph = metadata.get(
            "paragraph",
            ""
        )

        nested_paragraph = metadata.get(
            "nested_paragraph",
            ""
        )

        text = get_page_content(
            document
        )

        source = f"""
[SOURCE {number}]
Act: {act}
Section: {section}
Heading: {heading}
Subsection: {subsection}
Paragraph: {paragraph}
Nested paragraph: {nested_paragraph}

{text}
"""

        context.append(source)

    return "\n".join(context)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, documents):

    context = build_context(
        documents
    )

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

    response = llm.invoke(
        prompt
    )

    return response.content.strip()


# ============================================================
# EXTRACT JSON
# ============================================================

def extract_json_object(text):

    if not text:
        return None

    text = text.strip()

    # Try normal JSON first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Remove Markdown code fences
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

    # Find JSON inside the response
    start = text.find("{")

    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(
        start,
        len(text)
    ):

        char = text[i]

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
                    return json.loads(
                        text[start:i + 1]
                    )
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

Use scores from 0 to 100 for:

1. ANSWER CORRECTNESS
How correctly does the answer answer the legal question?

2. GROUNDEDNESS
How well are the answer's claims supported by the
retrieved legal context?

3. COMPLETENESS
How completely does the answer provide the important
information required to answer the question?

4. CITATION/SOURCE ACCURACY
How accurately does the answer identify the relevant
legal source?

Also give:

REASONING SCORE:
5 = Completely correct
4 = Mostly correct
3 = Partially correct
2 = Mostly incorrect
1 = Incorrect

A claim is unsupported if it cannot be supported by the
retrieved legal context.

If the target section was not retrieved,
reasoning_correct MUST be false.

Do not use outside legal knowledge.

Return ONLY this JSON:

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
# EVALUATE ANSWER
# ============================================================

def evaluate_answer(
    question,
    expected_act,
    expected_section,
    answer,
    retrieved_documents,
    ground_truth_document
):

    # Check whether the target section was retrieved
    target_retrieved = False

    for document in retrieved_documents:

        if document_is_relevant(
            document,
            expected_act,
            expected_section
        ):
            target_retrieved = True
            break

    # Ground truth text
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

    response = llm.invoke(
        prompt
    )

    raw_response = response.content.strip()

    evaluation = extract_json_object(
        raw_response
    )

    retry_used = False

    # Retry once if JSON parsing failed
    if evaluation is None:

        retry_used = True

        retry_prompt = f"""
Evaluate the following legal QA result.

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

Return ONLY valid JSON:

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

        response = llm.invoke(
            retry_prompt
        )

        raw_response = (
            response.content.strip()
        )

        evaluation = extract_json_object(
            raw_response
        )

    # If parsing failed twice
    if evaluation is None:

        return {
            "answer_correctness": None,
            "groundedness": None,
            "completeness": None,
            "citation_source_accuracy": None,
            "reasoning_score": None,
            "answers_question": None,
            "reasoning_correct": None,
            "total_claims": None,
            "unsupported_claims": None,
            "hallucination_rate": None,
            "target_section_retrieved":
                target_retrieved,
            "evaluator_parse_failed": True,
            "evaluator_retry_used":
                retry_used,
            "explanation":
                "Evaluator response could not be parsed.",
            "raw_evaluator_response":
                raw_response
        }

    # Read evaluator results
    answer_correctness = int(
        evaluation.get(
            "answer_correctness",
            0
        )
    )

    groundedness = int(
        evaluation.get(
            "groundedness",
            0
        )
    )

    completeness = int(
        evaluation.get(
            "completeness",
            0
        )
    )

    citation_source_accuracy = int(
        evaluation.get(
            "citation_source_accuracy",
            0
        )
    )

    reasoning_score = int(
        evaluation.get(
            "reasoning_score",
            1
        )
    )

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

    total_claims = int(
        evaluation.get(
            "total_claims",
            0
        )
    )

    unsupported_claims = int(
        evaluation.get(
            "unsupported_claims",
            0
        )
    )

    explanation = evaluation.get(
        "explanation",
        ""
    )

    # Keep scores within valid ranges
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

    # ========================================================
    # OBJECTIVE REASONING CORRECTION
    # ========================================================

    if reasoning_score < REASONING_CORRECT_THRESHOLD:
        reasoning_correct = False

    if not answers_question:
        reasoning_correct = False

    if not target_retrieved:
        reasoning_correct = False

    # ========================================================
    # HALLUCINATION RATE
    # ========================================================

    if total_claims > 0:

        hallucination_rate = (
            unsupported_claims
            /
            total_claims
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
            False,

        "evaluator_retry_used":
            retry_used,

        "explanation":
            explanation,

        "raw_evaluator_response":
            raw_response
    }


# ============================================================
# BUILD GROUND TRUTH
# ============================================================

def build_ground_truth_lookup():

    data = baseline_vectorstore.get(
        include=[
            "documents",
            "metadatas"
        ]
    )

    lookup = {}

    for text, metadata in zip(
        data["documents"],
        data["metadatas"]
    ):

        metadata = metadata or {}

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

            lookup[key] = {
                "page_content": text,
                "metadata": metadata
            }

    return lookup


# ============================================================
# RETRIEVAL METRICS
# ============================================================

def precision_at_k(
    documents,
    expected_act,
    expected_section,
    k
):

    documents = documents[:k]

    if not documents:
        return 0.0

    relevant = 0

    for document in documents:

        if document_is_relevant(
            document,
            expected_act,
            expected_section
        ):
            relevant += 1

    return relevant / len(
        documents
    )


def recall_at_k(
    documents,
    expected_act,
    expected_section,
    k
):

    documents = documents[:k]

    for document in documents:

        if document_is_relevant(
            document,
            expected_act,
            expected_section
        ):
            return 1.0

    return 0.0


# ============================================================
# AVERAGE
# ============================================================

def average(values):

    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return 0.0

    return sum(values) / len(values)


# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_experiment():

    # --------------------------------------------------------
    # Load questions
    # --------------------------------------------------------

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        dataset = json.load(file)

    questions = dataset["questions"]

    print("=" * 80)
    print("FINAL EXPERIMENT: BASELINE VS STRUCTURE-AWARE RAG")
    print("=" * 80)

    print(
        f"Questions: {len(questions)}"
    )

    print(
        f"Embedding model: {EMBEDDING_MODEL}"
    )

    print(
        f"Answer model: {ANSWER_MODEL}"
    )

    print(
        f"Structure candidates: "
        f"{STRUCTURE_CANDIDATES}"
    )

    print(
        f"Top K: {TOP_K}"
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Metric lists
    # --------------------------------------------------------

    baseline_p1 = []
    baseline_p5 = []
    baseline_r1 = []
    baseline_r5 = []

    baseline_correctness = []
    baseline_groundedness = []
    baseline_completeness = []
    baseline_citation = []
    baseline_reasoning = []
    baseline_hallucination = []

    structure_p1 = []
    structure_p5 = []
    structure_r1 = []
    structure_r5 = []

    structure_correctness = []
    structure_groundedness = []
    structure_completeness = []
    structure_citation = []
    structure_reasoning = []
    structure_hallucination = []

    results = []

    baseline_failures = 0
    structure_failures = 0

    # ========================================================
    # PROCESS QUESTIONS
    # ========================================================

    for number, item in enumerate(
        questions,
        start=1
    ):

        question_id = item["id"]
        question = item["question"]
        expected_act = item["act"]
        expected_section = item["section"]

        print(
            f"\n[{number}/{len(questions)}] "
            f"Q{question_id}: {question}"
        )

        # ----------------------------------------------------
        # Ground truth document
        # ----------------------------------------------------

        key = (
            normalize_act(expected_act),
            normalize_section(expected_section)
        )

        ground_truth_document = (
            ground_truth_lookup.get(key)
        )

        # ====================================================
        # BASELINE
        # ====================================================

        baseline_documents = (
            retrieve_baseline(question)
        )

        baseline_p1_value = precision_at_k(
            baseline_documents,
            expected_act,
            expected_section,
            1
        )

        baseline_p5_value = precision_at_k(
            baseline_documents,
            expected_act,
            expected_section,
            5
        )

        baseline_r1_value = recall_at_k(
            baseline_documents,
            expected_act,
            expected_section,
            1
        )

        baseline_r5_value = recall_at_k(
            baseline_documents,
            expected_act,
            expected_section,
            5
        )

        print(
            f"  Baseline retrieval: "
            f"R@5={baseline_r5_value:.0f}"
        )

        baseline_answer = generate_answer(
            question,
            baseline_documents
        )

        baseline_evaluation = evaluate_answer(
            question,
            expected_act,
            expected_section,
            baseline_answer,
            baseline_documents,
            ground_truth_document
        )

        print(
            f"  Baseline correctness: "
            f"{baseline_evaluation['answer_correctness']}"
        )

        print(
            f"  Baseline groundedness: "
            f"{baseline_evaluation['groundedness']}"
        )

        print(
            f"  Baseline completeness: "
            f"{baseline_evaluation['completeness']}"
        )

        print(
            f"  Baseline source accuracy: "
            f"{baseline_evaluation['citation_source_accuracy']}"
        )

        if baseline_evaluation[
            "evaluator_parse_failed"
        ]:

            baseline_failures += 1

        # ====================================================
        # STRUCTURE-AWARE
        # ====================================================

        (
            structure_documents,
            structure_context
        ) = retrieve_structure(question)

        structure_p1_value = precision_at_k(
            structure_documents,
            expected_act,
            expected_section,
            1
        )

        structure_p5_value = precision_at_k(
            structure_documents,
            expected_act,
            expected_section,
            5
        )

        structure_r1_value = recall_at_k(
            structure_documents,
            expected_act,
            expected_section,
            1
        )

        structure_r5_value = recall_at_k(
            structure_documents,
            expected_act,
            expected_section,
            5
        )

        print(
            f"  Structure retrieval: "
            f"R@5={structure_r5_value:.0f}"
        )

        structure_answer = generate_answer(
            question,
            structure_context
        )

        structure_evaluation = evaluate_answer(
            question,
            expected_act,
            expected_section,
            structure_answer,
            structure_context,
            ground_truth_document
        )

        print(
            f"  Structure correctness: "
            f"{structure_evaluation['answer_correctness']}"
        )

        print(
            f"  Structure groundedness: "
            f"{structure_evaluation['groundedness']}"
        )

        print(
            f"  Structure completeness: "
            f"{structure_evaluation['completeness']}"
        )

        print(
            f"  Structure source accuracy: "
            f"{structure_evaluation['citation_source_accuracy']}"
        )

        if structure_evaluation[
            "evaluator_parse_failed"
        ]:

            structure_failures += 1

        # ----------------------------------------------------
        # Save retrieval metrics
        # ----------------------------------------------------

        baseline_p1.append(
            baseline_p1_value
        )

        baseline_p5.append(
            baseline_p5_value
        )

        baseline_r1.append(
            baseline_r1_value
        )

        baseline_r5.append(
            baseline_r5_value
        )

        structure_p1.append(
            structure_p1_value
        )

        structure_p5.append(
            structure_p5_value
        )

        structure_r1.append(
            structure_r1_value
        )

        structure_r5.append(
            structure_r5_value
        )

        # ----------------------------------------------------
        # Save baseline answer metrics
        # ----------------------------------------------------

        if baseline_evaluation[
            "answer_correctness"
        ] is not None:

            baseline_correctness.append(
                baseline_evaluation[
                    "answer_correctness"
                ]
            )

        if baseline_evaluation[
            "groundedness"
        ] is not None:

            baseline_groundedness.append(
                baseline_evaluation[
                    "groundedness"
                ]
            )

        if baseline_evaluation[
            "completeness"
        ] is not None:

            baseline_completeness.append(
                baseline_evaluation[
                    "completeness"
                ]
            )

        if baseline_evaluation[
            "citation_source_accuracy"
        ] is not None:

            baseline_citation.append(
                baseline_evaluation[
                    "citation_source_accuracy"
                ]
            )

        if baseline_evaluation[
            "reasoning_correct"
        ] is not None:

            baseline_reasoning.append(
                int(
                    baseline_evaluation[
                        "reasoning_correct"
                    ]
                )
            )

        if baseline_evaluation[
            "hallucination_rate"
        ] is not None:

            baseline_hallucination.append(
                baseline_evaluation[
                    "hallucination_rate"
                ]
            )

        # ----------------------------------------------------
        # Save structure answer metrics
        # ----------------------------------------------------

        if structure_evaluation[
            "answer_correctness"
        ] is not None:

            structure_correctness.append(
                structure_evaluation[
                    "answer_correctness"
                ]
            )

        if structure_evaluation[
            "groundedness"
        ] is not None:

            structure_groundedness.append(
                structure_evaluation[
                    "groundedness"
                ]
            )

        if structure_evaluation[
            "completeness"
        ] is not None:

            structure_completeness.append(
                structure_evaluation[
                    "completeness"
                ]
            )

        if structure_evaluation[
            "citation_source_accuracy"
        ] is not None:

            structure_citation.append(
                structure_evaluation[
                    "citation_source_accuracy"
                ]
            )

        if structure_evaluation[
            "reasoning_correct"
        ] is not None:

            structure_reasoning.append(
                int(
                    structure_evaluation[
                        "reasoning_correct"
                    ]
                )
            )

        if structure_evaluation[
            "hallucination_rate"
        ] is not None:

            structure_hallucination.append(
                structure_evaluation[
                    "hallucination_rate"
                ]
            )

        # ====================================================
        # SAVE QUESTION RESULT
        # ====================================================

        results.append(
            {
                "question_number": question_id,
                "question": question,
                "expected_act": expected_act,
                "expected_section": expected_section,

                "baseline": {
                    "retrieval": {
                        "precision_at_1":
                            baseline_p1_value,

                        "precision_at_5":
                            baseline_p5_value,

                        "recall_at_1":
                            baseline_r1_value,

                        "recall_at_5":
                            baseline_r5_value,

                        "hit_at_5":
                            baseline_r5_value == 1.0,
                    },

                    "answer":
                        baseline_answer,

                    "qa_evaluation":
                        baseline_evaluation,
                },

                "structure_aware": {
                    "retrieval": {
                        "precision_at_1":
                            structure_p1_value,

                        "precision_at_5":
                            structure_p5_value,

                        "recall_at_1":
                            structure_r1_value,

                        "recall_at_5":
                            structure_r5_value,

                        "hit_at_5":
                            structure_r5_value == 1.0,
                    },

                    "answer":
                        structure_answer,

                    "qa_evaluation":
                        structure_evaluation,
                },
            }
        )

    # ========================================================
    # SUMMARIES
    # ========================================================

    baseline_summary = {
        "precision_at_1":
            average(baseline_p1),

        "precision_at_5":
            average(baseline_p5),

        "recall_at_1":
            average(baseline_r1),

        "recall_at_5":
            average(baseline_r5),

        "answer_correctness":
            average(baseline_correctness),

        "groundedness":
            average(baseline_groundedness),

        "completeness":
            average(baseline_completeness),

        "citation_source_accuracy":
            average(baseline_citation),

        "reasoning_accuracy":
            average(baseline_reasoning),

        "hallucination_rate":
            average(baseline_hallucination),
    }

    structure_summary = {
        "precision_at_1":
            average(structure_p1),

        "precision_at_5":
            average(structure_p5),

        "recall_at_1":
            average(structure_r1),

        "recall_at_5":
            average(structure_r5),

        "answer_correctness":
            average(structure_correctness),

        "groundedness":
            average(structure_groundedness),

        "completeness":
            average(structure_completeness),

        "citation_source_accuracy":
            average(structure_citation),

        "reasoning_accuracy":
            average(structure_reasoning),

        "hallucination_rate":
            average(structure_hallucination),
    }

    # Calculate Structure - Baseline
    difference = {}

    for metric in baseline_summary:

        difference[metric] = (
            structure_summary[metric]
            -
            baseline_summary[metric]
        )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    final_results = {
        "experiment":
            "Baseline RAG vs Structure-Aware RAG",

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

        "structure_retrieval": {
            "top_k_sections":
                TOP_K,

            "candidate_k":
                structure_retriever.candidate_k,

            "retriever":
                "StructureAwareRetriever",
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

        "reasoning_accuracy_definition":
            "A generated answer is reasoning-correct only "
            "when it answers the question, has a reasoning "
            "score of 4 or 5, and the target legal section "
            "was retrieved.",

        "hallucination_formula":
            "unsupported_claims / total_claims",

        "ground_truth_provisions_available":
            len(ground_truth_lookup),

        "evaluator_parse_failures": {
            "baseline":
                baseline_failures,

            "structure_aware":
                structure_failures,
        },

        "summary": {
            "baseline":
                baseline_summary,

            "structure_aware":
                structure_summary,

            "difference_structure_minus_baseline":
                difference,
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
    # PRINT FINAL RESULTS
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print("\nRetrieval Metrics")

    print(
        f"Precision@1: "
        f"{baseline_summary['precision_at_1']:.4f}"
        f" -> "
        f"{structure_summary['precision_at_1']:.4f}"
    )

    print(
        f"Precision@5: "
        f"{baseline_summary['precision_at_5']:.4f}"
        f" -> "
        f"{structure_summary['precision_at_5']:.4f}"
    )

    print(
        f"Recall@1: "
        f"{baseline_summary['recall_at_1']:.4f}"
        f" -> "
        f"{structure_summary['recall_at_1']:.4f}"
    )

    print(
        f"Recall@5: "
        f"{baseline_summary['recall_at_5']:.4f}"
        f" -> "
        f"{structure_summary['recall_at_5']:.4f}"
    )

    print("\nAnswer Evaluation Metrics")

    print(
        f"Correctness: "
        f"{baseline_summary['answer_correctness']:.2f}"
        f" -> "
        f"{structure_summary['answer_correctness']:.2f}"
    )

    print(
        f"Groundedness: "
        f"{baseline_summary['groundedness']:.2f}"
        f" -> "
        f"{structure_summary['groundedness']:.2f}"
    )

    print(
        f"Completeness: "
        f"{baseline_summary['completeness']:.2f}"
        f" -> "
        f"{structure_summary['completeness']:.2f}"
    )

    print(
        f"Citation/source accuracy: "
        f"{baseline_summary['citation_source_accuracy']:.2f}"
        f" -> "
        f"{structure_summary['citation_source_accuracy']:.2f}"
    )

    print("\nSupporting Metrics")

    print(
        f"Reasoning accuracy: "
        f"{baseline_summary['reasoning_accuracy']:.4f}"
        f" -> "
        f"{structure_summary['reasoning_accuracy']:.4f}"
    )

    print(
        f"Hallucination rate: "
        f"{baseline_summary['hallucination_rate']:.4f}"
        f" -> "
        f"{structure_summary['hallucination_rate']:.4f}"
    )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nExperiment complete."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    run_experiment()