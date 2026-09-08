import json
import re
from pathlib import Path

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


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

STRUCTURE_VECTORSTORE = Path(
    "data/vectorstore/property_law_structure"
)

EMBEDDING_MODEL = "text-embedding-3-small"
ANSWER_MODEL = "gpt-5-mini"

TOP_K = 5
STRUCTURE_CANDIDATES = 50

REASONING_CORRECT_THRESHOLD = 4

EVALUATOR_RETRIES = 1


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
# VECTOR STORES
# ============================================================

baseline_vectorstore = Chroma(
    collection_name="property_law_baseline",
    persist_directory=str(BASELINE_VECTORSTORE),
    embedding_function=embeddings,
)

structure_vectorstore = Chroma(
    collection_name="property_law_structure",
    persist_directory=str(STRUCTURE_VECTORSTORE),
    embedding_function=embeddings,
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
        section == normalize_section(
            expected_section
        )
    )


# ============================================================
# BASELINE RETRIEVAL
# ============================================================

def retrieve_baseline(question):
    """
    Standard semantic retrieval.
    """

    results = baseline_vectorstore.similarity_search(
        question,
        k=TOP_K
    )

    return results


# ============================================================
# STRUCTURE SCORING
# ============================================================

def calculate_structure_score(
    document,
    question
):
    """
    Calculates the structural component of the
    structure-aware retrieval score.

    This follows the scoring weights used by the
    existing structure-aware retrieval implementation.
    """

    metadata = get_metadata(document)

    heading = str(
        metadata.get("heading", "")
    ).lower()

    subsection = str(
        metadata.get("subsection", "")
    ).lower()

    paragraph = str(
        metadata.get("paragraph", "")
    ).lower()

    nested = str(
        metadata.get("nested_paragraph", "")
    ).lower()

    hierarchy = str(
        metadata.get("hierarchy", "")
    )

    text = get_page_content(
        document
    ).lower()

    query = question.lower()

    # --------------------------------------------------------
    # Query token overlap
    # --------------------------------------------------------

    query_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            query
        )
    )

    def overlap(value):

        if not value:
            return 0.0

        value_words = set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                value
            )
        )

        if not query_words:
            return 0.0

        return (
            len(
                query_words.intersection(
                    value_words
                )
            )
            /
            len(query_words)
        )

    heading_score = overlap(
        heading
    )

    subsection_score = overlap(
        subsection
    )

    paragraph_score = overlap(
        paragraph
    )

    nested_score = overlap(
        nested
    )

    text_score = overlap(
        text
    )

    # --------------------------------------------------------
    # Hierarchy depth
    # --------------------------------------------------------

    hierarchy_parts = [
        part.strip()
        for part in hierarchy.split(">")
        if part.strip()
    ]

    depth = len(
        hierarchy_parts
    )

    if depth >= 4:
        hierarchy_score = 1.0

    elif depth == 3:
        hierarchy_score = 0.75

    elif depth == 2:
        hierarchy_score = 0.50

    elif depth == 1:
        hierarchy_score = 0.25

    else:
        hierarchy_score = 0.0

    # --------------------------------------------------------
    # Structure score
    # --------------------------------------------------------

    structure_score = (
        0.30 * heading_score
        + 0.10 * subsection_score
        + 0.10 * paragraph_score
        + 0.05 * nested_score
        + 0.30 * text_score
        + 0.15 * hierarchy_score
    )

    return structure_score


# ============================================================
# STRUCTURE-AWARE RETRIEVAL
# ============================================================

def retrieve_structure(question):
    """
    Structure-aware retrieval.

    Process:

    1. Retrieve semantic candidates using distances.
    2. Calculate semantic score.
    3. Calculate structure score.
    4. Combine semantic + structure scores.
    5. Remove duplicate Act/section results.
    6. Return top-k sections.
    """

    candidates = (
        structure_vectorstore
        .similarity_search_with_score(
            question,
            k=STRUCTURE_CANDIDATES
        )
    )

    scored = []

    for document, distance in candidates:

        semantic_score = (
            1.0
            /
            (
                1.0
                +
                float(distance)
            )
        )

        structure_score = (
            calculate_structure_score(
                document,
                question
            )
        )

        final_score = (
            0.65 * semantic_score
            +
            0.35 * structure_score
        )

        scored.append(
            (
                final_score,
                document
            )
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    selected = []

    seen_sections = set()

    for score, document in scored:

        metadata = get_metadata(
            document
        )

        act = normalize_act(
            metadata.get("act")
        )

        section = normalize_section(
            metadata.get("section")
        )

        key = (
            act,
            section
        )

        if key in seen_sections:
            continue

        seen_sections.add(
            key
        )

        selected.append(
            document
        )

        if len(selected) >= TOP_K:
            break

    return selected


# ============================================================
# CONTEXT
# ============================================================

def build_context(documents):

    context_parts = []

    for index, document in enumerate(
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

        text = get_page_content(
            document
        )

        context_parts.append(
            f"""
[SOURCE {index}]
Act: {act}
Section: {section}
Heading: {heading}

{text}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(
    question,
    documents
):
    """
    Generates an answer using ONLY the retrieved
    legal context.
    """

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
# ROBUST JSON PARSER
# ============================================================

def extract_json_object(raw_text):
    """
    Attempts to extract a JSON object from an LLM response.

    Handles:
    - plain JSON
    - ```json ... ```
    - surrounding explanatory text
    """

    if not raw_text:
        return None

    text = raw_text.strip()

    # --------------------------------------------------------
    # First attempt: direct JSON
    # --------------------------------------------------------

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # Remove Markdown code fences
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Find first JSON object in response
    # --------------------------------------------------------

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
                    return json.loads(
                        candidate
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

Evaluate the generated answer.

IMPORTANT RULES:

1. The answer must actually answer the question.

2. A refusal such as "the context is insufficient"
   is NOT a correct answer.

3. If the target section was not retrieved,
   reasoning_correct MUST be false.

4. Compare the answer against the ground-truth provision.

5. Check whether the legal information is correct.

6. Identify factual or legal claims made by the answer.

7. A claim is unsupported if it cannot be supported
   by the retrieved context.

8. Do not reward an answer simply because it is cautious.

Reasoning score:

5 = Completely correct and directly answers the question.
4 = Mostly correct with only minor omissions.
3 = Partially correct but missing important information.
2 = Mostly incorrect or materially incomplete.
1 = Incorrect or does not answer the question.

Return ONLY this JSON object:

{{
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
    """
    Evaluates:

    - whether the answer answers the question
    - reasoning correctness
    - unsupported claims
    - hallucination rate

    The evaluator is given one retry if its first response
    cannot be parsed as JSON.

    Evaluator parsing failures are recorded separately.
    They are NOT silently converted into a reasoning failure.
    """

    target_retrieved = any(
        document_is_relevant(
            document,
            expected_act,
            expected_section
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
    evaluation = None
    raw_response = ""

    # --------------------------------------------------------
    # First evaluation attempt
    # --------------------------------------------------------

    response = llm.invoke(
        prompt
    )

    raw_response = response.content.strip()

    evaluation = extract_json_object(
        raw_response
    )

    # --------------------------------------------------------
    # Retry if parsing failed
    # --------------------------------------------------------

    if evaluation is None:

        evaluator_retry_used = True

        retry_prompt = f"""
Your previous evaluation response could not be parsed.

Evaluate the following legal QA result again.

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

Return ONLY valid JSON.
Do not use Markdown.
Do not add explanations outside the JSON.

Use exactly these fields:

{{
    "reasoning_score": 1,
    "answers_question": false,
    "reasoning_correct": false,
    "total_claims": 0,
    "unsupported_claims": 0,
    "explanation": "brief explanation"
}}

Reasoning score:
5 = Completely correct.
4 = Mostly correct with minor omissions.
3 = Partially correct.
2 = Mostly incorrect or materially incomplete.
1 = Incorrect or does not answer.

A refusal caused by insufficient retrieved context is NOT
a correct answer.

If the target section was not retrieved,
reasoning_correct MUST be false.
"""

        retry_response = llm.invoke(
            retry_prompt
        )

        retry_raw = (
            retry_response.content.strip()
        )

        retry_evaluation = extract_json_object(
            retry_raw
        )

        if retry_evaluation is not None:

            evaluation = retry_evaluation
            raw_response = retry_raw

        else:

            evaluator_parse_failed = True
            raw_response = retry_raw

    # --------------------------------------------------------
    # Handle successful evaluation
    # --------------------------------------------------------

    if evaluation is not None:

        try:
            reasoning_score = int(
                evaluation.get(
                    "reasoning_score",
                    1
                )
            )
        except (
            TypeError,
            ValueError
        ):
            reasoning_score = 1

        try:
            answers_question = bool(
                evaluation.get(
                    "answers_question",
                    False
                )
            )
        except Exception:
            answers_question = False

        try:
            reasoning_correct = bool(
                evaluation.get(
                    "reasoning_correct",
                    False
                )
            )
        except Exception:
            reasoning_correct = False

        try:
            total_claims = int(
                evaluation.get(
                    "total_claims",
                    0
                )
            )
        except (
            TypeError,
            ValueError
        ):
            total_claims = 0

        try:
            unsupported_claims = int(
                evaluation.get(
                    "unsupported_claims",
                    0
                )
            )
        except (
            TypeError,
            ValueError
        ):
            unsupported_claims = 0

        explanation = str(
            evaluation.get(
                "explanation",
                ""
            )
        )

    # --------------------------------------------------------
    # Evaluator failed after retry
    # --------------------------------------------------------

    else:

        reasoning_score = None
        answers_question = None
        reasoning_correct = None
        total_claims = None
        unsupported_claims = None
        explanation = (
            "Evaluator response could not be parsed "
            "after retry."
        )

    # --------------------------------------------------------
    # Objective correction
    # --------------------------------------------------------

    if not evaluator_parse_failed:

        if reasoning_score < REASONING_CORRECT_THRESHOLD:
            reasoning_correct = False

        if not answers_question:
            reasoning_correct = False

        if not target_retrieved:
            reasoning_correct = False

    # --------------------------------------------------------
    # Hallucination rate
    # --------------------------------------------------------

    if (
        not evaluator_parse_failed
        and total_claims is not None
        and total_claims > 0
    ):

        hallucination_rate = (
            unsupported_claims
            /
            total_claims
        )

    else:

        hallucination_rate = None

    return {
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
    """
    Retrieves ALL documents from the existing baseline
    Chroma collection.

    The baseline corpus contains 1,154 usable documents.
    """

    collection_data = (
        baseline_vectorstore
        .get(
            include=[
                "documents",
                "metadatas"
            ]
        )
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

        metadata = get_metadata(
            document
        )

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

    relevant = sum(
        1
        for document in top_documents
        if document_is_relevant(
            document,
            expected_act,
            expected_section
        )
    )

    return relevant / len(
        top_documents
    )


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

    valid_values = [
        value
        for value in values
        if value is not None
    ]

    if not valid_values:
        return 0.0

    return sum(valid_values) / len(
        valid_values
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        dataset = json.load(f)

    questions = dataset[
        "questions"
    ]

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
    # Metric storage
    # --------------------------------------------------------

    baseline_p1 = []
    baseline_p5 = []
    baseline_r1 = []
    baseline_r5 = []
    baseline_reasoning = []
    baseline_hallucination = []

    structure_p1 = []
    structure_p5 = []
    structure_r1 = []
    structure_r5 = []
    structure_reasoning = []
    structure_hallucination = []

    results = []

    baseline_evaluator_failures = 0
    structure_evaluator_failures = 0

    # --------------------------------------------------------
    # Questions
    # --------------------------------------------------------

    for index, item in enumerate(
        questions,
        start=1
    ):

        question_id = item[
            "id"
        ]

        question = item[
            "question"
        ]

        expected_act = item[
            "act"
        ]

        expected_section = item[
            "section"
        ]

        print(
            f"\n[{index}/{len(questions)}] "
            f"Q{question_id}: {question}"
        )

        ground_truth_key = (
            normalize_act(
                expected_act
            ),
            normalize_section(
                expected_section
            )
        )

        ground_truth_document = (
            ground_truth_lookup.get(
                ground_truth_key
            )
        )

        # ====================================================
        # BASELINE
        # ====================================================

        baseline_documents = (
            retrieve_baseline(
                question
            )
        )

        baseline_p1_value = (
            precision_at_k(
                baseline_documents,
                expected_act,
                expected_section,
                1
            )
        )

        baseline_p5_value = (
            precision_at_k(
                baseline_documents,
                expected_act,
                expected_section,
                5
            )
        )

        baseline_r1_value = (
            recall_at_k(
                baseline_documents,
                expected_act,
                expected_section,
                1
            )
        )

        baseline_r5_value = (
            recall_at_k(
                baseline_documents,
                expected_act,
                expected_section,
                5
            )
        )

        print(
            f"  Baseline retrieval: "
            f"R@5={baseline_r5_value:.0f}"
        )

        baseline_answer = (
            generate_answer(
                question,
                baseline_documents
            )
        )

        baseline_evaluation = (
            evaluate_answer(
                question,
                expected_act,
                expected_section,
                baseline_answer,
                baseline_documents,
                ground_truth_document
            )
        )

        print(
            f"  Baseline reasoning: "
            f"{baseline_evaluation['reasoning_correct']}"
        )

        if baseline_evaluation[
            "evaluator_parse_failed"
        ]:

            baseline_evaluator_failures += 1

            print(
                "  WARNING: baseline evaluator "
                "parse failed after retry."
            )

        # ====================================================
        # STRUCTURE-AWARE
        # ====================================================

        structure_documents = (
            retrieve_structure(
                question
            )
        )

        structure_p1_value = (
            precision_at_k(
                structure_documents,
                expected_act,
                expected_section,
                1
            )
        )

        structure_p5_value = (
            precision_at_k(
                structure_documents,
                expected_act,
                expected_section,
                5
            )
        )

        structure_r1_value = (
            recall_at_k(
                structure_documents,
                expected_act,
                expected_section,
                1
            )
        )

        structure_r5_value = (
            recall_at_k(
                structure_documents,
                expected_act,
                expected_section,
                5
            )
        )

        print(
            f"  Structure retrieval: "
            f"R@5={structure_r5_value:.0f}"
        )

        structure_answer = (
            generate_answer(
                question,
                structure_documents
            )
        )

        structure_evaluation = (
            evaluate_answer(
                question,
                expected_act,
                expected_section,
                structure_answer,
                structure_documents,
                ground_truth_document
            )
        )

        print(
            f"  Structure reasoning: "
            f"{structure_evaluation['reasoning_correct']}"
        )

        if structure_evaluation[
            "evaluator_parse_failed"
        ]:

            structure_evaluator_failures += 1

            print(
                "  WARNING: structure evaluator "
                "parse failed after retry."
            )

        # ----------------------------------------------------
        # Store retrieval metrics
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

        # ----------------------------------------------------
        # Save question result
        # ----------------------------------------------------

        results.append(
            {
                "question_number":
                    question_id,

                "question":
                    question,

                "expected_act":
                    expected_act,

                "expected_section":
                    expected_section,

                "baseline":
                    {
                        "retrieval":
                            {
                                "precision_at_1":
                                    baseline_p1_value,

                                "precision_at_5":
                                    baseline_p5_value,

                                "recall_at_1":
                                    baseline_r1_value,

                                "recall_at_5":
                                    baseline_r5_value,

                                "hit_at_5":
                                    baseline_r5_value
                                    == 1.0,
                            },

                        "answer":
                            baseline_answer,

                        "qa_evaluation":
                            baseline_evaluation,
                    },

                "structure_aware":
                    {
                        "retrieval":
                            {
                                "precision_at_1":
                                    structure_p1_value,

                                "precision_at_5":
                                    structure_p5_value,

                                "recall_at_1":
                                    structure_r1_value,

                                "recall_at_5":
                                    structure_r5_value,

                                "hit_at_5":
                                    structure_r5_value
                                    == 1.0,
                            },

                        "answer":
                            structure_answer,

                        "qa_evaluation":
                            structure_evaluation,
                    },
            }
        )

    # ========================================================
    # FINAL SUMMARY
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

        "reasoning_accuracy":
            average(structure_reasoning),

        "hallucination_rate":
            average(structure_hallucination),
    }

    difference = {
        metric:
            structure_summary[metric]
            -
            baseline_summary[metric]

        for metric in baseline_summary
    }

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
            "within the top-k retrieved documents.",

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
            {
                "baseline":
                    baseline_evaluator_failures,

                "structure_aware":
                    structure_evaluator_failures,
            },

        "reasoning_evaluation_sample_size":
            {
                "baseline":
                    len(baseline_reasoning),

                "structure_aware":
                    len(structure_reasoning),
            },

        "hallucination_evaluation_sample_size":
            {
                "baseline":
                    len(baseline_hallucination),

                "structure_aware":
                    len(structure_hallucination),
            },

        "summary":
            {
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
    ) as f:

        json.dump(
            final_results,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(
        "\nMetric                         "
        "Baseline      Structure"
    )

    print("-" * 80)

    print(
        f"Precision@1                   "
        f"{baseline_summary['precision_at_1']:.4f}        "
        f"{structure_summary['precision_at_1']:.4f}"
    )

    print(
        f"Precision@5                   "
        f"{baseline_summary['precision_at_5']:.4f}        "
        f"{structure_summary['precision_at_5']:.4f}"
    )

    print(
        f"Recall@1                      "
        f"{baseline_summary['recall_at_1']:.4f}        "
        f"{structure_summary['recall_at_1']:.4f}"
    )

    print(
        f"Recall@5                      "
        f"{baseline_summary['recall_at_5']:.4f}        "
        f"{structure_summary['recall_at_5']:.4f}"
    )

    print(
        f"Reasoning accuracy            "
        f"{baseline_summary['reasoning_accuracy']:.4f}        "
        f"{structure_summary['reasoning_accuracy']:.4f}"
    )

    print(
        f"Hallucination rate            "
        f"{baseline_summary['hallucination_rate']:.4f}        "
        f"{structure_summary['hallucination_rate']:.4f}"
    )

    print("\nDifference (Structure - Baseline)")
    print("-" * 80)

    for metric, value in difference.items():

        print(
            f"{metric:30s} "
            f"{value:+.4f}"
        )

    print(
        "\nEvaluator parse failures:"
    )

    print(
        f"  Baseline: "
        f"{baseline_evaluator_failures}"
    )

    print(
        f"  Structure-aware: "
        f"{structure_evaluator_failures}"
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


if __name__ == "__main__":
    run_experiment()