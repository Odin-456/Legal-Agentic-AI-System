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
# BASELINE VECTOR STORE
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

    return aliases.get(
        act,
        act
    )


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

    metadata = get_metadata(
        document
    )

    act = normalize_act(
        metadata.get("act")
    )

    section = normalize_section(
        metadata.get("section")
    )

    return (
        act == normalize_act(
            expected_act
        )
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
# STRUCTURE-AWARE RETRIEVAL
# ============================================================

def retrieve_structure(question):
    """
    Uses the same structure-aware retrieval pipeline
    as the prototype legal agent.
    """

    results = structure_retriever.retrieve(
        question
    )

    documents = []

    context_documents = []

    for result in results:

        document = result.get(
            "document"
        )

        if document is not None:
            documents.append(
                document
            )

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
# ROBUST JSON PARSER
# ============================================================

def extract_json_object(raw_text):

    if not raw_text:
        return None

    text = raw_text.strip()

    try:
        return json.loads(
            text
        )

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
        return json.loads(
            text
        )

    except json.JSONDecodeError:
        pass

    start = text.find(
        "{"
    )

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
    # Successful evaluation
    # --------------------------------------------------------

    if evaluation is not None:

        try:
            answer_correctness = int(
                evaluation.get(
                    "answer_correctness",
                    0
                )
            )
        except (
            TypeError,
            ValueError
        ):
            answer_correctness = 0

        try:
            groundedness = int(
                evaluation.get(
                    "groundedness",
                    0
                )
            )
        except (
            TypeError,
            ValueError
        ):
            groundedness = 0

        try:
            completeness = int(
                evaluation.get(
                    "completeness",
                    0
                )
            )
        except (
            TypeError,
            ValueError
        ):
            completeness = 0

        try:
            citation_source_accuracy = int(
                evaluation.get(
                    "citation_source_accuracy",
                    0
                )
            )
        except (
            TypeError,
            ValueError
        ):
            citation_source_accuracy = 0

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

        answer_correctness = max(
            0,
            min(
                100,
                answer_correctness
            )
        )

        groundedness = max(
            0,
            min(
                100,
                groundedness
            )
        )

        completeness = max(
            0,
            min(
                100,
                completeness
            )
        )

        citation_source_accuracy = max(
            0,
            min(
                100,
                citation_source_accuracy
            )
        )

        reasoning_score = max(
            1,
            min(
                5,
                reasoning_score
            )
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

    # --------------------------------------------------------
    # Objective reasoning correction
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

    collection_data = (
        baseline_vectorstore.get(
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

    return sum(
        valid_values
    ) / len(
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

        dataset = json.load(
            f
        )

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
        "\nStructure-aware retrieval:"
    )

    print(
        f"  Top sections: {TOP_K}"
    )

    print(
        f"  Candidate chunks: "
        f"{structure_retriever.candidate_k}"
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

    baseline_p1 = []
    baseline_p5 = []
    baseline_r1 = []
    baseline_r5 = []

    baseline_answer_correctness = []
    baseline_groundedness = []
    baseline_completeness = []
    baseline_citation_source_accuracy = []

    baseline_reasoning = []
    baseline_hallucination = []

    structure_p1 = []
    structure_p5 = []
    structure_r1 = []
    structure_r5 = []

    structure_answer_correctness = []
    structure_groundedness = []
    structure_completeness = []
    structure_citation_source_accuracy = []

    structure_reasoning = []
    structure_hallucination = []

    results = []

    baseline_evaluator_failures = 0
    structure_evaluator_failures = 0

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
            f"  Baseline answer correctness: "
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

            baseline_evaluator_failures += 1

            print(
                "  WARNING: baseline evaluator "
                "parse failed after retry."
            )

        # ====================================================
        # STRUCTURE-AWARE
        # ====================================================

        (
            structure_documents,
            structure_context_documents
        ) = retrieve_structure(
            question
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
                structure_context_documents
            )
        )

        structure_evaluation = (
            evaluate_answer(
                question,
                expected_act,
                expected_section,
                structure_answer,
                structure_context_documents,
                ground_truth_document
            )
        )

        print(
            f"  Structure answer correctness: "
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

            structure_evaluator_failures += 1

            print(
                "  WARNING: structure evaluator "
                "parse failed after retry."
            )

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
            "answer_correctness"
        ] is not None:

            baseline_answer_correctness.append(
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

            baseline_citation_source_accuracy.append(
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
            "answer_correctness"
        ] is not None:

            structure_answer_correctness.append(
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

            structure_citation_source_accuracy.append(
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
            average(
                baseline_p1
            ),

        "precision_at_5":
            average(
                baseline_p5
            ),

        "recall_at_1":
            average(
                baseline_r1
            ),

        "recall_at_5":
            average(
                baseline_r5
            ),

        "answer_correctness":
            average(
                baseline_answer_correctness
            ),

        "groundedness":
            average(
                baseline_groundedness
            ),

        "completeness":
            average(
                baseline_completeness
            ),

        "citation_source_accuracy":
            average(
                baseline_citation_source_accuracy
            ),

        "reasoning_accuracy":
            average(
                baseline_reasoning
            ),

        "hallucination_rate":
            average(
                baseline_hallucination
            ),
    }

    structure_summary = {
        "precision_at_1":
            average(
                structure_p1
            ),

        "precision_at_5":
            average(
                structure_p5
            ),

        "recall_at_1":
            average(
                structure_r1
            ),

        "recall_at_5":
            average(
                structure_r5
            ),

        "answer_correctness":
            average(
                structure_answer_correctness
            ),

        "groundedness":
            average(
                structure_groundedness
            ),

        "completeness":
            average(
                structure_completeness
            ),

        "citation_source_accuracy":
            average(
                structure_citation_source_accuracy
            ),

        "reasoning_accuracy":
            average(
                structure_reasoning
            ),

        "hallucination_rate":
            average(
                structure_hallucination
            ),
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

        "structure_retrieval":
            {
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
            {
                "baseline":
                    baseline_evaluator_failures,

                "structure_aware":
                    structure_evaluator_failures,
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
        "\nRetrieval Metrics"
    )

    print(
        f"{'Metric':35s}"
        f"{'Baseline':15s}"
        f"{'Structure':15s}"
    )

    print("-" * 80)

    print(
        f"{'Precision@1':35s}"
        f"{baseline_summary['precision_at_1']:.4f}"
        f"{'':8s}"
        f"{structure_summary['precision_at_1']:.4f}"
    )

    print(
        f"{'Precision@5':35s}"
        f"{baseline_summary['precision_at_5']:.4f}"
        f"{'':8s}"
        f"{structure_summary['precision_at_5']:.4f}"
    )

    print(
        f"{'Recall@1':35s}"
        f"{baseline_summary['recall_at_1']:.4f}"
        f"{'':8s}"
        f"{structure_summary['recall_at_1']:.4f}"
    )

    print(
        f"{'Recall@5':35s}"
        f"{baseline_summary['recall_at_5']:.4f}"
        f"{'':8s}"
        f"{structure_summary['recall_at_5']:.4f}"
    )

    print(
        "\nAnswer Evaluation Metrics"
    )

    print(
        f"{'Metric':35s}"
        f"{'Baseline':15s}"
        f"{'Structure':15s}"
    )

    print("-" * 80)

    print(
        f"{'Answer Correctness (%)':35s}"
        f"{baseline_summary['answer_correctness']:.2f}"
        f"{'':8s}"
        f"{structure_summary['answer_correctness']:.2f}"
    )

    print(
        f"{'Groundedness (%)':35s}"
        f"{baseline_summary['groundedness']:.2f}"
        f"{'':8s}"
        f"{structure_summary['groundedness']:.2f}"
    )

    print(
        f"{'Completeness (%)':35s}"
        f"{baseline_summary['completeness']:.2f}"
        f"{'':8s}"
        f"{structure_summary['completeness']:.2f}"
    )

    print(
        f"{'Citation/Source Accuracy (%)':35s}"
        f"{baseline_summary['citation_source_accuracy']:.2f}"
        f"{'':8s}"
        f"{structure_summary['citation_source_accuracy']:.2f}"
    )

    print(
        "\nSupporting Metrics"
    )

    print(
        f"{'Metric':35s}"
        f"{'Baseline':15s}"
        f"{'Structure':15s}"
    )

    print("-" * 80)

    print(
        f"{'Reasoning Accuracy':35s}"
        f"{baseline_summary['reasoning_accuracy']:.4f}"
        f"{'':8s}"
        f"{structure_summary['reasoning_accuracy']:.4f}"
    )

    print(
        f"{'Hallucination Rate':35s}"
        f"{baseline_summary['hallucination_rate']:.4f}"
        f"{'':8s}"
        f"{structure_summary['hallucination_rate']:.4f}"
    )

    print(
        "\nDifference (Structure - Baseline)"
    )

    print("-" * 80)

    for metric, value in difference.items():

        print(
            f"{metric:35s}"
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

