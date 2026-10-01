from pathlib import Path
import json
import shutil
import sys
import re

from dotenv import load_dotenv

from langchain_openai import (
    OpenAIEmbeddings,
    ChatOpenAI,
)

from langchain_chroma import Chroma
from langchain_core.documents import Document

# ------------------------------------------------------------
# Allow imports from src/
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from property_law_parser import parse_property_law_corpus
from structure_aware_chunker import (
    parse_structure_corpus,
    remove_duplicates,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASELINE_VECTORSTORE = (
    PROJECT_ROOT
    / "data"
    / "vectorstore"
    / "property_law_baseline"
)

STRUCTURE_VECTORSTORE = (
    PROJECT_ROOT
    / "data"
    / "vectorstore"
    / "property_law_structure"
)

QUESTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "qa_evaluation_questions.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "qa_answer_evaluation_results.json"
)

EMBEDDING_MODEL = "text-embedding-3-small"

ANSWER_MODEL = "gpt-5-mini"

TOP_K_BASELINE = 5

STRUCTURE_CANDIDATES = 50

TOP_K_STRUCTURE = 5


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def reset_vectorstore(path):
    """
    Remove an existing Chroma vectorstore.
    """

    if path.exists():

        print()
        print(
            f"Removing existing vectorstore:"
        )
        print(
            f"  {path}"
        )

        shutil.rmtree(path)

        print(
            "  Removed."
        )


def normalise_text(text):
    """
    Normalise text for simple lexical comparisons.
    """

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def build_baseline_documents():
    """
    Build baseline LangChain Documents from the
    existing property-law parser.
    """

    sections = parse_property_law_corpus()

    documents = []

    for section in sections:

        text = section.get(
            "text",
            ""
        ).strip()

        if not text:
            continue

        act = section.get(
            "act",
            ""
        )

        section_number = section.get(
            "section",
            ""
        )

        heading = section.get(
            "heading",
            ""
        )

        page_content = (
            f"Act: {act}\n"
            f"Section: {section_number}\n"
            f"Heading: {heading}\n\n"
            f"{text}"
        )

        metadata = {
            "act": act,
            "section": section_number,
            "heading": heading,
            "source_type": section.get(
                "source_type",
                "NSW Property Law Legislation"
            ),
        }

        documents.append(
            Document(
                page_content=page_content,
                metadata=metadata
            )
        )

    return documents


def build_structure_documents():
    """
    Build structure-aware Documents using the
    existing structure_aware_chunker.py.

    This uses the ACTUAL function:
        parse_structure_corpus()
    """

    documents = parse_structure_corpus()

    documents = remove_duplicates(
        documents
    )

    return documents


# ============================================================
# STRUCTURE SCORING
# ============================================================

def lexical_score(query, value):
    """
    Simple lexical overlap score.
    """

    query_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            normalise_text(query)
        )
    )

    value_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            normalise_text(value)
        )
    )

    if not query_words or not value_words:
        return 0.0

    overlap = (
        query_words
        & value_words
    )

    return len(overlap) / len(
        query_words
    )


def calculate_structure_score(
    query,
    document
):
    """
    Calculate the same structure-aware score
    used by the retrieval evaluation.

    Components:

        Heading       0.30
        Subsection    0.10
        Paragraph     0.10
        Nested        0.05
        Text          0.30
        Hierarchy     0.15
    """

    metadata = document.metadata

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

    nested = metadata.get(
        "nested_paragraph",
        ""
    )

    text = document.page_content

    hierarchy = metadata.get(
        "hierarchy",
        ""
    )

    heading_score = lexical_score(
        query,
        heading
    )

    subsection_score = lexical_score(
        query,
        subsection
    )

    paragraph_score = lexical_score(
        query,
        paragraph
    )

    nested_score = lexical_score(
        query,
        nested
    )

    text_score = lexical_score(
        query,
        text
    )

    hierarchy_depth = 0

    if hierarchy:

        hierarchy_depth = (
            hierarchy.count(">")
            + 1
        )

    if hierarchy_depth >= 4:

        hierarchy_score = 1.0

    elif hierarchy_depth == 3:

        hierarchy_score = 0.75

    elif hierarchy_depth == 2:

        hierarchy_score = 0.50

    elif hierarchy_depth == 1:

        hierarchy_score = 0.25

    else:

        hierarchy_score = 0.0

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
# BASELINE RETRIEVAL
# ============================================================

def retrieve_baseline(
    vectorstore,
    query,
    k=5
):
    """
    Retrieve baseline documents using
    semantic similarity.
    """

    results = vectorstore.similarity_search_with_score(
        query,
        k=k
    )

    documents = []

    for document, distance in results:

        documents.append(
            {
                "document": document,
                "distance": float(distance),
            }
        )

    return documents


# ============================================================
# STRUCTURE-AWARE RETRIEVAL
# ============================================================

def retrieve_structure(
    vectorstore,
    query,
    candidates=50,
    k=5
):
    """
    Retrieve semantic candidates and rerank them
    using structure-aware scoring.
    """

    results = vectorstore.similarity_search_with_score(
        query,
        k=candidates
    )

    reranked = []

    for document, distance in results:

        distance = float(
            distance
        )

        semantic_score = (
            1.0
            / (
                1.0
                + distance
            )
        )

        structure_score = (
            calculate_structure_score(
                query,
                document
            )
        )

        final_score = (
            0.65 * semantic_score
            + 0.35 * structure_score
        )

        reranked.append(
            {
                "document": document,
                "distance": distance,
                "semantic_score": semantic_score,
                "structure_score": structure_score,
                "final_score": final_score,
            }
        )

    reranked.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    # --------------------------------------------------------
    # Keep strongest chunk from each section
    # --------------------------------------------------------

    selected = []

    seen_sections = set()

    for item in reranked:

        document = item["document"]

        act = document.metadata.get(
            "act",
            ""
        )

        section = document.metadata.get(
            "section",
            ""
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
            item
        )

        if len(selected) >= k:
            break

    return selected


# ============================================================
# CONTEXT BUILDING
# ============================================================

def build_context(
    results
):
    """
    Convert retrieval results into an LLM context.
    """

    context_parts = []

    for index, item in enumerate(
        results,
        start=1
    ):

        document = item["document"]

        metadata = document.metadata

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

        context_parts.append(
            f"[SOURCE {index}]\n"
            f"Act: {act}\n"
            f"Section: {section}\n"
            f"Heading: {heading}\n\n"
            f"{document.page_content}"
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# LLM ANSWER
# ============================================================

def answer_question(
    llm,
    question,
    context
):
    """
    Generate a grounded legal answer.
    """

    prompt = f"""
You are evaluating a legal question-answering
system for NSW property law.

Answer the question ONLY using the supplied
legislation context.

Do not use outside knowledge.

If the context does not contain enough information,
say that the supplied context is insufficient.

Give a concise but complete answer.

QUESTION:
{question}

LEGISLATION CONTEXT:
{context}
"""

    response = llm.invoke(
        prompt
    )

    return response.content.strip()


# ============================================================
# EVALUATOR
# ============================================================

def evaluate_answer(
    evaluator,
    question,
    answer,
    context,
    expected_act,
    expected_section
):
    """
    Ask an LLM evaluator to score the answer.

    Scores:

        correctness
        grounding
        source_correctness
        completeness
        unsupported_claims

    Each score is 1-5.
    """

    prompt = f"""
You are evaluating an automated legal QA system.

The answer must be judged ONLY against the
legislation context supplied below.

QUESTION:
{question}

EXPECTED ACT:
{expected_act}

EXPECTED SECTION:
{expected_section}

SYSTEM ANSWER:
{answer}

LEGISLATION CONTEXT:
{context}

Evaluate the answer on five dimensions.

1. correctness
Does the answer correctly answer the legal question?

2. grounding
Are the claims supported by the supplied legislation?

3. source_correctness
Does the answer rely on the correct Act and section?

4. completeness
Does the answer include the important legal conditions,
exceptions, requirements or limitations needed to answer
the question?

5. unsupported_claims
How well does the answer avoid claims that are not
supported by the supplied legislation?

Use a 1-5 scale.

Return ONLY valid JSON in exactly this format:

{{
  "correctness": 1,
  "grounding": 1,
  "source_correctness": 1,
  "completeness": 1,
  "unsupported_claims": 1,
  "reason": "brief explanation"
}}

Do not include markdown.
"""

    response = evaluator.invoke(
        prompt
    )

    content = response.content.strip()

    # --------------------------------------------------------
    # Remove accidental markdown fences
    # --------------------------------------------------------

    if content.startswith(
        "```"
    ):

        content = re.sub(
            r"^```(?:json)?\s*",
            "",
            content
        )

        content = re.sub(
            r"\s*```$",
            "",
            content
        )

    try:

        return json.loads(
            content
        )

    except json.JSONDecodeError:

        return {
            "correctness": None,
            "grounding": None,
            "source_correctness": None,
            "completeness": None,
            "unsupported_claims": None,
            "reason": (
                "Evaluator returned invalid JSON: "
                + content
            ),
        }


# ============================================================
# QUESTION LOADING
# ============================================================
def load_questions():
    """
    Load QA evaluation questions.

    Supports both:
        1. A JSON list of question dictionaries
        2. A JSON dictionary containing question dictionaries
    """

    with QUESTIONS_FILE.open(
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    # --------------------------------------------------------
    # JSON list
    # --------------------------------------------------------

    if isinstance(data, list):

        return data

    # --------------------------------------------------------
    # JSON dictionary
    # --------------------------------------------------------

    if isinstance(data, dict):

        # If the file contains a wrapper such as:
        #
        # {
        #     "questions": [...]
        # }
        #
        # use the questions list directly.

        if isinstance(
            data.get("questions"),
            list
        ):

            return data["questions"]

        # Otherwise assume the dictionary values
        # are the individual question records.

        values = list(
            data.values()
        )

        if all(
            isinstance(item, dict)
            for item in values
        ):

            return values

    raise ValueError(
        "Unsupported evaluation question JSON format."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 80
    )

    print(
        "PROPERTY LAW QA ANSWER EVALUATION"
    )

    print(
        "=" * 80
    )

    print()

    # --------------------------------------------------------
    # Load questions
    # --------------------------------------------------------

    questions = load_questions()

    print(
        f"Evaluation questions: "
        f"{len(questions)}"
    )

    print(
        f"Embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    print()

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    print(
        "Creating embedding model..."
    )

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    # --------------------------------------------------------
    # Reset vectorstores
    # --------------------------------------------------------

    reset_vectorstore(
        BASELINE_VECTORSTORE
    )

    reset_vectorstore(
        STRUCTURE_VECTORSTORE
    )

    # --------------------------------------------------------
    # Build baseline documents
    # --------------------------------------------------------

    print()
    print(
        "Building baseline documents..."
    )

    baseline_documents = (
        build_baseline_documents()
    )

    print(
        f"Baseline documents: "
        f"{len(baseline_documents)}"
    )

    # --------------------------------------------------------
    # Build structure documents
    # --------------------------------------------------------

    print()
    print(
        "Building structure-aware documents..."
    )

    structure_documents = (
        build_structure_documents()
    )

    print(
        f"Structure-aware documents: "
        f"{len(structure_documents)}"
    )

    # --------------------------------------------------------
    # Create Chroma stores
    # --------------------------------------------------------

    print()
    print(
        "Creating baseline vectorstore..."
    )

    baseline_vectorstore = Chroma.from_documents(
        documents=baseline_documents,
        embedding=embeddings,
        collection_name="property_law_baseline",
        persist_directory=str(
            BASELINE_VECTORSTORE
        ),
    )

    print(
        "Baseline vectorstore created."
    )

    print()
    print(
        "Creating structure-aware vectorstore..."
    )

    structure_vectorstore = Chroma.from_documents(
        documents=structure_documents,
        embedding=embeddings,
        collection_name="property_law_structure",
        persist_directory=str(
            STRUCTURE_VECTORSTORE
        ),
    )

    print(
        "Structure-aware vectorstore created."
    )

    # --------------------------------------------------------
    # Create LLMs
    # --------------------------------------------------------

    print()
    print(
        f"Creating answer model: "
        f"{ANSWER_MODEL}"
    )

    answer_llm = ChatOpenAI(
        model=ANSWER_MODEL,
        temperature=0
    )

    print(
        "Creating evaluator model..."
    )

    evaluator_llm = ChatOpenAI(
        model=ANSWER_MODEL,
        temperature=0
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    results = []

    print()
    print(
        "=" * 80
    )

    print(
        "STARTING QA EVALUATION"
    )

    print(
        "=" * 80
    )

    for index, item in enumerate(
        questions,
        start=1
    ):

        question = item.get(
            "question",
            ""
        )

        expected_act = item.get(
            "act",
            ""
        )

        expected_section = str(
            item.get(
                "section",
                ""
            )
        )

        print()
        print(
            "-" * 80
        )

        print(
            f"Question {index}/{len(questions)}"
        )

        print(
            question
        )

        print(
            f"Expected: "
            f"{expected_act}, "
            f"Section {expected_section}"
        )

        # ----------------------------------------------------
        # Baseline retrieval
        # ----------------------------------------------------

        baseline_results = (
            retrieve_baseline(
                baseline_vectorstore,
                question,
                k=TOP_K_BASELINE
            )
        )

        baseline_context = (
            build_context(
                baseline_results
            )
        )

        # ----------------------------------------------------
        # Structure-aware retrieval
        # ----------------------------------------------------

        structure_results = (
            retrieve_structure(
                structure_vectorstore,
                question,
                candidates=STRUCTURE_CANDIDATES,
                k=TOP_K_STRUCTURE
            )
        )

        structure_context = (
            build_context(
                structure_results
            )
        )

        # ----------------------------------------------------
        # Generate answers
        # ----------------------------------------------------

        print(
            "  Generating baseline answer..."
        )

        baseline_answer = (
            answer_question(
                answer_llm,
                question,
                baseline_context
            )
        )

        print(
            "  Generating structure-aware answer..."
        )

        structure_answer = (
            answer_question(
                answer_llm,
                question,
                structure_context
            )
        )

        # ----------------------------------------------------
        # Evaluate answers
        # ----------------------------------------------------

        print(
            "  Evaluating baseline answer..."
        )

        baseline_evaluation = (
            evaluate_answer(
                evaluator_llm,
                question,
                baseline_answer,
                baseline_context,
                expected_act,
                expected_section
            )
        )

        print(
            "  Evaluating structure-aware answer..."
        )

        structure_evaluation = (
            evaluate_answer(
                evaluator_llm,
                question,
                structure_answer,
                structure_context,
                expected_act,
                expected_section
            )
        )

        # ----------------------------------------------------
        # Save question result
        # ----------------------------------------------------

        question_result = {
            "question_number": index,
            "question": question,
            "expected_act": expected_act,
            "expected_section": expected_section,

            "baseline": {
                "answer": baseline_answer,
                "retrieved_sources": [
                    {
                        "act": r["document"].metadata.get(
                            "act",
                            ""
                        ),
                        "section": r["document"].metadata.get(
                            "section",
                            ""
                        ),
                        "heading": r["document"].metadata.get(
                            "heading",
                            ""
                        ),
                        "distance": r["distance"],
                    }
                    for r in baseline_results
                ],
                "evaluation": baseline_evaluation,
            },

            "structure_aware": {
                "answer": structure_answer,
                "retrieved_sources": [
                    {
                        "act": r["document"].metadata.get(
                            "act",
                            ""
                        ),
                        "section": r["document"].metadata.get(
                            "section",
                            ""
                        ),
                        "heading": r["document"].metadata.get(
                            "heading",
                            ""
                        ),
                        "distance": r["distance"],
                        "semantic_score": r.get(
                            "semantic_score"
                        ),
                        "structure_score": r.get(
                            "structure_score"
                        ),
                        "final_score": r.get(
                            "final_score"
                        ),
                    }
                    for r in structure_results
                ],
                "evaluation": structure_evaluation,
            },
        }

        results.append(
            question_result
        )

        # ----------------------------------------------------
        # Print quick result
        # ----------------------------------------------------

        print()

        print(
            "  Baseline scores:"
        )

        print(
            baseline_evaluation
        )

        print(
            "  Structure-aware scores:"
        )

        print(
            structure_evaluation
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    def average_score(
        system_name,
        metric
    ):

        values = []

        for result in results:

            evaluation = (
                result[system_name][
                    "evaluation"
                ]
            )

            value = evaluation.get(
                metric
            )

            if isinstance(
                value,
                (int, float)
            ):

                values.append(
                    value
                )

        if not values:
            return None

        return sum(values) / len(
            values
        )

    metrics = [
        "correctness",
        "grounding",
        "source_correctness",
        "completeness",
        "unsupported_claims",
    ]

    summary = {}

    for system_name in [
        "baseline",
        "structure_aware"
    ]:

        summary[
            system_name
        ] = {}

        for metric in metrics:

            summary[
                system_name
            ][metric] = average_score(
                system_name,
                metric
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "evaluation_type": (
            "QA answer evaluation"
        ),
        "embedding_model": (
            EMBEDDING_MODEL
        ),
        "answer_model": (
            ANSWER_MODEL
        ),
        "total_questions": (
            len(questions)
        ),
        "summary": summary,
        "results": results,
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print()
    print()
    print(
        "=" * 80
    )

    print(
        "FINAL QA ANSWER EVALUATION RESULTS"
    )

    print(
        "=" * 80
    )

    print()

    print(
        f"Total questions: "
        f"{len(questions)}"
    )

    print()

    print(
        "AVERAGE SCORES (1-5)"
    )

    print()

    print(
        f"{'Metric':<25}"
        f"{'Baseline':>15}"
        f"{'Structure-aware':>20}"
        f"{'Difference':>15}"
    )

    print(
        "-" * 75
    )

    for metric in metrics:

        baseline = summary[
            "baseline"
        ][metric]

        structure = summary[
            "structure_aware"
        ][metric]

        if (
            baseline is None
            or structure is None
        ):

            print(
                f"{metric:<25}"
                f"{'N/A':>15}"
                f"{'N/A':>20}"
                f"{'N/A':>15}"
            )

            continue

        difference = (
            structure
            - baseline
        )

        print(
            f"{metric:<25}"
            f"{baseline:>15.3f}"
            f"{structure:>20.3f}"
            f"{difference:>15.3f}"
        )

    print()

    print(
        "=" * 80
    )

    print(
        "Results saved to:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        "=" * 80
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()