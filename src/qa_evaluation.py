from pathlib import Path
import json
import shutil
import re

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from property_law_parser import parse_property_law_corpus
from property_law_chunker import create_chunks
from structure_aware_chunker import (
    parse_structure_corpus,
    remove_duplicates,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# PATHS
# ============================================================

BASELINE_VECTORSTORE = Path(
    "data/vectorstore/property_law_baseline"
)

STRUCTURE_VECTORSTORE = Path(
    "data/vectorstore/property_law_structure"
)

EVALUATION_FILE = Path(
    "data/evaluation/evaluation_questions_60.json"
)

RESULTS_FILE = Path(
    "data/evaluation/retrieval_evaluation_results.json"
)


# ============================================================
# CHROMA COLLECTIONS
# ============================================================

BASELINE_COLLECTION = "property_law_baseline"

STRUCTURE_COLLECTION = "property_law_structure"


# ============================================================
# EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL = "text-embedding-3-small"


# ============================================================
# RETRIEVAL SETTINGS
# ============================================================

TOP_K = 5

STRUCTURE_CANDIDATE_K = 50


# ============================================================
# STOP WORDS
# ============================================================

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "may",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "these",
    "this",
    "to",
    "under",
    "what",
    "when",
    "which",
    "who",
    "with",
}


# ============================================================
# LOAD EVALUATION QUESTIONS
# ============================================================

def load_evaluation_questions():

    if not EVALUATION_FILE.exists():

        raise FileNotFoundError(
            f"Evaluation file not found:\n"
            f"{EVALUATION_FILE}"
        )

    with EVALUATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    questions = data["questions"]

    return questions


# ============================================================
# QUERY TERMS
# ============================================================

def get_query_terms(query):

    words = re.findall(
        r"[A-Za-z0-9]+",
        query.lower(),
    )

    return {
        word
        for word in words
        if word not in STOP_WORDS
    }


# ============================================================
# TEXT MATCH SCORE
# ============================================================

def text_match_score(query_terms, text):

    if not query_terms:
        return 0.0

    text_lower = text.lower()

    matches = 0

    for term in query_terms:

        if term in text_lower:
            matches += 1

    return matches / len(query_terms)


# ============================================================
# STRUCTURE SCORE
# ============================================================

def calculate_structure_score(
    query,
    document,
):

    query_terms = get_query_terms(query)

    metadata = document.metadata

    heading = str(
        metadata.get("heading", "")
    )

    subsection = str(
        metadata.get("subsection", "")
    )

    paragraph = str(
        metadata.get("paragraph", "")
    )

    nested_paragraph = str(
        metadata.get("nested_paragraph", "")
    )

    hierarchy = str(
        metadata.get("hierarchy", "")
    )

    text = document.page_content

    # --------------------------------------------------------
    # Heading relevance
    # --------------------------------------------------------

    heading_score = text_match_score(
        query_terms,
        heading,
    )

    # --------------------------------------------------------
    # Structural metadata relevance
    # --------------------------------------------------------

    subsection_score = 0.0

    if subsection:
        subsection_score = text_match_score(
            query_terms,
            subsection,
        )

    paragraph_score = 0.0

    if paragraph:
        paragraph_score = text_match_score(
            query_terms,
            paragraph,
        )

    nested_score = 0.0

    if nested_paragraph:
        nested_score = text_match_score(
            query_terms,
            nested_paragraph,
        )

    # --------------------------------------------------------
    # Legal text relevance
    # --------------------------------------------------------

    text_score = text_match_score(
        query_terms,
        text,
    )

    # --------------------------------------------------------
    # Hierarchy score
    # --------------------------------------------------------

    hierarchy_depth = len(
        hierarchy.split(">")
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

    # --------------------------------------------------------
    # Combined structural score
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
# SEMANTIC SCORE
# ============================================================

def calculate_semantic_score(distance):

    # Chroma distance:
    #
    # lower distance = more similar
    #
    # Convert to a bounded score where:
    #
    # smaller distance -> higher score

    return 1.0 / (
        1.0 + float(distance)
    )


# ============================================================
# FINAL STRUCTURE-AWARE SCORE
# ============================================================

def calculate_final_score(
    query,
    document,
    distance,
):

    semantic_score = calculate_semantic_score(
        distance
    )

    structure_score = calculate_structure_score(
        query,
        document,
    )

    final_score = (
        0.65 * semantic_score
        + 0.35 * structure_score
    )

    return (
        semantic_score,
        structure_score,
        final_score,
    )


# ============================================================
# GROUP RESULTS BY ACT + SECTION
# ============================================================

def group_by_section(
    results,
    top_k=5,
):

    grouped = {}

    for document, distance in results:

        act = document.metadata.get(
            "act",
            "",
        )

        section = document.metadata.get(
            "section",
            "",
        )

        key = (
            act,
            section,
        )

        if key not in grouped:

            grouped[key] = []

        grouped[key].append(
            (
                document,
                distance,
            )
        )

    # --------------------------------------------------------
    # Keep strongest chunk from each section
    # --------------------------------------------------------

    section_results = []

    for key, documents in grouped.items():

        documents.sort(
            key=lambda item: item[1]
        )

        section_results.append(
            documents[0]
        )

    # --------------------------------------------------------
    # Sort by semantic distance
    # --------------------------------------------------------

    section_results.sort(
        key=lambda item: item[1]
    )

    return section_results[:top_k]


# ============================================================
# STRUCTURE-AWARE RETRIEVAL
# ============================================================

def structure_aware_retrieve(
    vectorstore,
    query,
    candidate_k=50,
    top_k=5,
):

    candidates = vectorstore.similarity_search_with_score(
        query,
        k=candidate_k,
    )

    reranked = []

    for document, distance in candidates:

        (
            semantic_score,
            structure_score,
            final_score,
        ) = calculate_final_score(
            query,
            document,
            distance,
        )

        reranked.append(
            (
                document,
                distance,
                semantic_score,
                structure_score,
                final_score,
            )
        )

    # --------------------------------------------------------
    # Sort by final score
    # --------------------------------------------------------

    reranked.sort(
        key=lambda item: item[4],
        reverse=True,
    )

    # --------------------------------------------------------
    # Group by Act + Section
    # --------------------------------------------------------

    grouped = {}

    for item in reranked:

        document = item[0]

        act = document.metadata.get(
            "act",
            "",
        )

        section = document.metadata.get(
            "section",
            "",
        )

        key = (
            act,
            section,
        )

        if key not in grouped:

            grouped[key] = item

    final_results = list(
        grouped.values()
    )[:top_k]

    return final_results


# ============================================================
# FIND SECTION RANK
# ============================================================

def find_section_rank(
    results,
    expected_act,
    expected_section,
):

    for rank, item in enumerate(
        results,
        start=1,
    ):

        if isinstance(item, tuple):

            document = item[0]

        else:

            document = item

        act = document.metadata.get(
            "act",
            "",
        )

        section = str(
            document.metadata.get(
                "section",
                "",
            )
        )

        if (
            act == expected_act
            and section == str(expected_section)
        ):

            return rank

    return None


# ============================================================
# METRIC: TOP-K HIT
# ============================================================

def calculate_hit_rate(
    ranks,
    k,
):

    hits = sum(
        1
        for rank in ranks
        if rank is not None
        and rank <= k
    )

    return hits / len(ranks)


# ============================================================
# METRIC: MRR
# ============================================================

def calculate_mrr(ranks):

    reciprocal_ranks = []

    for rank in ranks:

        if rank is None:

            reciprocal_ranks.append(0.0)

        else:

            reciprocal_ranks.append(
                1.0 / rank
            )

    return sum(
        reciprocal_ranks
    ) / len(reciprocal_ranks)


# ============================================================
# DELETE VECTORSTORE
# ============================================================

def reset_vectorstore(path):

    if path.exists():

        print()
        print(
            f"Removing existing vectorstore:"
        )

        print(
            f"  {path}"
        )

        shutil.rmtree(path)

        print("  Removed.")


# ============================================================
# BUILD BASELINE VECTORSTORE
# ============================================================

def build_baseline_vectorstore(
    embeddings,
):

    print()
    print("=" * 80)
    print("BUILDING BASELINE DOCUMENTS")
    print("=" * 80)

    sections = parse_property_law_corpus()

    print(
        f"Sections parsed: {len(sections)}"
    )

    documents = create_chunks(
        sections
    )

    print(
        f"Baseline retrieval documents: "
        f"{len(documents)}"
    )

    print()
    print(
        "Creating vectorstore: "
        "property_law_baseline"
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=BASELINE_COLLECTION,
        persist_directory=str(
            BASELINE_VECTORSTORE
        ),
    )

    print(
        "Vectorstore created successfully."
    )

    return vectorstore


# ============================================================
# BUILD STRUCTURE VECTORSTORE
# ============================================================

def build_structure_vectorstore(
    embeddings,
):

    print()
    print("=" * 80)
    print("BUILDING STRUCTURE-AWARE DOCUMENTS")
    print("=" * 80)

    documents = parse_structure_corpus()

    print(
        f"Structure chunks created: "
        f"{len(documents)}"
    )

    documents = remove_duplicates(
        documents
    )

    print(
        f"Unique structure chunks: "
        f"{len(documents)}"
    )

    print()
    print(
        "Creating vectorstore: "
        "property_law_structure"
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=STRUCTURE_COLLECTION,
        persist_directory=str(
            STRUCTURE_VECTORSTORE
        ),
    )

    print(
        "Vectorstore created successfully."
    )

    return vectorstore


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    baseline_vectorstore,
    structure_vectorstore,
    questions,
):

    baseline_ranks = []
    structure_ranks = []

    detailed_results = []

    print()
    print("=" * 80)
    print("RUNNING EVALUATION")
    print("=" * 80)

    for index, item in enumerate(
        questions,
        start=1,
    ):

        question = item["question"]

        expected_act = item[
            "expected_act"
        ]

        expected_section = str(
            item["expected_section"]
        )

        print()
        print(
            f"[{index}/{len(questions)}] "
            f"{question}"
        )

        # ----------------------------------------------------
        # Baseline
        # ----------------------------------------------------

        baseline_results = (
            baseline_vectorstore
            .similarity_search_with_score(
                question,
                k=TOP_K,
            )
        )

        baseline_rank = find_section_rank(
            baseline_results,
            expected_act,
            expected_section,
        )

        # ----------------------------------------------------
        # Structure-aware
        # ----------------------------------------------------

        structure_results = (
            structure_aware_retrieve(
                structure_vectorstore,
                question,
                candidate_k=STRUCTURE_CANDIDATE_K,
                top_k=TOP_K,
            )
        )

        structure_rank = find_section_rank(
            structure_results,
            expected_act,
            expected_section,
        )

        baseline_ranks.append(
            baseline_rank
        )

        structure_ranks.append(
            structure_rank
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        baseline_display = (
            baseline_rank
            if baseline_rank is not None
            else "N/A"
        )

        structure_display = (
            structure_rank
            if structure_rank is not None
            else "N/A"
        )

        print(
            f"    Baseline rank: "
            f"{baseline_display}"
        )

        print(
            f"    Structure-aware rank: "
            f"{structure_display}"
        )

        # ----------------------------------------------------
        # Improvement
        # ----------------------------------------------------

        if (
            baseline_rank is not None
            and structure_rank is not None
        ):

            if structure_rank < baseline_rank:
                comparison = "improved"

            elif structure_rank > baseline_rank:
                comparison = "worsened"

            else:
                comparison = "same"

        elif (
            baseline_rank is None
            and structure_rank is not None
        ):

            comparison = "structure_found"

        elif (
            baseline_rank is not None
            and structure_rank is None
        ):

            comparison = "structure_lost"

        else:

            comparison = "both_not_found"

        detailed_results.append(
            {
                "id": item["id"],
                "question": question,
                "expected_act": expected_act,
                "expected_section": expected_section,
                "expected_heading": item.get(
                    "expected_heading",
                    "",
                ),
                "baseline_rank": baseline_rank,
                "structure_aware_rank": structure_rank,
                "comparison": comparison,
            }
        )

    return (
        baseline_ranks,
        structure_ranks,
        detailed_results,
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    questions,
    baseline_ranks,
    structure_ranks,
    detailed_results,
):

    baseline_top1 = calculate_hit_rate(
        baseline_ranks,
        1,
    )

    structure_top1 = calculate_hit_rate(
        structure_ranks,
        1,
    )

    baseline_top5 = calculate_hit_rate(
        baseline_ranks,
        5,
    )

    structure_top5 = calculate_hit_rate(
        structure_ranks,
        5,
    )

    baseline_mrr = calculate_mrr(
        baseline_ranks
    )

    structure_mrr = calculate_mrr(
        structure_ranks
    )

    results = {

        "evaluation": {
            "total_questions": len(
                questions
            ),
            "embedding_model": EMBEDDING_MODEL,
            "baseline_top_k": TOP_K,
            "structure_candidate_k": (
                STRUCTURE_CANDIDATE_K
            ),
            "structure_final_k": TOP_K,
        },

        "metrics": {

            "top_1": {
                "baseline": baseline_top1,
                "structure_aware": structure_top1,
                "difference": (
                    structure_top1
                    - baseline_top1
                ),
            },

            "top_5": {
                "baseline": baseline_top5,
                "structure_aware": structure_top5,
                "difference": (
                    structure_top5
                    - baseline_top5
                ),
            },

            "mrr": {
                "baseline": baseline_mrr,
                "structure_aware": structure_mrr,
                "difference": (
                    structure_mrr
                    - baseline_mrr
                ),
            },
        },

        "question_results": detailed_results,
    }

    with RESULTS_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

def print_final_results(
    questions,
    baseline_ranks,
    structure_ranks,
):

    baseline_top1 = calculate_hit_rate(
        baseline_ranks,
        1,
    )

    structure_top1 = calculate_hit_rate(
        structure_ranks,
        1,
    )

    baseline_top5 = calculate_hit_rate(
        baseline_ranks,
        5,
    )

    structure_top5 = calculate_hit_rate(
        structure_ranks,
        5,
    )

    baseline_mrr = calculate_mrr(
        baseline_ranks
    )

    structure_mrr = calculate_mrr(
        structure_ranks
    )

    print()
    print("=" * 80)
    print("FINAL RETRIEVAL EVALUATION RESULTS")
    print("=" * 80)

    print()
    print(
        f"Total questions: "
        f"{len(questions)}"
    )

    print()
    print(
        "TOP-1 SECTION RETRIEVAL"
    )

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
        f"{(structure_top1 - baseline_top1):+.4f} "
        f"({(structure_top1 - baseline_top1) * 100:+.2f}%)"
    )

    print()
    print(
        "TOP-5 SECTION RETRIEVAL"
    )

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
        f"{(structure_top5 - baseline_top5):+.4f} "
        f"({(structure_top5 - baseline_top5) * 100:+.2f}%)"
    )

    print()
    print(
        "MEAN RECIPROCAL RANK (MRR)"
    )

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
        f"{(structure_mrr - baseline_mrr):+.4f}"
    )

    print()
    print(
        f"Results saved to:"
    )

    print(
        f"  {RESULTS_FILE}"
    )

    print()
    print(
        "Evaluation complete."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("PROPERTY LAW RETRIEVAL EVALUATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Load questions
    # --------------------------------------------------------

    questions = load_evaluation_questions()

    print()
    print(
        f"Evaluation questions: "
        f"{len(questions)}"
    )

    print(
        f"Embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    print()
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
    # Build vectorstores
    # --------------------------------------------------------

    baseline_vectorstore = (
        build_baseline_vectorstore(
            embeddings
        )
    )

    structure_vectorstore = (
        build_structure_vectorstore(
            embeddings
        )
    )

    # --------------------------------------------------------
    # Run evaluation
    # --------------------------------------------------------

    (
        baseline_ranks,
        structure_ranks,
        detailed_results,
    ) = evaluate(
        baseline_vectorstore,
        structure_vectorstore,
        questions,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        questions,
        baseline_ranks,
        structure_ranks,
        detailed_results,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_final_results(
        questions,
        baseline_ranks,
        structure_ranks,
    )


if __name__ == "__main__":
    main()