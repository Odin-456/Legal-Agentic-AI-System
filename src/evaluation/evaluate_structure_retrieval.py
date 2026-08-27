import json
import sys
from pathlib import Path

# ============================================================
# ALLOW IMPORTS FROM src/
# ============================================================

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from structure_retrieval import StructureAwareRetriever


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "BAAI/bge-base-en-v1.5"

TOP_K_VALUES = [1, 3, 5, 10]

OUTPUT_PATH = Path(
    "data/evaluation/structure_retrieval_results.json"
)


# ============================================================
# TEST DATASET
# ============================================================

TEST_QUERIES = [

    {
        "query": "What is the definition of agricultural land under 7 CFR part 781?",
        "expected_citations": ["7 CFR 781.2"],
    },

    {
        "query": "Who qualifies as a foreign person under 7 CFR 781.2?",
        "expected_citations": ["7 CFR 781.2"],
    },

    {
        "query": "What information must a foreign person report under 7 CFR 781.3(f)(1)?",
        "expected_citations": ["7 CFR 781.3"],
    },

    {
        "query": "What are the reporting requirements for foreign investment in agricultural land?",
        "expected_citations": ["7 CFR 781.3"],
    },

    {
        "query": "What information must be reported about foreign individuals or governments holding significant interest or substantial control?",
        "expected_citations": ["7 CFR 781.3"],
    },

    {
        "query": "What is the purpose of the regulations concerning foreign investment in agricultural land?",
        "expected_citations": ["7 CFR 781.1"],
    },

    {
        "query": "What is AFIDA under 7 CFR 781.2?",
        "expected_citations": ["7 CFR 781.2"],
    },

    {
        "query": "What does agricultural land include under the regulations?",
        "expected_citations": ["7 CFR 781.2"],
    },

    {
        "query": "What information must be provided about agricultural land in a report?",
        "expected_citations": ["7 CFR 781.3"],
    },

    {
        "query": "When must a foreign person file a report concerning agricultural land?",
        "expected_citations": ["7 CFR 781.3"],
    },

    {
        "query": "What happens when land becomes agricultural land after being acquired?",
        "expected_citations": ["7 CFR 781.3"],
    },

    {
        "query": "What information is required concerning the location and acreage of agricultural land?",
        "expected_citations": ["7 CFR 781.3"],
    },

    {
        "query": "What is the legal name and address information required for foreign persons?",
        "expected_citations": ["7 CFR 781.3"],
    },

    {
        "query": "What information must be reported concerning the use of agricultural land?",
        "expected_citations": ["7 CFR 781.3"],
    },

    {
        "query": "What is a foreign individual under 7 CFR 781.2?",
        "expected_citations": ["7 CFR 781.2"],
    },

    {
        "query": "What is the meaning of any interest under 7 CFR 781.2?",
        "expected_citations": ["7 CFR 781.2"],
    },

    {
        "query": "What are the requirements for reporting foreign ownership of agricultural land?",
        "expected_citations": ["7 CFR 781.3"],
    },

    {
        "query": "Which regulation implements the Agricultural Foreign Investment Disclosure Act?",
        "expected_citations": ["7 CFR 781.1"],
    },

    {
        "query": "What form must a foreign person file under the reporting requirements?",
        "expected_citations": ["7 CFR 781.3"],
    },

    {
        "query": "What agency receives reports concerning foreign investment in agricultural land?",
        "expected_citations": ["7 CFR 781.3"],
    },
]


# ============================================================
# NORMALIZE CITATION
# ============================================================

def normalize_citation(citation):
    """Normalize citation strings."""

    if not citation:
        return ""

    citation = str(citation).strip()

    while citation.endswith((".", ",", ";", ":")):
        citation = citation[:-1]

    return citation


# ============================================================
# GET CITATION
# ============================================================

def get_result_citation(result):
    """Extract citation from retrieval result."""

    metadata = result.get(
        "metadata",
        {}
    )

    return normalize_citation(
        metadata.get(
            "citation",
            ""
        )
    )


# ============================================================
# CHECK RELEVANCE
# ============================================================

def is_relevant(
    result,
    expected_citations
):
    """Return True if the result citation matches."""

    result_citation = get_result_citation(
        result
    )

    expected = [
        normalize_citation(citation)
        for citation in expected_citations
    ]

    return result_citation in expected


# ============================================================
# RECALL@K
# ============================================================

def calculate_recall_at_k(
    results,
    expected_citations,
    k
):
    """Return 1 if expected citation appears within top-k."""

    for result in results[:k]:

        if is_relevant(
            result,
            expected_citations
        ):
            return 1

    return 0


# ============================================================
# RECIPROCAL RANK
# ============================================================

def calculate_reciprocal_rank(
    results,
    expected_citations
):
    """Calculate reciprocal rank."""

    for rank, result in enumerate(
        results,
        start=1
    ):

        if is_relevant(
            result,
            expected_citations
        ):
            return 1 / rank

    return 0


# ============================================================
# CREATE COMPACT RESULT
# ============================================================

def create_compact_result(
    result,
    rank
):
    """Store compact retrieval information."""

    metadata = result.get(
        "metadata",
        {}
    )

    text = result.get(
        "text",
        ""
    )

    return {
        "rank": rank,

        "score": round(
            float(
                result.get(
                    "score",
                    0
                )
            ),
            4
        ),

        "retrieval_type":
            result.get(
                "retrieval_type"
            ),

        "citation":
            metadata.get(
                "citation"
            ),

        "paragraph":
            metadata.get(
                "paragraph"
            ),

        "subparagraph":
            metadata.get(
                "subparagraph"
            ),

        "clause":
            metadata.get(
                "clause"
            ),

        "text_preview":
            text[:300],
    }


# ============================================================
# EVALUATE DATASET
# ============================================================

def evaluate(
    retriever
):
    """Evaluate the structure-aware retriever."""

    print("\n" + "=" * 60)
    print("STRUCTURE-AWARE RETRIEVAL EVALUATION")
    print("=" * 60)

    print(
        f"Test queries: {len(TEST_QUERIES)}"
    )

    print(
        f"Top-k values: {TOP_K_VALUES}"
    )

    all_results = []

    recall_scores = {
        k: []
        for k in TOP_K_VALUES
    }

    reciprocal_ranks = []

    # --------------------------------------------------------
    # PROCESS EACH QUERY
    # --------------------------------------------------------

    for query_number, test_case in enumerate(
        TEST_QUERIES,
        start=1
    ):

        query = test_case["query"]

        expected_citations = (
            test_case[
                "expected_citations"
            ]
        )

        # Retrieve more than enough results
        # for Recall@10.

        results = retriever.retrieve(
            query,
            top_k=10,
            max_related=20
        )

        # ----------------------------------------------------
        # Calculate Recall@K
        # ----------------------------------------------------

        query_recall = {}

        for k in TOP_K_VALUES:

            score = calculate_recall_at_k(
                results,
                expected_citations,
                k
            )

            recall_scores[k].append(
                score
            )

            query_recall[
                str(k)
            ] = score

        # ----------------------------------------------------
        # Calculate Reciprocal Rank
        # ----------------------------------------------------

        reciprocal_rank = (
            calculate_reciprocal_rank(
                results,
                expected_citations
            )
        )

        reciprocal_ranks.append(
            reciprocal_rank
        )

        # ----------------------------------------------------
        # Find relevant rank
        # ----------------------------------------------------

        relevant_rank = None

        for rank, result in enumerate(
            results,
            start=1
        ):

            if is_relevant(
                result,
                expected_citations
            ):

                relevant_rank = rank
                break

        # ----------------------------------------------------
        # Store top 10 results
        # ----------------------------------------------------

        compact_results = []

        for rank, result in enumerate(
            results[:10],
            start=1
        ):

            compact_results.append(
                create_compact_result(
                    result,
                    rank
                )
            )

        # ----------------------------------------------------
        # Store query result
        # ----------------------------------------------------

        all_results.append(
            {
                "query_number":
                    query_number,

                "query":
                    query,

                "expected_citations":
                    expected_citations,

                "relevant_rank":
                    relevant_rank,

                "reciprocal_rank":
                    round(
                        reciprocal_rank,
                        4
                    ),

                "recall_at_k":
                    query_recall,

                "top_10_results":
                    compact_results,
            }
        )

        # ----------------------------------------------------
        # Terminal output
        # ----------------------------------------------------

        status = (
            "PASS"
            if relevant_rank is not None
            else "FAIL"
        )

        rank_text = (
            str(relevant_rank)
            if relevant_rank is not None
            else ">10"
        )

        print(
            f"{query_number:2d}. "
            f"{status} | "
            f"Relevant rank: {rank_text} | "
            f"{query}"
        )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    total_queries = len(
        TEST_QUERIES
    )

    final_metrics = {}

    print("\n" + "=" * 60)
    print("STRUCTURE-AWARE RETRIEVAL RESULTS")
    print("=" * 60)

    for k in TOP_K_VALUES:

        recall = (
            sum(
                recall_scores[k]
            )
            / total_queries
        )

        final_metrics[
            f"Recall@{k}"
        ] = recall

        print(
            f"Recall@{k}: "
            f"{recall:.4f}"
        )

    mrr = (
        sum(reciprocal_ranks)
        / total_queries
    )

    final_metrics[
        "MRR"
    ] = mrr

    print(
        f"MRR:       {mrr:.4f}"
    )

    # ========================================================
    # FAILED QUERIES
    # ========================================================

    failed_queries = [
        result
        for result in all_results
        if result[
            "relevant_rank"
        ] is None
    ]

    print("\n" + "=" * 60)
    print("FAILED QUERIES")
    print("=" * 60)

    if not failed_queries:

        print(
            "No failed queries."
        )

    else:

        for result in failed_queries:

            print(
                f"\nQuery "
                f"{result['query_number']}:"
            )

            print(
                result["query"]
            )

            print(
                "Expected: "
                + ", ".join(
                    result[
                        "expected_citations"
                    ]
                )
            )

            print(
                "\nTop retrieved citations:"
            )

            for retrieved in result[
                "top_10_results"
            ]:

                print(
                    f"  Rank "
                    f"{retrieved['rank']}: "
                    f"{retrieved['citation']} "
                    f"({retrieved['score']:.4f}) "
                    f"[{retrieved['retrieval_type']}]"
                )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    evaluation_output = {

        "evaluation":
            "structure_aware_retrieval",

        "model":
            MODEL_NAME,

        "index":
            str(
                retriever.index_path
                if hasattr(
                    retriever,
                    "index_path"
                )
                else "data/vectorstore/title-7/legal_documents.faiss"
            ),

        "metadata":
            "data/vectorstore/title-7/metadata.json",

        "number_of_queries":
            total_queries,

        "top_k_values":
            TOP_K_VALUES,

        "structure_expansion":
            {
                "top_k_semantic":
                    10,

                "max_related":
                    20
            },

        "metrics":
            final_metrics,

        "failed_query_count":
            len(
                failed_queries
            ),

        "query_results":
            all_results,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            evaluation_output,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("RESULTS SAVED")
    print("=" * 60)

    print(
        OUTPUT_PATH
    )

    return final_metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("eCFR STRUCTURE-AWARE RETRIEVAL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load structure-aware retriever
    # --------------------------------------------------------

    retriever = StructureAwareRetriever()

    # --------------------------------------------------------
    # Run evaluation
    # --------------------------------------------------------

    evaluate(
        retriever
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
