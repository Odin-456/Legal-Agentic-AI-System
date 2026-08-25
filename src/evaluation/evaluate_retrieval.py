import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_PATH = Path(
    "data/vectorstore/title-7/legal_documents.faiss"
)

METADATA_PATH = Path(
    "data/vectorstore/title-7/metadata.json"
)

MODEL_NAME = "BAAI/bge-base-en-v1.5"

TOP_K_VALUES = [1, 3, 5, 10]

OUTPUT_PATH = Path(
    "data/evaluation/baseline_retrieval_results.json"
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
# LOAD FAISS INDEX
# ============================================================

def load_index():
    """Load the FAISS vector index."""

    print("=" * 60)
    print("LOADING FAISS INDEX")
    print("=" * 60)

    index = faiss.read_index(str(INDEX_PATH))

    print(f"Vectors: {index.ntotal}")
    print(f"Dimensions: {index.d}")

    return index


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():
    """Load metadata corresponding to FAISS vectors."""

    print("\n" + "=" * 60)
    print("LOADING METADATA")
    print("=" * 60)

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    print(f"Metadata records: {len(metadata)}")

    return metadata


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_model():
    """Load the same embedding model used to build FAISS."""

    print("\n" + "=" * 60)
    print("LOADING EMBEDDING MODEL")
    print("=" * 60)

    model = SentenceTransformer(MODEL_NAME)

    print(f"Model: {MODEL_NAME}")

    return model


# ============================================================
# NORMALIZE CITATION
# ============================================================

def normalize_citation(citation):
    """Normalize citation strings."""

    if not citation:
        return ""

    citation = str(citation).strip()

    citation = re.sub(
        r"[.,;:]+$",
        "",
        citation
    )

    return citation


# ============================================================
# GET CITATION FROM RESULT
# ============================================================

def get_result_citation(result):
    """
    Extract citation from the nested metadata structure.

    Expected structure:

        result
        ├── id
        ├── text
        ├── retrieval_text
        ├── metadata
        │   ├── citation
        │   ├── paragraph
        │   ├── subparagraph
        │   └── clause
        └── score
    """

    metadata = result.get("metadata", {})

    return normalize_citation(
        metadata.get("citation", "")
    )


# ============================================================
# CHECK WHETHER RESULT IS RELEVANT
# ============================================================

def is_relevant(result, expected_citations):
    """Return True if result citation matches an expected citation."""

    result_citation = get_result_citation(result)

    expected = [
        normalize_citation(citation)
        for citation in expected_citations
    ]

    return result_citation in expected


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve(
    query,
    model,
    index,
    metadata,
    top_k
):
    """Embed a query and retrieve the top-k documents."""

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):

        if index_id < 0:
            continue

        result = metadata[index_id].copy()

        result["score"] = float(score)

        results.append(result)

    return results


# ============================================================
# RECALL@K
# ============================================================

def calculate_recall_at_k(
    results,
    expected_citations,
    k
):
    """Return 1 if an expected citation appears within top-k."""

    top_results = results[:k]

    for result in top_results:

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
    """Calculate reciprocal rank of the first relevant result."""

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
    """
    Store only the information needed for evaluation.

    This prevents the output JSON from becoming unnecessarily large.
    """

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
            float(result.get("score", 0)),
            4
        ),
        "citation": metadata.get(
            "citation"
        ),
        "paragraph": metadata.get(
            "paragraph"
        ),
        "subparagraph": metadata.get(
            "subparagraph"
        ),
        "clause": metadata.get(
            "clause"
        ),
        "text_preview": text[:300],
    }


# ============================================================
# EVALUATE DATASET
# ============================================================

def evaluate(
    model,
    index,
    metadata
):
    """Evaluate the complete retrieval test dataset."""

    print("\n" + "=" * 60)
    print("BASELINE RETRIEVAL EVALUATION")
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
            test_case["expected_citations"]
        )

        results = retrieve(
            query=query,
            model=model,
            index=index,
            metadata=metadata,
            top_k=max(TOP_K_VALUES)
        )

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        query_recall = {}

        for k in TOP_K_VALUES:

            score = calculate_recall_at_k(
                results,
                expected_citations,
                k
            )

            recall_scores[k].append(score)

            query_recall[
                str(k)
            ] = score

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
        # Find first relevant result
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
        # Compact retrieved results
        # ----------------------------------------------------

        compact_results = []

        for rank, result in enumerate(
            results[:5],
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
                "query_number": query_number,

                "query": query,

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

                "top_5_results":
                    compact_results,
            }
        )

        # ----------------------------------------------------
        # Short terminal output
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
            f"[{status}] "
            f"Query {query_number:02d}/"
            f"{len(TEST_QUERIES)} "
            f"| Relevant rank: {rank_text} "
            f"| Expected: "
            f"{', '.join(expected_citations)}"
        )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    total_queries = len(
        TEST_QUERIES
    )

    final_metrics = {}

    print("\n" + "=" * 60)
    print("BASELINE RETRIEVAL RESULTS")
    print("=" * 60)

    for k in TOP_K_VALUES:

        recall = (
            sum(recall_scores[k])
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
        if result["relevant_rank"] is None
    ]

    print("\n" + "=" * 60)
    print("FAILED QUERIES")
    print("=" * 60)

    if not failed_queries:

        print("No failed queries.")

    else:

        for result in failed_queries:

            print(
                f"\nQuery {result['query_number']}:"
            )

            print(
                f"  {result['query']}"
            )

            print(
                f"Expected: "
                f"{', '.join(result['expected_citations'])}"
            )

            print(
                "\nTop retrieved citations:"
            )

            for retrieved in result["top_5_results"]:

                print(
                    f"  Rank {retrieved['rank']}: "
                    f"{retrieved['citation']} "
                    f"({retrieved['score']:.4f})"
                )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    evaluation_output = {
        "evaluation": "baseline_retrieval",

        "model": MODEL_NAME,

        "index": str(
            INDEX_PATH
        ),

        "metadata": str(
            METADATA_PATH
        ),

        "number_of_queries":
            total_queries,

        "top_k_values":
            TOP_K_VALUES,

        "metrics":
            final_metrics,

        "failed_query_count":
            len(failed_queries),

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
    print("eCFR BASELINE RETRIEVAL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    if not INDEX_PATH.exists():

        print(
            f"\nERROR: FAISS index not found:\n"
            f"{INDEX_PATH}"
        )

        return

    if not METADATA_PATH.exists():

        print(
            f"\nERROR: Metadata file not found:\n"
            f"{METADATA_PATH}"
        )

        return

    # --------------------------------------------------------
    # Load components
    # --------------------------------------------------------

    index = load_index()

    metadata = load_metadata()

    # --------------------------------------------------------
    # Validate index and metadata
    # --------------------------------------------------------

    if index.ntotal != len(metadata):

        print("\nERROR:")
        print(
            "FAISS index and metadata counts do not match."
        )

        print(
            f"FAISS vectors: {index.ntotal}"
        )

        print(
            f"Metadata records: {len(metadata)}"
        )

        return

    print("\nIndex and metadata are consistent.")

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = evaluate(
        model=model,
        index=index,
        metadata=metadata
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    for metric, value in metrics.items():

        print(
            f"{metric}: {value:.4f}"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()

