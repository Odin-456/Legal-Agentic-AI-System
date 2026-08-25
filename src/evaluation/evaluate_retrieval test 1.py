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


# ============================================================
# TEST DATASET
# ============================================================

TEST_QUERIES = [
    {
        "query": "What is the definition of agricultural land under 7 CFR part 781?",
        "expected_citations": [
            "7 CFR 781.2"
        ],
    },
    {
        "query": "Who qualifies as a foreign person under 7 CFR 781.2?",
        "expected_citations": [
            "7 CFR 781.2"
        ],
    },
    {
        "query": "What information must a foreign person report under 7 CFR 781.3(f)(1)?",
        "expected_citations": [
            "7 CFR 781.3"
        ],
    },
    {
        "query": "What are the reporting requirements for foreign investment in agricultural land?",
        "expected_citations": [
            "7 CFR 781.3"
        ],
    },
    {
        "query": "What information must be reported about foreign individuals or governments holding significant interest or substantial control?",
        "expected_citations": [
            "7 CFR 781.3"
        ],
    },
    {
        "query": "What is the purpose of the regulations concerning foreign investment in agricultural land?",
        "expected_citations": [
            "7 CFR 781.1"
        ],
    },
    {
        "query": "What is AFIDA under 7 CFR 781.2?",
        "expected_citations": [
            "7 CFR 781.2"
        ],
    },
    {
        "query": "What does agricultural land include under the regulations?",
        "expected_citations": [
            "7 CFR 781.2"
        ],
    },
    {
        "query": "What information must be provided about agricultural land in a report?",
        "expected_citations": [
            "7 CFR 781.3"
        ],
    },
    {
        "query": "When must a foreign person file a report concerning agricultural land?",
        "expected_citations": [
            "7 CFR 781.3"
        ],
    },
    {
        "query": "What happens when land becomes agricultural land after being acquired?",
        "expected_citations": [
            "7 CFR 781.3"
        ],
    },
    {
        "query": "What information is required concerning the location and acreage of agricultural land?",
        "expected_citations": [
            "7 CFR 781.3"
        ],
    },
    {
        "query": "What is the legal name and address information required for foreign persons?",
        "expected_citations": [
            "7 CFR 781.3"
        ],
    },
    {
        "query": "What information must be reported concerning the use of agricultural land?",
        "expected_citations": [
            "7 CFR 781.3"
        ],
    },
    {
        "query": "What is a foreign individual under 7 CFR 781.2?",
        "expected_citations": [
            "7 CFR 781.2"
        ],
    },
    {
        "query": "What is the meaning of any interest under 7 CFR 781.2?",
        "expected_citations": [
            "7 CFR 781.2"
        ],
    },
    {
        "query": "What are the requirements for reporting foreign ownership of agricultural land?",
        "expected_citations": [
            "7 CFR 781.3"
        ],
    },
    {
        "query": "Which regulation implements the Agricultural Foreign Investment Disclosure Act?",
        "expected_citations": [
            "7 CFR 781.1"
        ],
    },
    {
        "query": "What form must a foreign person file under the reporting requirements?",
        "expected_citations": [
            "7 CFR 781.3"
        ],
    },
    {
        "query": "What agency receives reports concerning foreign investment in agricultural land?",
        "expected_citations": [
            "7 CFR 781.3"
        ],
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

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    print(
        f"Vectors: {index.ntotal}"
    )

    print(
        f"Dimensions: {index.d}"
    )

    return index


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():
    """Load metadata corresponding to FAISS vectors."""

    print("\n")
    print("=" * 60)
    print("LOADING METADATA")
    print("=" * 60)

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    print(
        f"Metadata records: {len(metadata)}"
    )

    return metadata


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_model():
    """Load the same embedding model used to build FAISS."""

    print("\n")
    print("=" * 60)
    print("LOADING EMBEDDING MODEL")
    print("=" * 60)

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    return model


# ============================================================
# NORMALIZE CITATION
# ============================================================

def normalize_citation(citation):
    """
    Normalize citation strings.

    Example:

        7 CFR 781.3
        7 CFR 781.3.
        7 CFR 781.3;

    become:

        7 CFR 781.3
    """

    if citation is None:
        return ""

    citation = str(citation).strip()

    citation = re.sub(
        r"[.,;:]+$",
        "",
        citation
    )

    # Normalize repeated whitespace
    citation = re.sub(
        r"\s+",
        " ",
        citation
    )

    return citation


# ============================================================
# GET CITATION FROM RESULT
# ============================================================

def get_result_citation(result):
    """
    Extract the citation from a retrieved result.

    Supports both metadata structures:

    1. Citation at the top level:
       {
           "citation": "7 CFR 781.3"
       }

    2. Citation inside a metadata object:
       {
           "metadata": {
               "citation": "7 CFR 781.3"
           }
       }
    """

    # --------------------------------------------------------
    # Case 1: citation at top level
    # --------------------------------------------------------

    citation = result.get(
        "citation"
    )

    if citation:
        return normalize_citation(
            citation
        )

    # --------------------------------------------------------
    # Case 2: citation inside metadata
    # --------------------------------------------------------

    result_metadata = result.get(
        "metadata",
        {}
    )

    if isinstance(
        result_metadata,
        dict
    ):
        citation = result_metadata.get(
            "citation"
        )

        if citation:
            return normalize_citation(
                citation
            )

    return ""


# ============================================================
# GET FIELD FROM RESULT
# ============================================================

def get_result_field(
    result,
    field_name,
    default=None
):
    """
    Retrieve a field from either the top-level result
    or the nested metadata object.
    """

    # Top-level field
    if field_name in result:
        return result[field_name]

    # Nested metadata field
    result_metadata = result.get(
        "metadata",
        {}
    )

    if isinstance(
        result_metadata,
        dict
    ):
        return result_metadata.get(
            field_name,
            default
        )

    return default


# ============================================================
# CHECK WHETHER RESULT IS RELEVANT
# ============================================================

def is_relevant(
    result,
    expected_citations
):
    """
    Determine whether a retrieved result belongs to one
    of the expected legal citations.
    """

    result_citation = get_result_citation(
        result
    )

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
    """
    Embed a query and retrieve the top-k documents.
    """

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

        if index_id >= len(metadata):
            continue

        result = metadata[index_id].copy()

        result["score"] = float(
            score
        )

        results.append(
            result
        )

    return results


# ============================================================
# RECALL@K
# ============================================================

def calculate_recall_at_k(
    results,
    expected_citations,
    k
):
    """
    Recall@K for citation-level retrieval.

    Returns 1 if at least one expected citation
    appears within the top-k results.
    """

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
    """
    Calculate reciprocal rank.

    Correct result at rank 1:
        RR = 1

    Correct result at rank 2:
        RR = 1/2

    Correct result at rank 5:
        RR = 1/5

    No correct result:
        RR = 0
    """

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
# EVALUATE DATASET
# ============================================================

def evaluate(
    model,
    index,
    metadata
):
    """
    Evaluate all test queries.
    """

    print("\n")
    print("=" * 60)
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

        query = test_case[
            "query"
        ]

        expected_citations = (
            test_case[
                "expected_citations"
            ]
        )

        print("\n")
        print(
            "#" * 60
        )

        print(
            f"QUERY {query_number}/{len(TEST_QUERIES)}"
        )

        print(
            "#" * 60
        )

        print("\nQuery:")
        print(query)

        print("\nExpected citation(s):")

        for citation in expected_citations:
            print(
                f"  {citation}"
            )

        # ----------------------------------------------------
        # RETRIEVE TOP 10
        # ----------------------------------------------------

        results = retrieve(
            query=query,
            model=model,
            index=index,
            metadata=metadata,
            top_k=10
        )

        # ----------------------------------------------------
        # CALCULATE RECALL@K
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
        # CALCULATE RECIPROCAL RANK
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
        # DISPLAY TOP 5 RESULTS
        # ----------------------------------------------------

        print("\nTop 5 results:")

        for rank, result in enumerate(
            results[:5],
            start=1
        ):

            relevant = is_relevant(
                result,
                expected_citations
            )

            marker = "✓" if relevant else " "

            citation = get_result_citation(
                result
            )

            paragraph = get_result_field(
                result,
                "paragraph"
            )

            subparagraph = get_result_field(
                result,
                "subparagraph"
            )

            clause = get_result_field(
                result,
                "clause"
            )

            text = get_result_field(
                result,
                "text",
                ""
            )

            # Some datasets use retrieval_text
            # instead of text.
            if not text:
                text = get_result_field(
                    result,
                    "retrieval_text",
                    ""
                )

            print(
                f"\n{marker} Rank: {rank}"
            )

            print(
                f"  Score: "
                f"{result.get('score', 0):.4f}"
            )

            print(
                f"  Citation: "
                f"{citation if citation else 'None'}"
            )

            print(
                f"  Paragraph: "
                f"{paragraph if paragraph else 'None'}"
            )

            print(
                f"  Subparagraph: "
                f"{subparagraph if subparagraph else 'None'}"
            )

            print(
                f"  Clause: "
                f"{clause if clause else 'None'}"
            )

            print(
                f"  Text: "
                f"{str(text)[:250]}"
            )

        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        all_results.append(
            {
                "query": query,
                "expected_citations":
                    expected_citations,
                "retrieved_results":
                    results,
                "reciprocal_rank":
                    reciprocal_rank,
                "recall_at_k":
                    query_recall,
            }
        )

    # ========================================================
    # CALCULATE FINAL METRICS
    # ========================================================

    print("\n")
    print("=" * 60)
    print("BASELINE RETRIEVAL RESULTS")
    print("=" * 60)

    total_queries = len(
        TEST_QUERIES
    )

    final_metrics = {}

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

    # --------------------------------------------------------
    # Mean Reciprocal Rank
    # --------------------------------------------------------

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
    # SAVE EVALUATION RESULTS
    # ========================================================

    output_directory = Path(
        "data/evaluation"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_directory
        / "baseline_retrieval_results.json"
    )

    evaluation_output = {
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

        "query_results":
            all_results,
    }

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            evaluation_output,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n")
    print(
        "Results saved to:"
    )

    print(
        output_file
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
            "FAISS index and metadata counts "
            "do not match."
        )

        print(
            f"FAISS vectors: {index.ntotal}"
        )

        print(
            f"Metadata records: {len(metadata)}"
        )

        return

    print("\n")
    print(
        "Index and metadata are consistent."
    )

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

    print("\n")
    print("=" * 60)
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

