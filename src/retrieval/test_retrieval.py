import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_FILE = Path(
    "data/vectorstore/title-7/legal_documents.faiss"
)

METADATA_FILE = Path(
    "data/vectorstore/title-7/metadata.json"
)

MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Number of results to retrieve
TOP_K = 5


# ============================================================
# LOAD FAISS INDEX
# ============================================================

def load_index():

    print("=" * 60)
    print("LOADING FAISS INDEX")
    print("=" * 60)

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    print(
        f"\nVectors:"
        f" {index.ntotal}"
    )

    print(
        f"Dimensions:"
        f" {index.d}"
    )

    return index


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    print(
        f"Metadata records:"
        f" {len(metadata)}"
    )

    return metadata


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_model():

    print("\n")
    print("=" * 60)
    print("LOADING EMBEDDING MODEL")
    print("=" * 60)

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        f"\nModel:"
        f" {MODEL_NAME}"
    )

    return model


# ============================================================
# SEARCH
# ============================================================

def search(
    query,
    model,
    index,
    metadata,
    top_k=5
):

    # --------------------------------------------------------
    # Convert question into an embedding
    # --------------------------------------------------------

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    # --------------------------------------------------------
    # Search FAISS
    # --------------------------------------------------------

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for rank, (
        score,
        index_position
    ) in enumerate(
        zip(
            scores[0],
            indices[0]
        ),
        start=1
    ):

        if index_position < 0:
            continue

        document = metadata[
            index_position
        ]

        results.append(
            {
                "rank": rank,
                "score": float(score),
                "document": document,
            }
        )

    return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    query,
    results
):

    print("\n")
    print("=" * 60)
    print("QUERY")
    print("=" * 60)

    print(query)

    print("\n")
    print("=" * 60)
    print("RETRIEVAL RESULTS")
    print("=" * 60)

    for result in results:

        document = result[
            "document"
        ]

        metadata = document.get(
            "metadata",
            {}
        )

        print("\n" + "-" * 60)

        print(
            f"Rank: {result['rank']}"
        )

        print(
            f"Score: {result['score']:.4f}"
        )

        print(
            f"ID: {document.get('id')}"
        )

        print(
            f"Citation:"
            f" {metadata.get('citation')}"
        )

        print(
            f"Section:"
            f" {metadata.get('section')}"
        )

        print(
            f"Section title:"
            f" {metadata.get('section_title')}"
        )

        print(
            f"Paragraph:"
            f" {metadata.get('paragraph')}"
        )

        print(
            f"Subparagraph:"
            f" {metadata.get('subparagraph')}"
        )

        print(
            f"Clause:"
            f" {metadata.get('clause')}"
        )

        print(
            f"Subclause:"
            f" {metadata.get('subclause')}"
        )

        print("\nText:")

        text = document.get(
            "text",
            ""
        )

        print(
            text[:1000]
        )


# ============================================================
# TEST QUESTIONS
# ============================================================

TEST_QUERIES = [

    (
        "What information must a foreign person "
        "report under 7 CFR 781.3(f)(1)?"
    ),

    (
        "What is the definition of agricultural land "
        "under 7 CFR part 781?"
    ),

    (
        "Who qualifies as a foreign person under "
        "7 CFR 781.2?"
    ),

    (
        "What information must be reported about "
        "foreign individuals or governments holding "
        "significant interest or substantial control?"
    ),

    (
        "What are the reporting requirements for "
        "foreign investment in agricultural land?"
    ),
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("eCFR RETRIEVAL TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not INDEX_FILE.exists():

        print(
            f"\nERROR: FAISS index not found:"
            f"\n{INDEX_FILE}"
        )

        return

    if not METADATA_FILE.exists():

        print(
            f"\nERROR: Metadata file not found:"
            f"\n{METADATA_FILE}"
        )

        return

    # --------------------------------------------------------
    # Load components
    # --------------------------------------------------------

    index = load_index()

    metadata = load_metadata()

    model = load_model()

    # --------------------------------------------------------
    # Verify counts
    # --------------------------------------------------------

    if index.ntotal != len(metadata):

        raise ValueError(
            "FAISS index and metadata counts "
            "do not match."
        )

    # --------------------------------------------------------
    # Run test queries
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print(
        f"RUNNING {len(TEST_QUERIES)} TEST QUERIES"
    )
    print("=" * 60)

    for number, query in enumerate(
        TEST_QUERIES,
        start=1
    ):

        print("\n\n")
        print(
            "#" * 60
        )

        print(
            f"TEST QUERY {number}"
        )

        print(
            "#" * 60
        )

        results = search(
            query=query,
            model=model,
            index=index,
            metadata=metadata,
            top_k=TOP_K
        )

        display_results(
            query,
            results
        )

    print("\n")
    print("=" * 60)
    print("RETRIEVAL TEST COMPLETE")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()