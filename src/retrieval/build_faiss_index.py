import json
from pathlib import Path

import faiss
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path(
    "data/processed/embeddings/title-7"
)

OUTPUT_DIR = Path(
    "data/vectorstore/title-7"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INDEX_FILE = (
    OUTPUT_DIR /
    "legal_documents.faiss"
)

METADATA_FILE = (
    OUTPUT_DIR /
    "metadata.json"
)


# Embedding dimension for
# BAAI/bge-base-en-v1.5
EMBEDDING_DIMENSION = 768


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings():

    print("=" * 60)
    print("LOADING EMBEDDINGS")
    print("=" * 60)

    json_files = sorted(
        INPUT_DIR.glob("*.json")
    )

    print(
        f"\nEmbedding files found:"
        f" {len(json_files)}"
    )

    if not json_files:

        raise FileNotFoundError(
            "No embedding JSON files found."
        )

    all_embeddings = []
    all_metadata = []

    total_documents = 0

    # --------------------------------------------------------
    # Process embedding files
    # --------------------------------------------------------

    for number, input_file in enumerate(
        json_files,
        start=1
    ):

        print(
            f"\rProcessing:"
            f" {number}/{len(json_files)}",
            end=""
        )

        with open(
            input_file,
            "r",
            encoding="utf-8"
        ) as f:

            documents = json.load(f)

        for document in documents:

            embedding = document.get(
                "embedding"
            )

            if embedding is None:

                raise ValueError(
                    f"Missing embedding in "
                    f"{input_file.name}"
                )

            if len(embedding) != (
                EMBEDDING_DIMENSION
            ):

                raise ValueError(
                    f"Invalid embedding dimension "
                    f"in {input_file.name}: "
                    f"{len(embedding)}"
                )

            all_embeddings.append(
                embedding
            )

            # Store everything needed
            # to interpret retrieval results.
            all_metadata.append(
                {
                    "id": document.get(
                        "id"
                    ),

                    "text": document.get(
                        "text",
                        ""
                    ),

                    "retrieval_text": document.get(
                        "retrieval_text",
                        ""
                    ),

                    "metadata": document.get(
                        "metadata",
                        {}
                    ),
                }
            )

            total_documents += 1

    print()

    print(
        f"\nTotal embeddings:"
        f" {total_documents}"
    )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    embeddings = np.asarray(
        all_embeddings,
        dtype=np.float32
    )

    print(
        f"Embedding matrix shape:"
        f" {embeddings.shape}"
    )

    return embeddings, all_metadata


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_index(embeddings):

    print("\n")
    print("=" * 60)
    print("BUILDING FAISS INDEX")
    print("=" * 60)

    # --------------------------------------------------------
    # IndexFlatIP
    #
    # Embeddings were normalized during
    # embedding generation.
    #
    # Therefore:
    #
    # Inner Product = Cosine Similarity
    # --------------------------------------------------------

    index = faiss.IndexFlatIP(
        EMBEDDING_DIMENSION
    )

    index.add(
        embeddings
    )

    print(
        f"\nVectors added:"
        f" {index.ntotal}"
    )

    return index


# ============================================================
# SAVE INDEX
# ============================================================

def save_index(
    index,
    metadata
):

    print("\n")
    print("=" * 60)
    print("SAVING VECTOR DATABASE")
    print("=" * 60)

    # --------------------------------------------------------
    # Save FAISS index
    # --------------------------------------------------------

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    print(
        f"\nFAISS index saved to:"
        f"\n{INDEX_FILE}"
    )

    # --------------------------------------------------------
    # Save metadata mapping
    # --------------------------------------------------------

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False
        )

    print(
        f"\nMetadata saved to:"
        f"\n{METADATA_FILE}"
    )


# ============================================================
# VERIFY INDEX
# ============================================================

def verify_index():

    print("\n")
    print("=" * 60)
    print("VERIFYING FAISS INDEX")
    print("=" * 60)

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    print(
        f"\nFAISS vectors:"
        f" {index.ntotal}"
    )

    print(
        f"Metadata records:"
        f" {len(metadata)}"
    )

    print(
        f"Vector dimension:"
        f" {index.d}"
    )

    # --------------------------------------------------------
    # Check consistency
    # --------------------------------------------------------

    if index.ntotal != len(metadata):

        raise ValueError(
            "FAISS index and metadata "
            "record counts do not match."
        )

    if index.d != EMBEDDING_DIMENSION:

        raise ValueError(
            "FAISS index has an unexpected "
            "embedding dimension."
        )

    print(
        "\nSUCCESS:"
    )

    print(
        "FAISS index and metadata are consistent."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("eCFR FAISS VECTOR DATABASE")
    print("=" * 60)

    # --------------------------------------------------------
    # Load embeddings
    # --------------------------------------------------------

    embeddings, metadata = load_embeddings()

    # --------------------------------------------------------
    # Build index
    # --------------------------------------------------------

    index = build_index(
        embeddings
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_index(
        index,
        metadata
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify_index()

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("FAISS INDEX BUILD COMPLETE")
    print("=" * 60)

    print(
        f"Vectors:"
        f"        {index.ntotal}"
    )

    print(
        f"Dimension:"
        f"       {index.d}"
    )

    print(
        f"Index:"
        f"            {INDEX_FILE}"
    )

    print(
        f"Metadata:"
        f"         {METADATA_FILE}"
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()