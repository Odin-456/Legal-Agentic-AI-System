import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path(
    "data/processed/chunks/title-7"
)

OUTPUT_DIR = Path(
    "data/processed/embeddings/title-7"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Embedding model
MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Number of documents embedded at one time.
# Increase if you have enough RAM/GPU memory.
BATCH_SIZE = 32


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 60)
    print("LOADING EMBEDDING MODEL")
    print("=" * 60)

    print(
        f"\nModel: {MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "Embedding model loaded successfully."
    )

    print(
        f"Embedding dimension: "
        f"{model.get_sentence_embedding_dimension()}"
    )

    return model


# ============================================================
# LOAD JSON
# ============================================================

def load_documents(input_file):

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(
    model,
    documents
):

    # --------------------------------------------------------
    # Extract retrieval text
    # --------------------------------------------------------

    texts = [
        document["retrieval_text"]
        for document in documents
    ]

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return embeddings


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

def save_embeddings(
    output_file,
    documents,
    embeddings
):

    output = []

    for index, document in enumerate(
        documents
    ):

        metadata = document.get(
            "metadata",
            {}
        )

        output.append(
            {
                "id": metadata.get(
                    "chunk_id",
                    f"chunk_{index}"
                ),

                "embedding": embeddings[
                    index
                ].tolist(),

                "text": document.get(
                    "text",
                    ""
                ),

                "retrieval_text": document.get(
                    "retrieval_text",
                    ""
                ),

                "metadata": metadata,
            }
        )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("eCFR EMBEDDING GENERATION")
    print("=" * 60)

    print(
        f"\nInput directory:"
        f" {INPUT_DIR}"
    )

    print(
        f"Output directory:"
        f" {OUTPUT_DIR}"
    )

    print(
        f"Model:"
        f" {MODEL_NAME}"
    )

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    json_files = sorted(
        INPUT_DIR.glob("*.json")
    )

    print(
        f"\nJSON files found:"
        f" {len(json_files)}"
    )

    if not json_files:

        print(
            "\nERROR:"
            " No chunk JSON files found."
        )

        return

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_documents = 0
    files_processed = 0
    files_failed = 0

    # --------------------------------------------------------
    # Process files
    # --------------------------------------------------------

    for number, input_file in enumerate(
        json_files,
        start=1
    ):

        print("\n" + "-" * 60)

        print(
            f"[{number}/{len(json_files)}]"
            f" Processing: {input_file.name}"
        )

        output_file = (
            OUTPUT_DIR /
            input_file.name
        )

        try:

            # ------------------------------------------------
            # Load documents
            # ------------------------------------------------

            documents = load_documents(
                input_file
            )

            print(
                f"  Documents:"
                f" {len(documents)}"
            )

            if not documents:

                print(
                    "  WARNING:"
                    " File contains no documents."
                )

                continue

            # ------------------------------------------------
            # Generate embeddings
            # ------------------------------------------------

            embeddings = create_embeddings(
                model,
                documents
            )

            # ------------------------------------------------
            # Check embedding dimensions
            # ------------------------------------------------

            if len(embeddings) != len(
                documents
            ):

                raise ValueError(
                    "Number of embeddings does not "
                    "match number of documents."
                )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            save_embeddings(
                output_file,
                documents,
                embeddings
            )

            total_documents += len(
                documents
            )

            files_processed += 1

            print(
                f"  Embedding shape:"
                f" {embeddings.shape}"
            )

            print(
                f"  Saved to:"
                f" {output_file}"
            )

        except Exception as e:

            files_failed += 1

            print(
                f"  ERROR:"
                f" {e}"
            )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print("\n")

    print("=" * 60)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 60)

    print(
        f"JSON files found:"
        f"       {len(json_files)}"
    )

    print(
        f"Files processed:"
        f"       {files_processed}"
    )

    print(
        f"Files failed:"
        f"          {files_failed}"
    )

    print(
        f"Total documents embedded:"
        f" {total_documents}"
    )

    print(
        f"Embedding model:"
        f"       {MODEL_NAME}"
    )

    print(
        f"Output directory:"
        f"      {OUTPUT_DIR}"
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()