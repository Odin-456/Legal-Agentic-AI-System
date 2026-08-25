import json
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


# ============================================================
# LEGAL RETRIEVER
# ============================================================

class LegalRetriever:
    """
    Retrieval component for the eCFR legal document collection.

    This class reuses the validated baseline FAISS index and
    embedding model without modifying the underlying data.
    """

    def __init__(
        self,
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
        model_name=MODEL_NAME,
    ):
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)

        print("Loading FAISS index...")
        self.index = faiss.read_index(str(index_path))

        print("Loading metadata...")
        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:
            self.metadata = json.load(file)

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                "FAISS index and metadata counts do not match."
            )

        print(f"Vectors: {self.index.ntotal}")
        print(f"Dimensions: {self.index.d}")
        print(f"Metadata records: {len(self.metadata)}")

    # ========================================================
    # EMBEDDING
    # ========================================================

    def embed_query(self, query):
        """
        Convert a query into the same normalized embedding
        representation used by the baseline retrieval system.
        """

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        return np.asarray(
            embedding,
            dtype="float32"
        )

    # ========================================================
    # RETRIEVAL
    # ========================================================

    def retrieve(self, query, top_k=5):
        """
        Retrieve the top-k most similar legal document chunks.
        """

        query_embedding = self.embed_query(query)

        scores, indices = self.index.search(
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

            item = self.metadata[index_id]

            document_metadata = item.get(
                "metadata",
                {}
            )

            results.append(
                {
                    "score": float(score),
                    "text": item.get(
                        "text",
                        ""
                    ),
                    "citation": document_metadata.get(
                        "citation",
                        ""
                    ),
                    "paragraph": document_metadata.get(
                        "paragraph"
                    ),
                    "subparagraph": document_metadata.get(
                        "subparagraph"
                    ),
                    "clause": document_metadata.get(
                        "clause"
                    ),
                    "id": item.get(
                        "id",
                        ""
                    ),
                }
            )

        return results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    retriever = LegalRetriever()

    query = (
        "What happens when land becomes agricultural "
        "land after being acquired?"
    )

    results = retriever.retrieve(
        query,
        top_k=5
    )

    print("\nQUERY:")
    print(query)

    print("\nTOP 5 RESULTS:")

    for rank, result in enumerate(
        results,
        start=1
    ):
        print(
            f"\n{rank}. "
            f"{result['citation']} "
            f"(score={result['score']:.4f})"
        )

        print(
            f"Paragraph: "
            f"{result['paragraph']}"
        )

        print(
            f"Text: "
            f"{result['text'][:500]}"
        )