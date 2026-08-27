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
# STRUCTURE-AWARE RETRIEVER
# ============================================================

class StructureAwareRetriever:
    """
    Structure-aware legal retriever.

    Retrieval process:

        Query
          ↓
        Semantic retrieval
          ↓
        Candidate legal sections
          ↓
        Structural expansion
          ↓
        Re-ranking
          ↓
        Final results
    """

    def __init__(
        self,
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
        model_name=MODEL_NAME,
    ):

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            model_name
        )

        print("Loading FAISS index...")

        self.index = faiss.read_index(
            str(index_path)
        )

        print("Loading metadata...")

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.metadata = json.load(
                file
            )

        if self.index.ntotal != len(
            self.metadata
        ):

            raise ValueError(
                "FAISS index and metadata counts "
                "do not match."
            )

        print(
            f"Vectors: {self.index.ntotal}"
        )

        print(
            f"Dimensions: {self.index.d}"
        )

        print(
            f"Metadata records: "
            f"{len(self.metadata)}"
        )

        # ----------------------------------------------------
        # Build legal-section lookup
        # ----------------------------------------------------

        self.section_lookup = {}

        for index_id, item in enumerate(
            self.metadata
        ):

            metadata = item.get(
                "metadata",
                {}
            )

            citation = metadata.get(
                "citation"
            )

            if citation:

                self.section_lookup.setdefault(
                    citation,
                    []
                ).append(index_id)

        print(
            f"Legal sections indexed: "
            f"{len(self.section_lookup)}"
        )

    # ========================================================
    # EMBEDDING
    # ========================================================

    def embed_query(self, query):

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        return np.asarray(
            embedding,
            dtype="float32"
        )

    # ========================================================
    # SEMANTIC RETRIEVAL
    # ========================================================

    def semantic_retrieve(
        self,
        query,
        top_k=20
    ):

        query_embedding = self.embed_query(
            query
        )

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

            item = self.metadata[
                index_id
            ]

            results.append(
                {
                    "index_id": int(
                        index_id
                    ),
                    "score": float(
                        score
                    ),
                    "text": item.get(
                        "text",
                        ""
                    ),
                    "retrieval_text": item.get(
                        "retrieval_text",
                        ""
                    ),
                    "metadata": item.get(
                        "metadata",
                        {}
                    ),
                }
            )

        return results

    # ========================================================
    # STRUCTURE EXPANSION
    # ========================================================

    def expand_structure(
        self,
        semantic_results,
        max_sections=10
    ):

        expanded = []

        seen_sections = set()

        for result in semantic_results:

            metadata = result[
                "metadata"
            ]

            citation = metadata.get(
                "citation"
            )

            if not citation:
                continue

            # ------------------------------------------------
            # Avoid expanding the same section repeatedly
            # ------------------------------------------------

            if citation in seen_sections:
                continue

            seen_sections.add(
                citation
            )

            related_ids = self.section_lookup.get(
                citation,
                []
            )

            # ------------------------------------------------
            # Add every chunk belonging to this section
            # ------------------------------------------------

            for index_id in related_ids:

                item = self.metadata[
                    index_id
                ]

                item_metadata = item.get(
                    "metadata",
                    {}
                )

                expanded.append(
                    {
                        "index_id": index_id,
                        "section_score": result[
                            "score"
                        ],
                        "text": item.get(
                            "text",
                            ""
                        ),
                        "retrieval_text": item.get(
                            "retrieval_text",
                            ""
                        ),
                        "metadata": item_metadata,
                    }
                )

            if len(
                seen_sections
            ) >= max_sections:

                break

        return expanded

    # ========================================================
    # FINAL RETRIEVAL
    # ========================================================

    def retrieve(
        self,
        query,
        top_k=5,
        max_related=10
    ):

        # ----------------------------------------------------
        # Step 1: Retrieve more semantic candidates
        # ----------------------------------------------------

        semantic_results = (
            self.semantic_retrieve(
                query,
                top_k=20
            )
        )

        # ----------------------------------------------------
        # Step 2: Expand several candidate sections
        # ----------------------------------------------------

        structure_results = (
            self.expand_structure(
                semantic_results,
                max_sections=10
            )
        )

        # ----------------------------------------------------
        # Step 3: Build section scores
        # ----------------------------------------------------

        section_scores = {}

        for result in semantic_results:

            citation = result[
                "metadata"
            ].get(
                "citation"
            )

            if not citation:
                continue

            score = result[
                "score"
            ]

            if (
                citation not in
                section_scores
            ):

                section_scores[
                    citation
                ] = score

            else:

                section_scores[
                    citation
                ] = max(
                    section_scores[
                        citation
                    ],
                    score
                )

        # ----------------------------------------------------
        # Step 4: Combine semantic and structural results
        # ----------------------------------------------------

        combined = []

        seen = set()

        # ----------------------------------------------------
        # Semantic results
        # ----------------------------------------------------

        for result in semantic_results:

            index_id = result[
                "index_id"
            ]

            if index_id in seen:
                continue

            result[
                "retrieval_type"
            ] = "semantic"

            combined.append(
                result
            )

            seen.add(
                index_id
            )

        # ----------------------------------------------------
        # Structural results
        # ----------------------------------------------------

        for result in structure_results:

            index_id = result[
                "index_id"
            ]

            if index_id in seen:
                continue

            citation = result[
                "metadata"
            ].get(
                "citation"
            )

            result[
                "score"
            ] = section_scores.get(
                citation,
                result.get(
                    "section_score",
                    0.0
                )
            )

            result[
                "retrieval_type"
            ] = "structural"

            combined.append(
                result
            )

            seen.add(
                index_id
            )

        # ----------------------------------------------------
        # Step 5: Rank by score
        # ----------------------------------------------------

        combined.sort(
            key=lambda x: x[
                "score"
            ],
            reverse=True
        )

        # ----------------------------------------------------
        # Step 6: Limit final results
        # ----------------------------------------------------

        return combined[
            :(
                top_k +
                max_related
            )
        ]


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    retriever = StructureAwareRetriever()

    query = (
        "What happens when land becomes "
        "agricultural land after being acquired?"
    )

    print("\nQUERY:")
    print(query)

    results = retriever.retrieve(
        query,
        top_k=5,
        max_related=10
    )

    print(
        "\nSTRUCTURE-AWARE RESULTS:"
    )

    for number, result in enumerate(
        results,
        start=1
    ):

        metadata = result[
            "metadata"
        ]

        citation = metadata.get(
            "citation",
            "Unknown"
        )

        paragraph = metadata.get(
            "paragraph"
        )

        subparagraph = metadata.get(
            "subparagraph"
        )

        location = citation

        if paragraph:
            location += f" {paragraph}"

        if subparagraph:
            location += f" {subparagraph}"

        print(
            f"\n{number}. "
            f"{location} "
            f"(score="
            f"{result['score']:.4f}, "
            f"type="
            f"{result['retrieval_type']})"
        )

        print(
            "Text:",
            result["text"][:500]
        )
