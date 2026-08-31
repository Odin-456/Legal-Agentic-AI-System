import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from query_expansion import expand_query


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

SEMANTIC_TOP_K = 30
MAX_SECTIONS = 10


# ============================================================
# LEGAL TERM EXTRACTION
# ============================================================

STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "when",
    "what",
    "what's",
    "does",
    "do",
    "did",
    "must",
    "should",
    "can",
    "may",
    "will",
    "would",
    "could",
    "how",
    "why",
    "where",
    "which",
    "who",
    "after",
    "before",
    "during",
    "for",
    "from",
    "to",
    "of",
    "on",
    "in",
    "and",
    "or",
    "with",
    "by",
    "as",
    "when",
    "land",
}


def tokenize(text):
    """
    Convert text into simple lowercase tokens.
    """

    return re.findall(
        r"[a-z0-9]+(?:-[a-z0-9]+)*",
        text.lower()
    )


def important_terms(text):
    """
    Extract useful legal/search terms.
    """

    tokens = tokenize(text)

    return {
        token
        for token in tokens
        if token not in STOPWORDS
        and len(token) > 2
    }


# ============================================================
# STRUCTURE-AWARE RETRIEVER
# ============================================================

class StructureAwareRetriever:
    """
    Structure-aware legal retriever.

    Retrieval pipeline:

        Original query
              +
        Expanded query
              |
              v
        Semantic retrieval
              |
              v
        Legal-term matching
              |
              v
        Candidate sections
              |
              v
        Structural expansion
              |
              v
        Re-ranking
              |
              v
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

            self.metadata = json.load(file)

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
        # Build section lookup
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
        top_k=SEMANTIC_TOP_K
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

                    "semantic_score": float(
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

                    "retrieval_type": "semantic",
                }
            )

        return results

    # ========================================================
    # LEXICAL MATCHING
    # ========================================================

    def lexical_score(
        self,
        query,
        result
    ):
        """
        Measures overlap between important query
        terms and the retrieved legal text.

        This is deliberately simple and transparent.
        """

        query_terms = important_terms(
            query
        )

        if not query_terms:
            return 0.0

        metadata = result.get(
            "metadata",
            {}
        )

        text = " ".join(
            [
                metadata.get(
                    "section_title",
                    ""
                ),
                result.get(
                    "text",
                    ""
                ),
                result.get(
                    "retrieval_text",
                    ""
                ),
            ]
        )

        document_terms = important_terms(
            text
        )

        overlap = (
            query_terms
            &
            document_terms
        )

        return len(overlap) / len(
            query_terms
        )

    # ========================================================
    # COMBINE ORIGINAL + EXPANDED QUERY
    # ========================================================

    def retrieve_semantic_candidates(
        self,
        original_query,
        expanded_query,
        top_k=SEMANTIC_TOP_K
    ):
        """
        Retrieve using both the original and expanded query.

        This prevents query expansion from completely
        replacing important wording from the original question.
        """

        original_results = (
            self.semantic_retrieve(
                original_query,
                top_k=top_k
            )
        )

        expanded_results = (
            self.semantic_retrieve(
                expanded_query,
                top_k=top_k
            )
        )

        candidates = {}

        # ----------------------------------------------------
        # Add original-query results
        # ----------------------------------------------------

        for result in original_results:

            index_id = result[
                "index_id"
            ]

            candidates.setdefault(
                index_id,
                {
                    **result,
                    "original_score": 0.0,
                    "expanded_score": 0.0,
                }
            )

            candidates[
                index_id
            ][
                "original_score"
            ] = max(
                candidates[
                    index_id
                ][
                    "original_score"
                ],
                result[
                    "score"
                ]
            )

        # ----------------------------------------------------
        # Add expanded-query results
        # ----------------------------------------------------

        for result in expanded_results:

            index_id = result[
                "index_id"
            ]

            candidates.setdefault(
                index_id,
                {
                    **result,
                    "original_score": 0.0,
                    "expanded_score": 0.0,
                }
            )

            candidates[
                index_id
            ][
                "expanded_score"
            ] = max(
                candidates[
                    index_id
                ][
                    "expanded_score"
                ],
                result[
                    "score"
                ]
            )

        # ----------------------------------------------------
        # Calculate combined scores
        # ----------------------------------------------------

        final_candidates = []

        for result in candidates.values():

            original_score = result[
                "original_score"
            ]

            expanded_score = result[
                "expanded_score"
            ]

            lexical = self.lexical_score(
                original_query,
                result
            )

            # ------------------------------------------------
            # Weighted score
            #
            # Original query:
            # 45%
            #
            # Expanded query:
            # 35%
            #
            # Legal-term overlap:
            # 20%
            # ------------------------------------------------

            combined_score = (
                0.45 * original_score
                +
                0.35 * expanded_score
                +
                0.20 * lexical
            )

            result[
                "lexical_score"
            ] = lexical

            result[
                "combined_score"
            ] = combined_score

            result[
                "score"
            ] = combined_score

            result[
                "query_source"
            ] = "original + expanded"

            final_candidates.append(
                result
            )

        final_candidates.sort(
            key=lambda x: x[
                "combined_score"
            ],
            reverse=True
        )

        return final_candidates

    # ========================================================
    # STRUCTURAL EXPANSION
    # ========================================================

    def expand_structure(
        self,
        semantic_results,
        original_query,
        max_sections=MAX_SECTIONS
    ):
        """
        Structure-aware expansion.

        Instead of treating every chunk from the same CFR section
        equally, this method gives a bonus to chunks whose legal
        paragraph/subparagraph matches important terms in the query.

        Example:

            Query mentions "ceases to be agricultural"
            -> prefer 7 CFR 781.3 (i)

            Query mentions "becomes agricultural"
            -> prefer 7 CFR 781.3 (d)

            Query mentions "filed"
            -> prefer 7 CFR 781.3 (a)
        """

        expanded = []

        seen_sections = set()

        query_terms = important_terms(
            original_query
        )

        for result in semantic_results:

            metadata = result.get(
                "metadata",
                {}
            )

            citation = metadata.get(
                "citation"
            )

            if not citation:
                continue

            if citation in seen_sections:
                continue

            seen_sections.add(
                citation
            )

            related_ids = self.section_lookup.get(
                citation,
                []
            )

            section_score = result.get(
                "combined_score",
                result.get("score", 0.0)
            )

            for index_id in related_ids:

                item = self.metadata[
                    index_id
                ]

                item_metadata = item.get(
                    "metadata",
                    {}
                )

                structural_result = {
                    "index_id": index_id,

                    "text": item.get(
                        "text",
                        ""
                    ),

                    "retrieval_text": item.get(
                        "retrieval_text",
                        ""
                    ),

                    "metadata": item_metadata,

                    "retrieval_type":
                        "structural",

                    "section_score":
                        section_score,

                    "original_score":
                        result.get(
                            "original_score",
                            0.0
                        ),

                    "expanded_score":
                        result.get(
                            "expanded_score",
                            0.0
                        ),
                }

                # ------------------------------------------------
                # Lexical relevance
                # ------------------------------------------------

                lexical = self.lexical_score(
                    original_query,
                    structural_result
                )

                structural_result[
                    "lexical_score"
                ] = lexical

                # ------------------------------------------------
                # Build structural label
                # ------------------------------------------------

                paragraph = item_metadata.get(
                    "paragraph"
                )

                subparagraph = item_metadata.get(
                    "subparagraph"
                )

                clause = item_metadata.get(
                    "clause"
                )

                structural_label = " ".join(
                    part
                    for part in [
                        paragraph,
                        subparagraph,
                        clause
                    ]
                    if part
                )

                # ------------------------------------------------
                # Calculate relevance of the actual paragraph
                # ------------------------------------------------

                paragraph_text = " ".join(
                    [
                        structural_label,
                        item_metadata.get(
                            "section_title",
                            ""
                        ),
                        item.get(
                            "text",
                            ""
                        ),
                        item.get(
                            "retrieval_text",
                            ""
                        ),
                    ]
                )

                paragraph_terms = important_terms(
                    paragraph_text
                )

                paragraph_overlap = (
                    len(
                        query_terms
                        &
                        paragraph_terms
                    )
                    /
                    len(query_terms)
                    if query_terms
                    else 0.0
                )

                # ------------------------------------------------
                # Structural paragraph matching
                # ------------------------------------------------

                structural_bonus = 0.0

                # Strong bonus when the actual paragraph has
                # strong lexical relevance to the question.
                if paragraph_overlap >= 0.50:
                    structural_bonus = 0.20

                elif paragraph_overlap >= 0.25:
                    structural_bonus = 0.10

                # ------------------------------------------------
                # Final structural score
                #
                # 65% parent semantic score
                # 20% lexical relevance
                # 15% paragraph relevance
                # ------------------------------------------------

                structural_score = (
                    0.65 * section_score
                    +
                    0.20 * lexical
                    +
                    0.15 * paragraph_overlap
                    +
                    structural_bonus
                )

                structural_result[
                    "paragraph_overlap"
                ] = paragraph_overlap

                structural_result[
                    "structural_bonus"
                ] = structural_bonus

                structural_result[
                    "score"
                ] = structural_score

                expanded.append(
                    structural_result
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
        expanded_query=None,
        top_k=5,
        max_related=10
    ):
        """
        Complete retrieval pipeline.
        """

        # ----------------------------------------------------
        # Query expansion
        # ----------------------------------------------------

        if expanded_query is None:

            expanded_query = (
                expand_query(
                    query
                )
            )

        # ----------------------------------------------------
        # Step 1:
        # Semantic retrieval using both queries
        # ----------------------------------------------------

        semantic_candidates = (
            self.retrieve_semantic_candidates(
                query,
                expanded_query,
                top_k=SEMANTIC_TOP_K
            )
        )

        # ----------------------------------------------------
        # Step 2:
        # Structural expansion
        # ----------------------------------------------------

        structural_candidates = (
            self.expand_structure(
                semantic_candidates,
                query,
                max_sections=MAX_SECTIONS
            )
        )

        # ----------------------------------------------------
        # Step 3:
        # Combine results
        # ----------------------------------------------------

        combined = []

        seen = set()

        # ----------------------------------------------------
        # Semantic candidates
        # ----------------------------------------------------

        for result in semantic_candidates:

            index_id = result[
                "index_id"
            ]

            if index_id in seen:
                continue

            combined.append(
                result
            )

            seen.add(
                index_id
            )

        # ----------------------------------------------------
        # Structural candidates
        # ----------------------------------------------------

        for result in structural_candidates:

            index_id = result[
                "index_id"
            ]

            if index_id in seen:
                continue

            combined.append(
                result
            )

            seen.add(
                index_id
            )

        # ----------------------------------------------------
        # Step 4:
        # Final ranking
        # ----------------------------------------------------

        combined.sort(
            key=lambda x: x[
                "score"
            ],
            reverse=True
        )

        # ----------------------------------------------------
        # Step 5:
        # Return requested amount
        # ----------------------------------------------------

        return combined[
            :(top_k + max_related)
        ]


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    retriever = StructureAwareRetriever()

    test_questions = [
        "What happens when land becomes agricultural land after being acquired?",

        "What must a foreign person do when land subsequently becomes agricultural land?",

        "When must a report be filed for agricultural land?",

        "foreign person land subsequently becomes agricultural land report within 90 days",

        "What reporting requirement applies when land becomes agricultural land?",
    ]

    print()
    print("=" * 80)
    print("STRUCTURE-AWARE RETRIEVAL TEST")
    print("=" * 80)

    for query in test_questions:

        print()
        print("=" * 80)
        print("ORIGINAL QUERY:")
        print(query)

        expanded = expand_query(
            query
        )

        print()
        print("EXPANDED QUERY:")
        print(expanded)

        results = retriever.retrieve(
            query,
            expanded_query=expanded,
            top_k=5,
            max_related=10
        )

        print()
        print("FINAL RESULTS")
        print("=" * 80)

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

            section_title = metadata.get(
                "section_title"
            )

            location = citation

            if paragraph:
                location += (
                    f" {paragraph}"
                )

            if subparagraph:
                location += (
                    f" {subparagraph}"
                )

            print(
                f"\n{number}. {location}"
            )

            print(
                f"   Score: "
                f"{result['score']:.4f}"
            )

            print(
                f"   Original score: "
                f"{result.get('original_score', 0.0):.4f}"
            )

            print(
                f"   Expanded score: "
                f"{result.get('expanded_score', 0.0):.4f}"
            )

            print(
                f"   Lexical score: "
                f"{result.get('lexical_score', 0.0):.4f}"
            )

            print(
                f"   Type: "
                f"{result.get('retrieval_type')}"
            )

            print(
                f"   Section: "
                f"{section_title}"
            )

            print(
                f"   Text: "
                f"{result['text'][:500]}"
            )

            print("-" * 80)