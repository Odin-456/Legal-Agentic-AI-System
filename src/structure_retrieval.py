from pathlib import Path
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

BASELINE_COLLECTION = "property_law_baseline"
STRUCTURE_COLLECTION = "property_law_structure"


# ============================================================
# EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL = "text-embedding-3-small"


# ============================================================
# TEST QUERY
# ============================================================

TEST_QUERY = (
    "What exceptions apply to the title of a registered proprietor?"
)


# ============================================================
# STOP WORDS
# ============================================================

STOP_WORDS = {
    "what",
    "which",
    "who",
    "where",
    "when",
    "why",
    "how",
    "does",
    "do",
    "is",
    "are",
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "for",
    "from",
    "and",
    "or",
    "with",
    "under",
    "apply",
    "applies",
}


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings():

    print("\nCreating embedding model...")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    print(
        f"Embedding model: {EMBEDDING_MODEL}"
    )

    return embeddings


# ============================================================
# LOAD BASELINE DOCUMENTS
# ============================================================

def load_baseline_documents():

    print("\n" + "=" * 80)
    print("BUILDING BASELINE DOCUMENTS")
    print("=" * 80)

    sections = parse_property_law_corpus()

    print(
        f"\nSections parsed: {len(sections)}"
    )

    documents = create_chunks(
        sections
    )

    print(
        f"Baseline retrieval documents: "
        f"{len(documents)}"
    )

    return documents


# ============================================================
# LOAD STRUCTURE-AWARE DOCUMENTS
# ============================================================

def load_structure_documents():

    print("\n" + "=" * 80)
    print("BUILDING STRUCTURE-AWARE DOCUMENTS")
    print("=" * 80)

    documents = parse_structure_corpus()

    print(
        f"\nStructure chunks created: "
        f"{len(documents)}"
    )

    documents = remove_duplicates(
        documents
    )

    print(
        f"Unique structure chunks: "
        f"{len(documents)}"
    )

    return documents


# ============================================================
# RESET VECTORSTORE
# ============================================================

def reset_vectorstore(path):

    if path.exists():

        print(
            "\nRemoving existing vectorstore:"
        )

        print(f"  {path}")

        shutil.rmtree(path)

        print("  Removed.")


# ============================================================
# BUILD VECTORSTORE
# ============================================================

def build_vectorstore(
    documents,
    persist_directory,
    collection_name,
    embeddings,
):

    print("\n" + "-" * 80)

    print(
        f"Creating vectorstore: "
        f"{collection_name}"
    )

    print("-" * 80)

    print(
        f"Documents to embed: "
        f"{len(documents)}"
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(
            persist_directory
        ),
    )

    print(
        "Vectorstore created successfully."
    )

    return vectorstore


# ============================================================
# TOKENISE QUERY
# ============================================================

def get_query_terms(query):

    words = re.findall(
        r"[A-Za-z0-9]+",
        query.lower(),
    )

    terms = []

    for word in words:

        if (
            len(word) > 2
            and word not in STOP_WORDS
        ):
            terms.append(word)

    return set(terms)


# ============================================================
# TEXT MATCH SCORE
# ============================================================

def text_match_score(
    query_terms,
    text,
):

    if not text:
        return 0.0

    text_words = set(
        re.findall(
            r"[A-Za-z0-9]+",
            text.lower(),
        )
    )

    if not query_terms:
        return 0.0

    matched = (
        query_terms
        & text_words
    )

    return (
        len(matched)
        / len(query_terms)
    )


# ============================================================
# STRUCTURAL RERANK SCORE
# ============================================================

def calculate_structure_score(
    query,
    document,
):

    query_terms = get_query_terms(
        query
    )

    metadata = document.metadata

    heading = metadata.get(
        "heading",
        "",
    )

    subsection = metadata.get(
        "subsection",
        "",
    )

    paragraph = metadata.get(
        "paragraph",
        "",
    )

    nested_paragraph = metadata.get(
        "nested_paragraph",
        "",
    )

    page_content = (
        document.page_content
    )

    heading_score = text_match_score(
        query_terms,
        heading,
    )

    subsection_score = text_match_score(
        query_terms,
        subsection,
    )

    paragraph_score = text_match_score(
        query_terms,
        paragraph,
    )

    nested_score = text_match_score(
        query_terms,
        nested_paragraph,
    )

    text_score = text_match_score(
        query_terms,
        page_content,
    )

    hierarchy = metadata.get(
        "hierarchy",
        "",
    )

    hierarchy_parts = [
        part.strip()
        for part in hierarchy.split(">")
        if part.strip()
    ]

    hierarchy_depth = len(
        hierarchy_parts
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

    score = (
        0.30 * heading_score
        + 0.10 * subsection_score
        + 0.10 * paragraph_score
        + 0.05 * nested_score
        + 0.30 * text_score
        + 0.15 * hierarchy_score
    )

    return score


# ============================================================
# BASELINE RETRIEVAL
# ============================================================

def retrieve_baseline(
    vectorstore,
    query,
    k=5,
):

    results = (
        vectorstore
        .similarity_search_with_score(
            query,
            k=k,
        )
    )

    return results


# ============================================================
# STRUCTURE-AWARE CANDIDATE RETRIEVAL
# ============================================================

def retrieve_structure_candidates(
    vectorstore,
    query,
    candidate_k=50,
):

    results = (
        vectorstore
        .similarity_search_with_score(
            query,
            k=candidate_k,
        )
    )

    return results


# ============================================================
# STRUCTURE-AWARE RERANKING
# ============================================================

def rerank_structure_candidates(
    query,
    candidates,
    final_k=5,
):

    reranked = []

    for document, distance in candidates:

        semantic_score = (
            1.0
            / (1.0 + float(distance))
        )

        structure_score = (
            calculate_structure_score(
                query,
                document,
            )
        )

        final_score = (
            0.65 * semantic_score
            + 0.35 * structure_score
        )

        reranked.append(
            {
                "document": document,
                "distance": distance,
                "semantic_score": semantic_score,
                "structure_score": structure_score,
                "final_score": final_score,
            }
        )

    reranked.sort(
        key=lambda x: x["final_score"],
        reverse=True,
    )

    return reranked


# ============================================================
# GROUP BY SECTION
# ============================================================

def group_by_section(
    reranked_candidates,
    final_k=5,
):

    section_groups = {}

    for result in reranked_candidates:

        document = result["document"]

        metadata = document.metadata

        act = metadata.get(
            "act",
            "",
        )

        section = metadata.get(
            "section",
            "",
        )

        key = (
            act,
            section,
        )

        if key not in section_groups:

            section_groups[key] = result

        else:

            existing = section_groups[key]

            if (
                result["final_score"]
                > existing["final_score"]
            ):

                section_groups[key] = result

    sections = list(
        section_groups.values()
    )

    sections.sort(
        key=lambda x: x["final_score"],
        reverse=True,
    )

    return sections[:final_k]


# ============================================================
# EXPAND PARENT CONTEXT
# ============================================================

def expand_parent_context(
    ranked_sections,
    all_structure_documents,
    max_related=10,
):

    section_index = {}

    for document in all_structure_documents:

        metadata = document.metadata

        key = (
            metadata.get("act", ""),
            metadata.get("section", ""),
        )

        if key not in section_index:

            section_index[key] = []

        section_index[key].append(
            document
        )

    expanded = []

    for rank, result in enumerate(
        ranked_sections,
        start=1,
    ):

        document = result["document"]

        metadata = document.metadata

        key = (
            metadata.get("act", ""),
            metadata.get("section", ""),
        )

        context_chunks = list(
            section_index.get(
                key,
                [],
            )
        )

        def structural_order(doc):

            md = doc.metadata

            return (
                str(md.get("subsection", "")),
                str(md.get("paragraph", "")),
                str(md.get("nested_paragraph", "")),
            )

        context_chunks.sort(
            key=structural_order
        )

        if max_related > 0:

            context_chunks = context_chunks[
                :max_related
            ]

        expanded.append(
            {
                "rank": rank,
                "result": result,
                "context_chunks": context_chunks,
            }
        )

    return expanded


# ============================================================
# RETRIEVER CLASS
# ============================================================

class StructureAwareRetriever:

    def __init__(self):

        print(
            "Initialising Structure-Aware Retriever..."
        )

        self.embeddings = create_embeddings()

        self.baseline_vectorstore = None
        self.structure_vectorstore = None

        self.structure_documents = None

        self._load_vectorstores()

    # --------------------------------------------------------
    # LOAD EXISTING VECTORSTORES
    # --------------------------------------------------------

    def _load_vectorstores(self):

        if not BASELINE_VECTORSTORE.exists():

            raise FileNotFoundError(
                f"Baseline vectorstore not found: "
                f"{BASELINE_VECTORSTORE}\n"
                "Run structure_retrieval.py first "
                "to build the vectorstores."
            )

        if not STRUCTURE_VECTORSTORE.exists():

            raise FileNotFoundError(
                f"Structure vectorstore not found: "
                f"{STRUCTURE_VECTORSTORE}\n"
                "Run structure_retrieval.py first "
                "to build the vectorstores."
            )

        self.baseline_vectorstore = Chroma(
            collection_name=BASELINE_COLLECTION,
            persist_directory=str(
                BASELINE_VECTORSTORE
            ),
            embedding_function=self.embeddings,
        )

        self.structure_vectorstore = Chroma(
            collection_name=STRUCTURE_COLLECTION,
            persist_directory=str(
                STRUCTURE_VECTORSTORE
            ),
            embedding_function=self.embeddings,
        )

        self.structure_documents = (
            load_structure_documents()
        )

    # --------------------------------------------------------
    # BASELINE SEMANTIC RETRIEVAL
    # --------------------------------------------------------

    def semantic_retrieve(
        self,
        query,
        top_k=5,
    ):

        results = (
            self.baseline_vectorstore
            .similarity_search_with_score(
                query,
                k=top_k,
            )
        )

        formatted = []

        for document, distance in results:

            formatted.append(
                {
                    "text": document.page_content,
                    "metadata": document.metadata,
                    "distance": float(distance),
                    "score": (
                        1.0
                        / (1.0 + float(distance))
                    ),
                    "retrieval_type": "baseline",
                }
            )

        return formatted

    # --------------------------------------------------------
    # STRUCTURE-AWARE RETRIEVAL
    # --------------------------------------------------------

    def retrieve(
        self,
        query,
        expanded_query=None,
        top_k=5,
        max_related=10,
    ):

        retrieval_query = (
            expanded_query
            if expanded_query
            else query
        )

        candidates = (
            retrieve_structure_candidates(
                self.structure_vectorstore,
                retrieval_query,
                candidate_k=50,
            )
        )

        reranked = (
            rerank_structure_candidates(
                retrieval_query,
                candidates,
            )
        )

        ranked_sections = (
            group_by_section(
                reranked,
                final_k=top_k,
            )
        )

        expanded = (
            expand_parent_context(
                ranked_sections,
                self.structure_documents,
                max_related=max_related,
            )
        )

        formatted = []

        for item in expanded:

            main_result = item["result"]

            main_document = (
                main_result["document"]
            )

            metadata = (
                main_document.metadata
            )

            context_chunks = item[
                "context_chunks"
            ]

            combined_text = "\n\n".join(
                document.page_content
                for document in context_chunks
            )

            formatted.append(
                {
                    "text": combined_text,
                    "metadata": metadata,
                    "distance": float(
                        main_result["distance"]
                    ),
                    "semantic_score": float(
                        main_result["semantic_score"]
                    ),
                    "structure_score": float(
                        main_result["structure_score"]
                    ),
                    "score": float(
                        main_result["final_score"]
                    ),
                    "final_score": float(
                        main_result["final_score"]
                    ),
                    "retrieval_type": "structure-aware",
                    "context_chunks": context_chunks,
                }
            )

        return formatted


# ============================================================
# EXPERIMENT
# ============================================================

def main():

    print("\n")

    print("=" * 80)

    print(
        "STRUCTURE-AWARE PROPERTY LAW "
        "RETRIEVAL EXPERIMENT"
    )

    print("=" * 80)

    print("\nTest query:")
    print(TEST_QUERY)

    embeddings = create_embeddings()

    baseline_documents = (
        load_baseline_documents()
    )

    structure_documents = (
        load_structure_documents()
    )

    reset_vectorstore(
        BASELINE_VECTORSTORE
    )

    reset_vectorstore(
        STRUCTURE_VECTORSTORE
    )

    baseline_vectorstore = (
        build_vectorstore(
            documents=baseline_documents,
            persist_directory=BASELINE_VECTORSTORE,
            collection_name=BASELINE_COLLECTION,
            embeddings=embeddings,
        )
    )

    structure_vectorstore = (
        build_vectorstore(
            documents=structure_documents,
            persist_directory=STRUCTURE_VECTORSTORE,
            collection_name=STRUCTURE_COLLECTION,
            embeddings=embeddings,
        )
    )

    baseline_results = (
        retrieve_baseline(
            baseline_vectorstore,
            TEST_QUERY,
            k=5,
        )
    )

    candidates = (
        retrieve_structure_candidates(
            structure_vectorstore,
            TEST_QUERY,
            candidate_k=50,
        )
    )

    reranked_candidates = (
        rerank_structure_candidates(
            TEST_QUERY,
            candidates,
        )
    )

    structure_results = (
        group_by_section(
            reranked_candidates,
            final_k=5,
        )
    )

    expanded_results = (
        expand_parent_context(
            structure_results,
            structure_documents,
            max_related=10,
        )
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    print(
        f"\nBaseline documents: "
        f"{len(baseline_documents)}"
    )

    print(
        f"Structure documents: "
        f"{len(structure_documents)}"
    )

    print(
        f"\nBaseline top results: "
        f"{len(baseline_results)}"
    )

    print(
        f"Structure-aware top results: "
        f"{len(structure_results)}"
    )

    print(
        f"Expanded results: "
        f"{len(expanded_results)}"
    )


if __name__ == "__main__":
    main()