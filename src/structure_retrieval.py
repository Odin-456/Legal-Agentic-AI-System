from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


VECTORSTORE_DIR = (
    PROJECT_ROOT
    / "data"
    / "vectorstore"
    / "property_law_structure"
)

COLLECTION_NAME = "property_law_structure"

EMBEDDING_MODEL = "text-embedding-3-large"

CANDIDATE_K = 50

SEMANTIC_WEIGHT = 0.60
STRUCTURE_WEIGHT = 0.40


# ============================================================
# VECTORSTORE
# ============================================================

# Load the existing structure-aware Chroma vectorstore
def load_vectorstore():

    if not VECTORSTORE_DIR.exists():

        raise FileNotFoundError(
            f"Vectorstore not found: {VECTORSTORE_DIR}"
        )

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )

    print(
        f"Loaded vectorstore from: {VECTORSTORE_DIR}"
    )

    return vectorstore


# ============================================================
# STRUCTURE-AWARE RETRIEVER
# ============================================================

class StructureAwareRetriever:

    def __init__(self, candidate_k=CANDIDATE_K):

        self.candidate_k = candidate_k

        # Load the existing vectorstore
        self.vectorstore = load_vectorstore()

        print(
            "Structure-aware retriever ready."
        )


    # ========================================================
    # SEMANTIC RETRIEVAL
    # ========================================================

    # Retrieve the most semantically similar chunks
    def retrieve_candidates(self, query):

        results = self.vectorstore.similarity_search_with_score(
            query,
            k=self.candidate_k
        )

        return results


    # ========================================================
    # STRUCTURE SCORE
    # ========================================================

    # Calculate a score based on whether chunks belong
    # to the same legal structure
    def calculate_structure_score(
        self,
        document,
        reference_document
    ):

        metadata = document.metadata
        reference_metadata = reference_document.metadata

        score = 0.0


        # Same Act
        if metadata.get("act") == reference_metadata.get("act"):

            score += 0.4


        # Same section
        if metadata.get("section") == reference_metadata.get("section"):

            score += 0.4


        # Same subsection
        if (
            metadata.get("subsection")
            and metadata.get("subsection")
            == reference_metadata.get("subsection")
        ):

            score += 0.2


        return score


    # ========================================================
    # RERANKING
    # ========================================================

    # Combine semantic similarity with legal structure
    def rerank(
        self,
        query,
        candidates
    ):

        if not candidates:

            return []


        # Use the strongest semantic result as
        # the reference document
        reference_document = candidates[0][0]

        reranked = []


        for document, distance in candidates:

            # Convert Chroma distance into a similarity-like score
            semantic_score = 1 / (1 + distance)

            # Calculate structure similarity
            structure_score = self.calculate_structure_score(
                document,
                reference_document
            )

            # Combine the two scores
            final_score = (
                SEMANTIC_WEIGHT * semantic_score
                +
                STRUCTURE_WEIGHT * structure_score
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


        # Highest score first
        reranked.sort(
            key=lambda item: item["final_score"],
            reverse=True
        )

        return reranked


    # ========================================================
    # SELECT DIFFERENT LEGAL SECTIONS
    # ========================================================

    # Select the strongest result from each legal section
    def select_sections(
        self,
        reranked,
        top_k
    ):

        selected = []

        seen_sections = set()


        for item in reranked:

            document = item["document"]

            metadata = document.metadata

            act = metadata.get("act")
            section = metadata.get("section")

            section_key = (
                act,
                section
            )


            # Avoid returning many chunks
            # from the same legal section
            if section_key in seen_sections:

                continue


            seen_sections.add(section_key)

            selected.append(item)


            if len(selected) >= top_k:

                break


        return selected


    # ========================================================
    # GET CHUNKS FROM SAME LEGAL SECTION
    # ========================================================

    # Retrieve all stored chunks belonging to
    # the same Act and section directly from Chroma
    def get_section_documents(
        self,
        act,
        section
    ):

        data = self.vectorstore.get(
            where={
                "$and": [
                    {
                        "act": act
                    },
                    {
                        "section": section
                    },
                ]
            },
            include=[
                "documents",
                "metadatas",
            ],
        )


        documents = []


        stored_documents = data.get(
            "documents",
            []
        )

        stored_metadata = data.get(
            "metadatas",
            []
        )


        for page_content, metadata in zip(
            stored_documents,
            stored_metadata
        ):

            documents.append(
                Document(
                    page_content=page_content,
                    metadata=metadata or {},
                )
            )


        return documents


    # ========================================================
    # SELECT RELATED CHUNKS
    # ========================================================

    # Select additional chunks from the same legal section
    def select_related_chunks(
        self,
        query,
        selected_document,
        section_documents,
        max_related
    ):

        if not section_documents:

            return []


        reference_metadata = (
            selected_document.metadata
        )


        reference_subsection = (
            reference_metadata.get("subsection")
        )


        # If the main result belongs to a subsection,
        # prefer chunks from that subsection
        if reference_subsection:

            subsection_documents = [
                document
                for document in section_documents
                if document.metadata.get("subsection")
                == reference_subsection
            ]

            if subsection_documents:

                section_documents = subsection_documents


        # ----------------------------------------------------
        # Score chunks based on simple structural relevance
        # ----------------------------------------------------

        scored_documents = []


        query_words = set(
            query.lower().split()
        )


        for document in section_documents:

            text = document.page_content.lower()

            heading = document.metadata.get(
                "heading",
                ""
            ).lower()

            subsection = document.metadata.get(
                "subsection",
                ""
            ).lower()


            # Simple text overlap
            text_words = set(
                text.split()
            )

            if query_words:

                text_overlap = (
                    len(query_words & text_words)
                    / len(query_words)
                )

            else:

                text_overlap = 0.0


            # Heading overlap
            heading_words = set(
                heading.split()
            )

            if query_words:

                heading_overlap = (
                    len(query_words & heading_words)
                    / len(query_words)
                )

            else:

                heading_overlap = 0.0


            # Subsection match
            subsection_score = 0.0

            if (
                reference_subsection
                and subsection
                == reference_subsection.lower()
            ):

                subsection_score = 1.0


            # Keep the same weighting:
            # 70% text
            # 20% heading
            # 10% subsection
            final_score = (
                0.70 * text_overlap
                +
                0.20 * heading_overlap
                +
                0.10 * subsection_score
            )


            scored_documents.append(
                (
                    document,
                    final_score
                )
            )


        # Sort by relevance
        scored_documents.sort(
            key=lambda item: item[1],
            reverse=True
        )


        # Select the strongest related chunks
        selected_documents = [
            document
            for document, score
            in scored_documents[:max_related]
        ]


        # ----------------------------------------------------
        # Restore legal order
        # ----------------------------------------------------

        def legal_order(document):

            metadata = document.metadata

            return (
                metadata.get(
                    "subsection",
                    ""
                ),
                metadata.get(
                    "paragraph",
                    ""
                ),
                metadata.get(
                    "nested_paragraph",
                    ""
                ),
            )


        selected_documents.sort(
            key=legal_order
        )


        return selected_documents


    # ========================================================
    # MAIN RETRIEVAL PIPELINE
    # ========================================================

    def retrieve(
        self,
        query,
        top_k=5,
        max_related=10
    ):

        # Step 1:
        # Semantic retrieval
        candidates = self.retrieve_candidates(
            query
        )


        if not candidates:

            return []


        # Step 2:
        # Structure-aware reranking
        reranked = self.rerank(
            query,
            candidates
        )


        # Step 3:
        # Select strongest legal sections
        selected_sections = self.select_sections(
            reranked,
            top_k
        )


        results = []


        # Step 4:
        # Retrieve related chunks from
        # the same legal section
        for item in selected_sections:

            document = item["document"]

            metadata = document.metadata

            act = metadata.get(
                "act"
            )

            section = metadata.get(
                "section"
            )


            # Get all chunks for this section
            section_documents = (
                self.get_section_documents(
                    act,
                    section
                )
            )


            # Select the most relevant related chunks
            related_documents = (
                self.select_related_chunks(
                    query,
                    document,
                    section_documents,
                    max_related
                )
            )


            results.append(
                {
                    "document": document,
                    "metadata": metadata,
                    "score": item["final_score"],
                    "semantic_score": item["semantic_score"],
                    "structure_score": item["structure_score"],
                    "related_documents": related_documents,
                }
            )


        return results


# ============================================================
# FORMAT RESULTS
# ============================================================

# Convert retrieval results into readable text
def format_results(results):

    if not results:

        return (
            "No relevant NSW property-law "
            "legislation was found."
        )


    context = []


    for i, result in enumerate(
        results,
        1
    ):

        metadata = result.get(
            "metadata",
            {}
        )

        act = metadata.get(
            "act",
            "Unknown Act"
        )

        section = metadata.get(
            "section",
            "Unknown section"
        )

        heading = metadata.get(
            "heading",
            ""
        )


        documents = result.get(
            "related_documents",
            []
        )


        if not documents:

            documents = [
                result["document"]
            ]


        legal_text = "\n".join(
            document.page_content
            for document in documents
        )


        context.append(
            f"""
SOURCE {i}
ACT AND SECTION: {act} § {section}
HEADING: {heading}
LEGAL TEXT:
{legal_text}
"""
        )


    return "\n".join(context)


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    retriever = StructureAwareRetriever()


    question = (
        "What responsibilities does an association "
        "have under the Community Land Management Act 2021?"
    )


    results = retriever.retrieve(
        question,
        top_k=5,
        max_related=10
    )


    print()
    print("=" * 80)
    print("RETRIEVAL RESULTS")
    print("=" * 80)


    print(
        format_results(results)
    )