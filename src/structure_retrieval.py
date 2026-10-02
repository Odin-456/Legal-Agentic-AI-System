import re
from pathlib import Path
from difflib import SequenceMatcher

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


# ============================================================
# PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VECTORSTORE_DIR = (
    BASE_DIR
    / "data"
    / "vectorstore"
    / "property_law_structure"
)

COLLECTION_NAME = "property_law_structure"

EMBEDDING_MODEL = "text-embedding-3-large"

DEFAULT_TOP_K = 5
CANDIDATE_K = 50
MAX_RELATED = 5


# ============================================================
# LEGAL TERMS
# ============================================================

STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "what",
    "when",
    "where",
    "how",
    "can",
    "could",
    "would",
    "should",
    "does",
    "do",
    "did",
    "my",
    "me",
    "i",
    "you",
    "your",
    "about",
    "for",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "with",
    "under",
    "from",
}


LEGAL_TERM_GROUPS = {
    "landlord": ["landlord", "lessor"],
    "tenant": ["tenant", "lessee"],
    "rent": ["rent", "rental"],
    "lease": ["lease", "leasing", "tenancy"],
    "repair": ["repair", "repairs", "maintenance"],
    "notice": ["notice", "notices"],
    "termination": ["termination", "terminate", "terminated"],
    "strata": ["strata", "owners corporation"],
    "property": ["property", "land"],
}


# ============================================================
# QUERY HELPERS
# ============================================================

def get_query_terms(query):
    words = re.findall(r"\b[a-zA-Z]+\b", query.lower())

    return [
        word
        for word in words
        if word not in STOP_WORDS and len(word) > 2
    ]


def correct_query_terms(query):
    """
    Add common legal terms where useful.
    """

    corrected = query.lower()

    for main_term, related_terms in LEGAL_TERM_GROUPS.items():
        if any(term in corrected for term in related_terms):
            corrected += " " + " ".join(related_terms)

    return corrected


def text_match_score(query, text):
    """
    Measures how strongly query terms appear in the text.
    """

    if not text:
        return 0.0

    query_terms = get_query_terms(correct_query_terms(query))
    text_lower = text.lower()

    if not query_terms:
        return 0.0

    matches = 0

    for term in query_terms:
        if term in text_lower:
            matches += 1

    return matches / len(query_terms)


# ============================================================
# REFERENCE EXTRACTION
# ============================================================

def extract_section_reference(query):
    """
    Detect explicit section references such as:
    section 45
    s 45
    s.45
    """

    match = re.search(
        r"\b(?:section|s)\.?\s*(\d+[A-Za-z]?)\b",
        query,
        re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return None


def extract_subsection_reference(query):
    """
    Detect subsection references such as:
    subsection 45(2)
    section 45(2)
    s 45(2)
    """

    match = re.search(
        r"\b(?:section|s|subsection)\.?\s*(\d+[A-Za-z]?)\s*\((\d+)\)",
        query,
        re.IGNORECASE,
    )

    if match:
        return f"{match.group(1)}({match.group(2)})"

    return None


def extract_act_reference(query):
    """
    Detect common references to the NSW Acts.
    """

    query_lower = query.lower()

    act_names = {
        "real property act 1900": "Real Property Act 1900",
        "conveyancing act 1919": "Conveyancing Act 1919",
        "residential tenancies act 2010": "Residential Tenancies Act 2010",
        "strata schemes management act 2015": "Strata Schemes Management Act 2015",
        "community land management act 2021": "Community Land Management Act 2021",
    }

    for search_name, actual_name in act_names.items():
        if search_name in query_lower:
            return actual_name

    return None


# ============================================================
# STRUCTURE SCORING
# ============================================================

def calculate_structure_score(query, document):
    """
    Calculate a score based on legal document structure.

    The weighting remains the same as the original
    structure-aware retrieval method.
    """

    metadata = document.metadata

    act = metadata.get("act", "")
    section = metadata.get("section", "")
    heading = metadata.get("heading", "")
    subsection = metadata.get("subsection", "")
    paragraph = metadata.get("paragraph", "")
    nested_paragraph = metadata.get("nested_paragraph", "")

    query_terms = get_query_terms(correct_query_terms(query))

    text_score = text_match_score(
        query,
        document.page_content,
    )

    heading_score = text_match_score(
        query,
        heading,
    )

    paragraph_score = text_match_score(
        query,
        paragraph,
    )

    nested_score = text_match_score(
        query,
        nested_paragraph,
    )

    act_score = 0.0
    requested_act = extract_act_reference(query)

    if requested_act:
        if act.lower() == requested_act.lower():
            act_score = 1.0

    section_score = 0.0
    requested_section = extract_section_reference(query)

    if requested_section:
        if str(section).lower() == requested_section.lower():
            section_score = 1.0

    subsection_score = 0.0
    requested_subsection = extract_subsection_reference(query)

    if requested_subsection:
        if str(subsection).lower() == requested_subsection.lower():
            subsection_score = 1.0

    score = (
        0.50 * text_score
        + 0.22 * heading_score
        + 0.10 * act_score
        + 0.08 * section_score
        + 0.05 * subsection_score
        + 0.03 * paragraph_score
        + 0.02 * nested_score
    )

    return score


# ============================================================
# VECTORSTORE
# ============================================================

def load_vectorstore():
    """
    Load the existing structure-aware Chroma vectorstore.
    """

    load_dotenv(BASE_DIR / ".env")

    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            f"Vectorstore not found: {VECTORSTORE_DIR}"
        )

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
    )


# ============================================================
# DOCUMENT HELPERS
# ============================================================

def get_section_key(document):
    """
    Identify the legal section belonging to a document.
    """

    metadata = document.metadata

    return (
        metadata.get("act", ""),
        metadata.get("section", ""),
    )


def structure_order(document):
    """
    Used to restore the original legal order.
    """

    metadata = document.metadata

    return (
        str(metadata.get("subsection", "")),
        str(metadata.get("paragraph", "")),
        str(metadata.get("nested_paragraph", "")),
    )


# ============================================================
# SECTION SELECTION
# ============================================================

def select_sections(scored_documents, top_k):
    """
    Select the strongest legal sections.

    Only one strongest document is initially selected
    from each Act/section.
    """

    selected = {}
    
    for document, score in scored_documents:
        key = get_section_key(document)

        if key not in selected:
            selected[key] = (document, score)

    ranked = sorted(
        selected.values(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        document
        for document, score in ranked[:top_k]
    ]


# ============================================================
# RELATED CHUNKS
# ============================================================

def get_section_documents(vectorstore, act, section):
    """
    Get all chunks belonging to one legal section directly
    from Chroma.

    This replaces reparsing all XML files at startup.
    """

    result = vectorstore.get(
        where={
            "$and": [
                {"act": act},
                {"section": section},
            ]
        },
        include=[
            "documents",
            "metadatas",
        ],
    )

    documents = []

    contents = result.get("documents", [])
    metadatas = result.get("metadatas", [])

    for content, metadata in zip(contents, metadatas):
        documents.append(
            Document(
                page_content=content,
                metadata=metadata,
            )
        )

    return documents


def select_related_chunks(
    query,
    selected_document,
    vectorstore,
    max_related=MAX_RELATED,
):
    """
    Retrieve related chunks from the same legal section.

    The legal structure and scoring method remain the same.
    """

    selected_metadata = selected_document.metadata

    act = selected_metadata.get("act", "")
    section = selected_metadata.get("section", "")

    section_documents = get_section_documents(
        vectorstore,
        act,
        section,
    )

    # --------------------------------------------------------
    # If the user explicitly asks for a subsection,
    # prefer chunks from that subsection.
    # --------------------------------------------------------

    requested_subsection = extract_subsection_reference(query)

    if requested_subsection:

        matching_documents = [
            document
            for document in section_documents
            if str(
                document.metadata.get("subsection", "")
            ).lower()
            == requested_subsection.lower()
        ]

        if matching_documents:
            section_documents = matching_documents

    # --------------------------------------------------------
    # Score chunks within the selected legal section.
    # --------------------------------------------------------

    scored_chunks = []

    for document in section_documents:

        text_score = text_match_score(
            query,
            document.page_content,
        )

        heading_score = text_match_score(
            query,
            document.metadata.get("heading", ""),
        )

        subsection_score = text_match_score(
            query,
            document.metadata.get("subsection", ""),
        )

        chunk_score = (
            0.70 * text_score
            + 0.20 * heading_score
            + 0.10 * subsection_score
        )

        scored_chunks.append(
            (document, chunk_score)
        )

    # --------------------------------------------------------
    # Select strongest related chunks.
    # --------------------------------------------------------

    scored_chunks.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    selected_chunks = [
        document
        for document, score
        in scored_chunks[:max_related]
    ]

    # --------------------------------------------------------
    # Restore legal structure order.
    # --------------------------------------------------------

    selected_chunks.sort(
        key=structure_order
    )

    return selected_chunks


# ============================================================
# STRUCTURE-AWARE RETRIEVER
# ============================================================

class StructureAwareRetriever:

    def __init__(
        self,
        candidate_k=CANDIDATE_K,
    ):

        self.candidate_k = candidate_k

        print("Loading structure-aware legal documents...")

        # Load the existing Chroma vectorstore.
        self.vectorstore = load_vectorstore()

        print("Structure-aware vectorstore loaded.")

    # --------------------------------------------------------
    # Initial semantic retrieval
    # --------------------------------------------------------

    def retrieve_candidates(self, query):

        return self.vectorstore.similarity_search_with_score(
            query,
            k=self.candidate_k,
        )

    # --------------------------------------------------------
    # Reranking
    # --------------------------------------------------------

    def rerank(self, query, candidates):

        scored_documents = []

        for document, distance in candidates:

            # Convert Chroma distance into a similarity-like score.
            semantic_score = 1 / (1 + distance)

            structure_score = calculate_structure_score(
                query,
                document,
            )

            final_score = (
                0.60 * semantic_score
                + 0.40 * structure_score
            )

            scored_documents.append(
                (
                    document,
                    final_score,
                )
            )

        scored_documents.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return scored_documents

    # --------------------------------------------------------
    # Main retrieval function
    # --------------------------------------------------------

    def retrieve(
        self,
        query,
        expanded_query=None,
        top_k=DEFAULT_TOP_K,
        max_related=MAX_RELATED,
    ):

        retrieval_query = expanded_query or query

        # Step 1: retrieve semantic candidates.
        candidates = self.retrieve_candidates(
            retrieval_query
        )

        # Step 2: rerank candidates using structure.
        ranked_documents = self.rerank(
            query,
            candidates,
        )

        # Step 3: select strongest legal sections.
        selected_sections = select_sections(
            ranked_documents,
            top_k,
        )

        # Step 4: retrieve related chunks from
        # the selected legal sections.
        final_documents = []

        for document in selected_sections:

            related_chunks = select_related_chunks(
                query,
                document,
                self.vectorstore,
                max_related,
            )

            final_documents.extend(
                related_chunks
            )

        return final_documents

    # --------------------------------------------------------
    # Format results for the agent
    # --------------------------------------------------------

    def format_results(self, documents):

        if not documents:
            return "No relevant legal provisions were found."

        formatted = []

        for document in documents:

            metadata = document.metadata

            act = metadata.get("act", "")
            section = metadata.get("section", "")
            heading = metadata.get("heading", "")

            formatted.append(
                f"Act: {act}\n"
                f"Section: {section}\n"
                f"Heading: {heading}\n\n"
                f"{document.page_content}"
            )

        return "\n\n---\n\n".join(formatted)


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    retriever = StructureAwareRetriever()

    question = (
        "Can my landlord increase my rent?"
    )

    results = retriever.retrieve(
        question,
        top_k=5,
        max_related=10,
    )

    print("\nRetrieved documents:")
    print("--------------------")

    for i, document in enumerate(results, 1):

        print(
            f"\nResult {i}: "
            f"{document.metadata.get('act')} - "
            f"Section "
            f"{document.metadata.get('section')}"
        )

    print("\nFormatted results:")
    print("------------------")

    print(
        retriever.format_results(results)
    )
