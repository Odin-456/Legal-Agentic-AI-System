import re
from pathlib import Path
from difflib import SequenceMatcher

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from structure_aware_chunker import parse_structure_corpus


load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VECTORSTORE_DIR = (
    BASE_DIR / "data" / "vectorstore" / "property_law_structure"
)

COLLECTION_NAME = "property_law_structure"

EMBEDDING_MODEL = "text-embedding-3-large"

DEFAULT_TOP_K = 5
CANDIDATE_K = 50
MAX_RELATED = 10


# ============================================================
# STOP WORDS AND LEGAL TERMS
# ============================================================

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "can", "could", "does", "for", "from", "how",
    "if", "in", "is", "it", "of", "on", "or",
    "that", "the", "this", "to", "what", "when",
    "where", "which", "who", "why", "with",
}

# Groups of related legal terms used during text matching
LEGAL_TERM_GROUPS = {
    "landlord": {"landlord", "lessor", "owner"},
    "tenant": {"tenant", "lessee", "renter"},
    "property": {"property", "land", "premises", "dwelling"},
    "owner": {"owner", "proprietor", "registered", "ownership"},
    "repair": {
        "repair", "repairs", "repairing",
        "maintenance", "maintain", "maintaining"
    },
    "sell": {
        "sell", "sale", "selling", "transfer",
        "transferee", "convey", "conveyance"
    },
    "buy": {
        "buy", "purchase", "purchaser",
        "acquire", "acquisition"
    },
    "lease": {
        "lease", "leasing", "rent",
        "rental", "tenancy"
    },
    "mortgage": {
        "mortgage", "mortgagee",
        "mortgagor", "mortgages"
    },
    "caveat": {
        "caveat", "caveator", "caveats"
    },
    "easement": {
        "easement", "easements",
        "right", "rights"
    },
    "notice": {
        "notice", "notify", "notified",
        "notification", "inform", "information"
    },
    "responsibility": {
        "responsibility", "responsible",
        "obligation", "obligations",
        "duty", "duties", "required",
        "requirement", "requirements"
    },
    "permission": {
        "permission", "consent", "approval",
        "authorisation", "authorization",
        "allow", "allowed", "prohibited"
    },
    "function": {
        "function", "functions", "power", "powers",
        "authority", "exercise", "exercising",
        "delegate", "delegation"
    },
    "application": {
        "application", "apply", "applicant",
        "applying", "request"
    },
    "registration": {
        "registration", "register", "registered",
        "lodgment", "lodged", "record"
    },
}


# ============================================================
# QUERY HELPERS
# ============================================================

def get_query_terms(query):
    """Get meaningful terms from a legal question."""

    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        query.lower()
    )

    return {
        word for word in words
        if word not in STOP_WORDS and len(word) > 1
    }


def correct_query_terms(query_terms):
    """
    Find likely misspellings by comparing query terms
    with legal terms.
    """

    legal_words = set()

    for group in LEGAL_TERM_GROUPS.values():
        legal_words.update(group)

    corrected_terms = set(query_terms)

    for term in query_terms:
        if len(term) < 4:
            continue

        best_match = None
        best_score = 0.0

        for legal_word in legal_words:
            similarity = SequenceMatcher(
                None,
                term,
                legal_word
            ).ratio()

            if similarity > best_score:
                best_score = similarity
                best_match = legal_word

        if best_match and best_score >= 0.80:
            corrected_terms.add(best_match)

    return corrected_terms


def text_match_score(query_terms, text):
    """
    Calculate text matching while allowing related
    legal terms to be treated as similar.
    """

    if not query_terms:
        return 0.0

    query_terms = correct_query_terms(query_terms)

    words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            str(text).lower()
        )
    )

    matched_terms = 0

    for term in query_terms:
        if term in words:
            matched_terms += 1
            continue

        for group in LEGAL_TERM_GROUPS.values():
            if term in group:
                if words.intersection(group):
                    matched_terms += 1
                break

    return min(
        matched_terms / len(query_terms),
        1.0
    )


def extract_section_reference(query):
    """Find an explicit section reference such as 'section 42' or 's 42'."""

    match = re.search(
        r"\b(?:section|sec\.?|s\.?)\s*(\d+[A-Za-z]*)\b",
        query.lower()
    )

    return match.group(1) if match else ""


def extract_subsection_reference(query):
    """Find an explicit subsection such as 'subsection (1)'."""

    match = re.search(
        r"subsection\s*(\([0-9A-Za-z]+\))",
        query.lower()
    )

    return match.group(1) if match else ""


def extract_act_reference(query):
    """Find an Act explicitly mentioned in the question."""

    act_names = {
        "real property act 1900": "Real Property Act 1900",
        "conveyancing act 1919": "Conveyancing Act 1919",
        "residential tenancies act 2010": "Residential Tenancies Act 2010",
        "strata schemes management act 2015": "Strata Schemes Management Act 2015",
        "community land management act 2021": "Community Land Management Act 2021",
    }

    query_lower = query.lower()

    for name, act in act_names.items():
        if name in query_lower:
            return act

    return ""


# ============================================================
# STRUCTURE SCORE
# ============================================================

def calculate_structure_score(document, query):
    """
    Calculate relevance using the document text,
    heading, Act, section and structural metadata.
    """

    metadata = document.metadata
    query_terms = get_query_terms(query)

    if not query_terms:
        return 0.0

    # Match the Act if the user explicitly mentions one
    requested_act = extract_act_reference(query)

    document_act = str(metadata.get("act", ""))

    act_score = 0.0

    if requested_act and document_act.lower() == requested_act.lower():
        act_score = 1.0

    # Get structural information
    heading = metadata.get("heading", "")
    subsection = metadata.get("subsection", "")
    paragraph = metadata.get("paragraph", "")
    nested_paragraph = metadata.get("nested_paragraph", "")

    # Match the question against the legal text
    text_score = text_match_score(
        query_terms,
        document.page_content
    )

    # Match the question against the section heading
    heading_score = text_match_score(
        query_terms,
        heading
    )

    # Match structural information
    paragraph_score = text_match_score(
        query_terms,
        paragraph
    )

    nested_score = text_match_score(
        query_terms,
        nested_paragraph
    )

    # Check for an explicit section reference
    section_score = 0.0
    section_reference = extract_section_reference(query)

    if section_reference:
        document_section = str(
            metadata.get("section", "")
        )

        if document_section.lower() == section_reference.lower():
            section_score = 1.0

    # Check for an explicit subsection reference
    subsection_score = 0.0
    subsection_reference = extract_subsection_reference(query)

    if subsection_reference:
        if str(subsection).lower() == subsection_reference.lower():
            subsection_score = 1.0

    # Combine all structure signals
    score = (
        0.50 * text_score
        + 0.22 * heading_score
        + 0.10 * act_score
        + 0.08 * section_score
        + 0.05 * subsection_score
        + 0.03 * paragraph_score
        + 0.02 * nested_score
    )

    return min(score, 1.0)


# ============================================================
# VECTORSTORE
# ============================================================

def load_vectorstore():
    """Load the structure-aware Chroma vector database."""

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
# STRUCTURE HELPERS
# ============================================================

def get_section_key(document):
    """Identify a document using its Act and section."""

    metadata = document.metadata

    return (
        metadata.get("act", ""),
        metadata.get("section", "")
    )


def structure_order(document):
    """Restore the original legal order of chunks."""

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
    Select the strongest document from each Act/section.

    This prevents one section from taking all top results.
    """

    sections = {}

    for score, document in scored_documents:
        key = get_section_key(document)

        if key not in sections or score > sections[key][0]:
            sections[key] = (score, document)

    ranked = sorted(
        sections.values(),
        key=lambda item: item[0],
        reverse=True
    )

    return ranked[:top_k]


# ============================================================
# CHUNK SELECTION WITHIN SECTION
# ============================================================

def select_related_chunks(
    query,
    selected_document,
    all_documents,
    max_related
):
    """
    Select relevant chunks from the selected section.

    If the question specifies a subsection, only matching
    subsection chunks are considered when available.
    """

    metadata = selected_document.metadata

    act = metadata.get("act", "")
    section = metadata.get("section", "")

    # Find all chunks belonging to the selected section
    section_documents = [
        document
        for document in all_documents
        if document.metadata.get("act") == act
        and document.metadata.get("section") == section
    ]

    if not section_documents:
        return [selected_document]

    # Check for a specific subsection
    subsection_reference = extract_subsection_reference(query)

    if subsection_reference:
        subsection_documents = [
            document
            for document in section_documents
            if str(
                document.metadata.get("subsection", "")
            ).lower() == subsection_reference.lower()
        ]

        if subsection_documents:
            section_documents = subsection_documents

    # Score chunks within the selected section
    query_terms = get_query_terms(query)
    scored_chunks = []

    for document in section_documents:

        text_score = text_match_score(
            query_terms,
            document.page_content
        )

        heading_score = text_match_score(
            query_terms,
            document.metadata.get("heading", "")
        )

        subsection_score = 0.0

        if subsection_reference:
            if str(
                document.metadata.get("subsection", "")
            ).lower() == subsection_reference.lower():
                subsection_score = 1.0

        chunk_score = (
            0.70 * text_score
            + 0.20 * heading_score
            + 0.10 * subsection_score
        )

        scored_chunks.append(
            (chunk_score, document)
        )

    # Select the strongest chunks
    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True
    )

    selected = [
        document
        for _, document in scored_chunks[:max_related]
    ]

    # Restore the original legal order
    selected.sort(key=structure_order)

    return selected


# ============================================================
# STRUCTURE-AWARE RETRIEVER
# ============================================================

class StructureAwareRetriever:

    def __init__(self, candidate_k=CANDIDATE_K):

        self.candidate_k = candidate_k
        self.vectorstore = load_vectorstore()

        print(
            "Loading structure-aware legal documents..."
        )

        self.all_documents = parse_structure_corpus()

        print(
            f"Loaded {len(self.all_documents)} "
            f"structure-aware chunks."
        )

    # --------------------------------------------------------
    # Candidate retrieval
    # --------------------------------------------------------

    def retrieve_candidates(self, query):
        """Retrieve the initial semantic candidates."""

        return self.vectorstore.similarity_search_with_score(
            query,
            k=self.candidate_k
        )

    # --------------------------------------------------------
    # Reranking
    # --------------------------------------------------------

    def rerank(self, query, candidates):
        """
        Combine semantic similarity with structural relevance.
        """

        scored = []

        for document, distance in candidates:

            semantic_score = 1.0 / (1.0 + float(distance))

            structure_score = calculate_structure_score(
                document,
                query
            )

            # Combine semantic and structure scores
            final_score = (
                0.60 * semantic_score
                + 0.40 * structure_score
            )

            scored.append(
                (final_score, document)
            )

        scored.sort(
            key=lambda item: item[0],
            reverse=True
        )

        return scored

    # --------------------------------------------------------
    # Main retrieval
    # --------------------------------------------------------

    def retrieve(
        self,
        query,
        top_k=DEFAULT_TOP_K,
        max_related=MAX_RELATED
    ):
        """Retrieve relevant sections and related chunks."""

        # Step 1: Retrieve semantic candidates
        candidates = self.retrieve_candidates(query)

        # Step 2: Rerank using structure-aware scoring
        reranked = self.rerank(
            query,
            candidates
        )

        # Step 3: Select different legal sections
        selected_sections = select_sections(
            reranked,
            top_k
        )

        results = []

        # Step 4: Retrieve related chunks from each section
        for section_score, document in selected_sections:

            related_documents = select_related_chunks(
                query=query,
                selected_document=document,
                all_documents=self.all_documents,
                max_related=max_related
            )

            results.append({
                "document": document,
                "score": section_score,
                "metadata": document.metadata,
                "related_documents": related_documents,
            })

        return results


# ============================================================
# FORMAT RESULTS
# ============================================================

def format_results(results):
    """Format retrieved legislation for the legal agent."""

    output = []

    for index, result in enumerate(results, 1):

        metadata = result["metadata"]

        act = metadata.get("act", "")
        section = metadata.get("section", "")
        heading = metadata.get("heading", "")

        documents = result["related_documents"]

        if not documents:
            documents = [result["document"]]

        legal_text = "\n".join(
            document.page_content
            for document in documents
        )

        output.append(
            f"""
SOURCE {index}

ACT AND SECTION:
{act}, section {section}

HEADING:
{heading}

LEGAL TEXT:
{legal_text}
"""
        )

    return "\n".join(output)


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    retriever = StructureAwareRetriever()

    question = input(
        "\nEnter a legal question: "
    ).strip()

    if not question:
        print("No question entered.")

    else:
        results = retriever.retrieve(
            question,
            top_k=5,
            max_related=10
        )

        print(
            "\n" + "=" * 80
        )

        print(
            "STRUCTURE-AWARE RETRIEVAL RESULTS"
        )

        print(
            "=" * 80
        )

        for index, result in enumerate(results, 1):

            metadata = result["metadata"]

            print(
                f"\n[{index}] "
                f"{metadata.get('act', '')} "
                f"- Section "
                f"{metadata.get('section', '')}"
            )

            print(
                f"Score: {result['score']:.4f}"
            )

            print(
                f"Heading: "
                f"{metadata.get('heading', '')}"
            )

            print("-" * 60)

            for document in result["related_documents"]:

                print(
                    document.page_content[:500]
                )

                print("-" * 40)