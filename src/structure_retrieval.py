import re
from pathlib import Path
from difflib import SequenceMatcher
from dotenv import load_dotenv

load_dotenv()

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from structure_aware_chunker import parse_structure_corpus


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VECTORSTORE_DIR = (
    BASE_DIR
    / "data"
    / "vectorstore"
    / "property_law_structure"
)

COLLECTION_NAME = "property_law_structure"

EMBEDDING_MODEL = "text-embedding-3-small"

DEFAULT_TOP_K = 5
CANDIDATE_K = 50
MAX_RELATED = 10


# ============================================================
# STOP WORDS
# ============================================================

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "can", "could", "does", "for", "from", "how",
    "if", "in", "is", "it", "of", "on", "or",
    "that", "the", "this", "to", "what", "when",
    "where", "which", "who", "why", "with",
}

LEGAL_TERM_GROUPS = {
    "landlord": {
        "landlord",
        "lessor",
        "owner",
    },

    "tenant": {
        "tenant",
        "lessee",
        "renter",
    },

    "property": {
        "property",
        "land",
        "premises",
        "dwelling",
    },

    "owner": {
        "owner",
        "proprietor",
        "registered",
        "ownership",
    },

    "repair": {
        "repair",
        "repairs",
        "repairing",
        "maintenance",
        "maintain",
        "maintaining",
    },

    "sell": {
        "sell",
        "sale",
        "selling",
        "transfer",
        "transferee",
        "convey",
        "conveyance",
    },

    "buy": {
        "buy",
        "purchase",
        "purchaser",
        "acquire",
        "acquisition",
    },

    "lease": {
        "lease",
        "leasing",
        "rent",
        "rental",
        "tenancy",
    },

    "mortgage": {
        "mortgage",
        "mortgagee",
        "mortgagor",
        "mortgages",
    },

    "caveat": {
        "caveat",
        "caveator",
        "caveats",
    },

    "easement": {
        "easement",
        "easements",
        "right",
        "rights",
    },

    "notice": {
        "notice",
        "notify",
        "notified",
        "notification",
        "inform",
        "information",
    },

    "responsibility": {
        "responsibility",
        "responsible",
        "obligation",
        "obligations",
        "duty",
        "duties",
        "required",
        "requirement",
        "requirements",
    },

    "permission": {
        "permission",
        "consent",
        "approval",
        "authorisation",
        "authorization",
        "allow",
        "allowed",
        "prohibited",
    },

    "function": {
        "function",
        "functions",
        "power",
        "powers",
        "authority",
        "exercise",
        "exercising",
        "delegate",
        "delegation",
    },

    "application": {
        "application",
        "apply",
        "applicant",
        "applying",
        "request",
    },

    "registration": {
        "registration",
        "register",
        "registered",
        "registration",
        "lodgment",
        "lodged",
        "record",
    },
}

# ============================================================
# QUERY HELPERS
# ============================================================

def get_query_terms(query):
    """
    Get meaningful terms from a legal question.
    """

    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        query.lower()
    )

    return {
        word
        for word in words
        if word not in STOP_WORDS
        and len(word) > 1
    }

def correct_query_terms(query_terms):
    """
    Find likely misspellings in the query by comparing
    query terms with legal terms used in the corpus.
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
    Calculate text similarity while allowing
    common legal terms to be treated as related.
    """

    if not query_terms:
        return 0.0

    query_terms = correct_query_terms(
    query_terms
)

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
    """
    Detect an explicit section reference such as:

        section 42
        s 42
        s.42
        section 111
    """

    match = re.search(
        r"\b(?:section|sec\.?|s\.?)\s*(\d+[A-Za-z]*)\b",
        query.lower()
    )

    if match:
        return match.group(1)

    return ""


def extract_subsection_reference(query):
    """
    Detect an explicit subsection reference such as:

        subsection (1)
        subsection (2)
        subsection (1)(a)

    Only the first subsection number is returned.
    """

    match = re.search(
        r"subsection\s*(\([0-9A-Za-z]+\))",
        query.lower()
    )

    if match:
        return match.group(1)

    return ""

def extract_act_reference(query):
    """
    Detect whether the question explicitly mentions
    one of the Acts in the property-law corpus.
    """

    query_lower = query.lower()

    act_names = {
        "real property act 1900":
            "Real Property Act 1900",

        "conveyancing act 1919":
            "Conveyancing Act 1919",

        "residential tenancies act 2010":
            "Residential Tenancies Act 2010",

        "strata schemes management act 2015":
            "Strata Schemes Management Act 2015",

        "community land management act 2021":
            "Community Land Management Act 2021",
    }

    for name, act in act_names.items():

        if name in query_lower:
            return act

    return ""


# ============================================================
# STRUCTURE SCORE
# ============================================================

def calculate_structure_score(document, query):
    """
    Calculate how relevant a legal document is to
    the user's question using its text and structure.
    """

    metadata = document.metadata

    query_terms = get_query_terms(query)

    requested_act= extract_act_reference(
        query
    )

    document_act = str(
        metadata.get(
            "act",
            ""
        )
    )
    act_score = 0.0

    if requested_act:
        if (
            document_act.lower()
            == requested_act.lower()
        ):
            act_score = 1.0

    if not query_terms:
        return 0.0

    heading = metadata.get(
        "heading",
        ""
    )

    subsection = metadata.get(
        "subsection",
        ""
    )

    paragraph = metadata.get(
        "paragraph",
        ""
    )

    nested_paragraph = metadata.get(
        "nested_paragraph",
        ""
    )

    page_content = document.page_content

    # --------------------------------------------------------
    # Match the question against the legal text
    # --------------------------------------------------------

    text_score = text_match_score(
        query_terms,
        page_content
    )

    # --------------------------------------------------------
    # Match the question against the section heading
    # --------------------------------------------------------

    heading_score = text_match_score(
        query_terms,
        heading
    )

    # --------------------------------------------------------
    # Match against the structural information
    # --------------------------------------------------------

    paragraph_score = text_match_score(
        query_terms,
        paragraph
    )

    nested_score = text_match_score(
        query_terms,
        nested_paragraph
    )

    # --------------------------------------------------------
    # Explicit section reference
    # --------------------------------------------------------

    section_score = 0.0

    section_reference = extract_section_reference(
        query
    )

    if section_reference:

        document_section = str(
            metadata.get(
                "section",
                ""
            )
        )

        if (
            document_section.lower()
            == section_reference.lower()
        ):
            section_score = 1.0

    # --------------------------------------------------------
    # Explicit subsection reference
    # --------------------------------------------------------

    subsection_score = 0.0

    subsection_reference = (
        extract_subsection_reference(query)
    )

    if subsection_reference:

        document_subsection = str(
            subsection
        )

        if (
            document_subsection.lower()
            == subsection_reference.lower()
        ):
            subsection_score = 1.0

    # --------------------------------------------------------
    # Combine the structure signals
    # --------------------------------------------------------

    score = (
        0.50 * text_score
        + 0.22 * heading_score
        + 0.10 * act_score
        + 0.08 * section_score
        + 0.05 * subsection_score
        + 0.03 * paragraph_score
        + 0.02 * nested_score
    )

    return min(
        score,
        1.0
    )


# ============================================================
# VECTORSTORE
# ============================================================

def load_vectorstore():
    """
    Load the structure-aware Chroma vector database.
    """

    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            f"Vectorstore not found: {VECTORSTORE_DIR}"
        )

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(
            VECTORSTORE_DIR
        ),
        embedding_function=embeddings,
    )


# ============================================================
# SECTION IDENTIFIER
# ============================================================

def get_section_key(document):
    """
    Identify a document using its Act and section.
    """

    metadata = document.metadata

    return (
        metadata.get("act", ""),
        metadata.get("section", ""),
    )


# ============================================================
# STRUCTURE ORDER
# ============================================================

def structure_order(document):
    """
    Return the legal structure used to restore
    the original order of the selected chunks.
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
    Select the strongest document from each Act/section.

    This prevents several chunks from the same section
    from taking all of the top results.
    """

    sections = {}

    for score, document in scored_documents:

        key = get_section_key(document)

        if key not in sections:
            sections[key] = (
                score,
                document
            )

        elif score > sections[key][0]:
            sections[key] = (
                score,
                document
            )

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
    Select relevant chunks within the selected section.

    If the question contains an explicit subsection,
    chunks from that subsection are prioritised.
    Otherwise, relevant chunks from the whole section
    are considered.
    """

    selected_metadata = selected_document.metadata

    act = selected_metadata.get(
        "act",
        ""
    )

    section = selected_metadata.get(
        "section",
        ""
    )

    # --------------------------------------------------------
    # Get all chunks from the selected section
    # --------------------------------------------------------

    section_documents = []

    for document in all_documents:

        metadata = document.metadata

        if (
            metadata.get("act") == act
            and
            metadata.get("section") == section
        ):
            section_documents.append(
                document
            )

    if not section_documents:
        return [selected_document]

    # --------------------------------------------------------
    # Check whether the question specifies a subsection
    # --------------------------------------------------------

    subsection_reference = (
        extract_subsection_reference(query)
    )

    if subsection_reference:

        subsection_documents = []

        for document in section_documents:

            document_subsection = str(
                document.metadata.get(
                    "subsection",
                    ""
                )
            )

            if (
                document_subsection.lower()
                == subsection_reference.lower()
            ):
                subsection_documents.append(
                    document
                )

        # Only restrict the context if matching chunks
        # were actually found.
        if subsection_documents:
            section_documents = subsection_documents

    # --------------------------------------------------------
    # Score chunks
    # --------------------------------------------------------

    query_terms = get_query_terms(query)

    scored_chunks = []

    for document in section_documents:

        metadata = document.metadata

        text_score = text_match_score(
            query_terms,
            document.page_content
        )

        heading_score = text_match_score(
            query_terms,
            metadata.get("heading", "")
        )

        subsection_score = 0.0

        if subsection_reference:

            document_subsection = str(
                metadata.get(
                    "subsection",
                    ""
                )
            )

            if (
                document_subsection.lower()
                == subsection_reference.lower()
            ):
                subsection_score = 1.0

        chunk_score = (
            0.70 * text_score
            + 0.20 * heading_score
            + 0.10 * subsection_score
        )

        scored_chunks.append(
            (
                chunk_score,
                document
            )
        )

    # --------------------------------------------------------
    # Select strongest chunks
    # --------------------------------------------------------

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True
    )

    selected = [
        document
        for _, document
        in scored_chunks[:max_related]
    ]

    # --------------------------------------------------------
    # Restore legal document order
    # --------------------------------------------------------

    selected.sort(
        key=structure_order
    )

    return selected


# ============================================================
# RETRIEVER
# ============================================================

class StructureAwareRetriever:

    def __init__(
        self,
        candidate_k=CANDIDATE_K
    ):

        self.candidate_k = candidate_k

        self.vectorstore = load_vectorstore()

        print(
            "Loading structure-aware legal documents..."
        )

        self.all_documents = (
            parse_structure_corpus()
        )

        print(
            f"Loaded {len(self.all_documents)} "
            f"structure-aware chunks."
        )

    # --------------------------------------------------------
    # Candidate retrieval
    # --------------------------------------------------------

    def retrieve_candidates(self, query):

        return (
            self.vectorstore
            .similarity_search_with_score(
                query,
                k=self.candidate_k
            )
        )

    # --------------------------------------------------------
    # Reranking
    # --------------------------------------------------------

    def rerank(
        self,
        query,
        candidates
    ):
        """
        Combine semantic similarity with structural
        relevance.
        """

        scored = []

        for document, distance in candidates:

            semantic_score = (
                1.0
                /
                (
                    1.0
                    + float(distance)
                )
            )

            structure_score = (
                calculate_structure_score(
                    document,
                    query
                )
            )

            final_score = (
                0.60 * semantic_score
                + 0.40 * structure_score
            )

            scored.append(
                (
                    final_score,
                    document
                )
            )

        scored.sort(
            key=lambda item: item[0],
            reverse=True
        )

        return scored

    # --------------------------------------------------------
    # Main retrieval function
    # --------------------------------------------------------

    def retrieve(
        self,
        query,
        top_k=DEFAULT_TOP_K,
        max_related=MAX_RELATED
    ):
        """
        Retrieve relevant sections and their related
        legal chunks.
        """

        candidates = (
            self.retrieve_candidates(
                query
            )
        )

        reranked = self.rerank(
            query,
            candidates
        )

        selected_sections = select_sections(
            reranked,
            top_k
        )

        results = []

        for section_score, document in (
            selected_sections
        ):

            related_documents = (
                select_related_chunks(
                    query=query,
                    selected_document=document,
                    all_documents=self.all_documents,
                    max_related=max_related
                )
            )

            results.append(
                {
                    "document": document,
                    "score": section_score,
                    "metadata": document.metadata,
                    "related_documents":
                        related_documents,
                }
            )

        return results


# ============================================================
# FORMAT RESULTS
# ============================================================

def format_results(results):
    """
    Format retrieved legislation for the legal agent.
    """

    output = []

    for index, result in enumerate(
        results,
        start=1
    ):

        metadata = result["metadata"]

        act = metadata.get(
            "act",
            ""
        )

        section = metadata.get(
            "section",
            ""
        )

        heading = metadata.get(
            "heading",
            ""
        )

        related_documents = (
            result["related_documents"]
        )

        text_parts = []

        for document in related_documents:
            text_parts.append(
                document.page_content
            )

        if not text_parts:
            text_parts.append(
                result["document"].page_content
            )

        output.append(
            f"""
SOURCE {index}

ACT AND SECTION:
{act}, section {section}

HEADING:
{heading}

LEGAL TEXT:
{"".join(text_parts)}
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

        print(
            "No question entered."
        )

    else:

        results = retriever.retrieve(
            question,
            top_k=5,
            max_related=10
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            "STRUCTURE-AWARE RETRIEVAL RESULTS"
        )

        print(
            "=" * 80
        )

        for index, result in enumerate(
            results,
            start=1
        ):

            metadata = result["metadata"]

            print(
                f"\n[{index}] "
                f"{metadata.get('act', '')} "
                f"- Section "
                f"{metadata.get('section', '')}"
            )

            print(
                f"Score: "
                f"{result['score']:.4f}"
            )

            print(
                f"Heading: "
                f"{metadata.get('heading', '')}"
            )

            print(
                "-" * 60
            )

            for document in (
                result["related_documents"]
            ):

                print(
                    document.page_content[:500]
                )

                print(
                    "-" * 40
                )