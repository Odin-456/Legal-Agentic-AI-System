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
    BASE_DIR
    / "data"
    / "vectorstore"
    / "property_law_structure"
)

COLLECTION_NAME = "property_law_structure"
EMBEDDING_MODEL = "text-embedding-3-large"

DEFAULT_TOP_K = 5
CANDIDATE_K = 50
MAX_RELATED = 10


# ============================================================
# STOP WORDS AND RELATED LEGAL TERMS
# ============================================================

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "can", "could", "does", "for", "from", "how", "if",
    "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "what", "when", "where", "which",
    "who", "why", "with",
}


LEGAL_TERM_GROUPS = {
    "landlord": {
        "landlord", "lessor", "owner",
    },

    "tenant": {
        "tenant", "lessee", "renter",
    },

    "property": {
        "property", "land", "premises", "dwelling",
    },

    "owner": {
        "owner", "proprietor", "registered", "ownership",
    },

    "repair": {
        "repair", "repairs", "repairing", "maintenance",
        "maintain", "maintaining",
    },

    "sell": {
        "sell", "sale", "selling", "transfer",
        "transferee", "convey", "conveyance",
    },

    "buy": {
        "buy", "purchase", "purchaser", "acquire",
        "acquisition",
    },

    "lease": {
        "lease", "leasing", "rent", "rental", "tenancy",
    },

    "mortgage": {
        "mortgage", "mortgagee", "mortgagor", "mortgages",
    },

    "caveat": {
        "caveat", "caveator", "caveats",
    },

    "easement": {
        "easement", "easements", "right", "rights",
    },

    "notice": {
        "notice", "notify", "notified", "notification",
        "inform", "information",
    },

    "responsibility": {
        "responsibility", "responsible", "obligation",
        "obligations", "duty", "duties", "required",
        "requirement", "requirements",
    },

    "permission": {
        "permission", "consent", "approval",
        "authorisation", "authorization", "allow",
        "allowed", "prohibited",
    },

    "function": {
        "function", "functions", "power", "powers",
        "authority", "exercise", "exercising",
        "delegate", "delegation",
    },

    "application": {
        "application", "apply", "applicant",
        "applying", "request",
    },

    "registration": {
        "registration", "register", "registered",
        "lodgment", "lodged", "record",
    },
}


# ============================================================
# QUERY HELPERS
# ============================================================

def get_query_terms(query):

    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        query.lower()
    )

    terms = set()

    for word in words:

        if word not in STOP_WORDS and len(word) > 1:
            terms.add(word)

    return terms


def correct_query_terms(query_terms):

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

    match = re.search(
        r"\b(?:section|sec\.?|s\.?)\s*(\d+[A-Za-z]*)\b",
        query.lower()
    )

    if match:
        return match.group(1)

    return ""


def extract_subsection_reference(query):

    match = re.search(
        r"subsection\s*(\([0-9A-Za-z]+\))",
        query.lower()
    )

    if match:
        return match.group(1)

    return ""


def extract_act_reference(query):

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
# STRUCTURE SCORING
# ============================================================

def calculate_structure_score(document, query):

    metadata = document.metadata

    query_terms = get_query_terms(query)

    requested_act = extract_act_reference(query)

    document_act = str(
        metadata.get("act", "")
    )

    act_score = 0.0

    if requested_act:

        if document_act.lower() == requested_act.lower():
            act_score = 1.0

    if not query_terms:
        return 0.0

    heading = metadata.get("heading", "")
    subsection = metadata.get("subsection", "")
    paragraph = metadata.get("paragraph", "")
    nested_paragraph = metadata.get(
        "nested_paragraph",
        ""
    )

    page_content = document.page_content

    text_score = text_match_score(
        query_terms,
        page_content
    )

    heading_score = text_match_score(
        query_terms,
        heading
    )

    paragraph_score = text_match_score(
        query_terms,
        paragraph
    )

    nested_score = text_match_score(
        query_terms,
        nested_paragraph
    )

    section_score = 0.0

    section_reference = extract_section_reference(query)

    if section_reference:

        document_section = str(
            metadata.get("section", "")
        )

        if document_section.lower() == section_reference.lower():
            section_score = 1.0

    subsection_score = 0.0

    subsection_reference = extract_subsection_reference(query)

    if subsection_reference:

        document_subsection = str(subsection)

        if (
            document_subsection.lower()
            == subsection_reference.lower()
        ):
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

    return min(score, 1.0)


# ============================================================
# VECTOR STORE
# ============================================================

def load_vectorstore():

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
# SECTION HELPERS
# ============================================================

def get_section_key(document):

    metadata = document.metadata

    return (
        metadata.get("act", ""),
        metadata.get("section", ""),
    )


def structure_order(document):

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

    sections = {}

    for score, document in scored_documents:

        key = get_section_key(document)

        if key not in sections:
            sections[key] = (score, document)

        elif score > sections[key][0]:
            sections[key] = (score, document)

    ranked = sorted(
        sections.values(),
        key=lambda item: item[0],
        reverse=True
    )

    return ranked[:top_k]


# ============================================================
# RELATED CHUNKS WITHIN A SECTION
# ============================================================

def select_related_chunks(
    query,
    selected_document,
    all_documents,
    max_related
):

    selected_metadata = selected_document.metadata

    act = selected_metadata.get("act", "")
    section = selected_metadata.get("section", "")

    section_documents = []

    for document in all_documents:

        metadata = document.metadata

        if (
            metadata.get("act") == act
            and metadata.get("section") == section
        ):
            section_documents.append(document)

    if not section_documents:
        return [selected_document]

    subsection_reference = extract_subsection_reference(query)

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
                subsection_documents.append(document)

        if subsection_documents:
            section_documents = subsection_documents

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
                metadata.get("subsection", "")
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
            (chunk_score, document)
        )

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True
    )

    selected = []

    for score, document in scored_chunks[:max_related]:
        selected.append(document)

    selected.sort(
        key=structure_order
    )

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

    def retrieve_candidates(self, query):

        return self.vectorstore.similarity_search_with_score(
            query,
            k=self.candidate_k
        )

    def rerank(self, query, candidates):

        scored = []

        for document, distance in candidates:

            semantic_score = (
                1.0 / (1.0 + float(distance))
            )

            structure_score = calculate_structure_score(
                document,
                query
            )

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

    def retrieve(
        self,
        query,
        top_k=DEFAULT_TOP_K,
        max_related=MAX_RELATED
    ):

        candidates = self.retrieve_candidates(
            query
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

        for section_score, document in selected_sections:

            related_documents = select_related_chunks(
                query=query,
                selected_document=document,
                all_documents=self.all_documents,
                max_related=max_related
            )

            results.append(
                {
                    "document": document,
                    "score": section_score,
                    "metadata": document.metadata,
                    "related_documents": related_documents,
                }
            )

        return results
