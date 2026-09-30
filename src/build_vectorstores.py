from pathlib import Path

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


# ============================================================
# CHROMA COLLECTIONS
# ============================================================

BASELINE_COLLECTION = "property_law_baseline"

STRUCTURE_COLLECTION = "property_law_structure"


# ============================================================
# EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL = "text-embedding-3-large"


# ============================================================
# BUILD BASELINE VECTORSTORE
# ============================================================

def build_baseline_vectorstore(embeddings):

    print()
    print("=" * 80)
    print("BUILDING BASELINE VECTORSTORE")
    print("=" * 80)

    sections = parse_property_law_corpus()

    print(f"Sections parsed: {len(sections)}")

    documents = create_chunks(sections)

    print(f"Baseline documents: {len(documents)}")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=BASELINE_COLLECTION,
        persist_directory=str(BASELINE_VECTORSTORE),
    )

    print("Baseline vectorstore created.")

    return vectorstore


# ============================================================
# BUILD STRUCTURE VECTORSTORE
# ============================================================

def build_structure_vectorstore(embeddings):

    print()
    print("=" * 80)
    print("BUILDING STRUCTURE VECTORSTORE")
    print("=" * 80)

    documents = parse_structure_corpus()

    print(f"Structure chunks created: {len(documents)}")

    documents = remove_duplicates(documents)

    print(f"Unique structure chunks: {len(documents)}")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=STRUCTURE_COLLECTION,
        persist_directory=str(STRUCTURE_VECTORSTORE),
    )

    print("Structure vectorstore created.")

    return vectorstore


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    build_baseline_vectorstore(embeddings)

    build_structure_vectorstore(embeddings)

    print()
    print("=" * 80)
    print("VECTORSTORE BUILD COMPLETE")
    print("=" * 80)