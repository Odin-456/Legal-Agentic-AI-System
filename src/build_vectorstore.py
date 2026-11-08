from pathlib import Path
import shutil
import sys

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


# Project paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# Project modules

from property_law_parser import parse_property_law_corpus
from property_law_chunker import create_chunks
from structure_aware_chunker import (
    parse_structure_corpus,
    remove_duplicates,
)


# Configuration

BASELINE_VECTORSTORE = PROJECT_ROOT/"data"/"vectorstore"/"property_law_baseline"

STRUCTURE_VECTORSTORE = PROJECT_ROOT/"data"/"vectorstore"/"property_law_structure"

EMBEDDING_MODEL = "text-embedding-3-large"


load_dotenv()


def reset_vectorstore(path):
    """
    Remove an existing vectorstore before rebuilding it.
    """

    if path.exists():

        print()
        print(f"Removing vectorstore: {path}")

        shutil.rmtree(path)

        print("Vectorstore removed.")


def build_baseline_documents():

    sections = parse_property_law_corpus()

    documents = create_chunks(sections)

    return documents


def build_structure_documents():

    documents = parse_structure_corpus()

    documents = remove_duplicates(documents)

    return documents


def main():

    print()
    print("=" * 60)
    print("PROPERTY LAW VECTORSTORE BUILD")
    print("=" * 60)
    print()

    print(f"Embedding model: {EMBEDDING_MODEL}")

    # Create embedding model

    print()
    print("Creating embedding model...")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    # Remove old vectorstores

    reset_vectorstore(
        BASELINE_VECTORSTORE
    )

    reset_vectorstore(
        STRUCTURE_VECTORSTORE
    )

    # Build baseline documents

    print()
    print("Building baseline documents...")

    baseline_documents = build_baseline_documents()

    print(
        f"Baseline documents: "
        f"{len(baseline_documents)}"
    )

    # Build structure-aware documents

    print()
    print("Building structure-aware documents...")

    structure_documents = build_structure_documents()

    print(
        f"Structure-aware documents: "
        f"{len(structure_documents)}"
    )

    # Create baseline vectorstore

    print()
    print("Creating baseline vectorstore...")

    Chroma.from_documents(
        documents=baseline_documents,
        embedding=embeddings,
        collection_name="property_law_baseline",
        persist_directory=str(
            BASELINE_VECTORSTORE
        ),
    )

    print("Baseline vectorstore created.")

    # Create structure-aware vectorstore

    print()
    print("Creating structure-aware vectorstore...")

    Chroma.from_documents(
        documents=structure_documents,
        embedding=embeddings,
        collection_name="property_law_structure",
        persist_directory=str(
            STRUCTURE_VECTORSTORE
        ),
    )

    print("Structure-aware vectorstore created.")

    print()
    print("=" * 60)
    print("VECTORSTORE BUILD COMPLETE")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()

