from pathlib import Path
import shutil
import sys

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from structure_aware_chunker import (
    parse_structure_corpus,
    remove_duplicates,
)


# ============================================================
# CONFIGURATION
# ============================================================

VECTORSTORE_DIR = (
    PROJECT_ROOT
    / "data"
    / "vectorstore"
    / "property_law_structure"
)

COLLECTION_NAME = "property_law_structure"

EMBEDDING_MODEL = "text-embedding-3-large"


# ============================================================
# MAIN
# ============================================================

def main():

    load_dotenv(PROJECT_ROOT / ".env")

    print("=" * 70)
    print("REBUILDING STRUCTURE-AWARE VECTORSTORE")
    print("=" * 70)

    # --------------------------------------------------------
    # Remove existing vectorstore
    # --------------------------------------------------------

    if VECTORSTORE_DIR.exists():

        print()
        print("Removing existing structure vectorstore...")

        shutil.rmtree(VECTORSTORE_DIR)

        print("Existing vectorstore removed.")

    else:

        print()
        print("No existing structure vectorstore found.")

    # --------------------------------------------------------
    # Parse current structure-aware corpus
    # --------------------------------------------------------

    print()
    print("Parsing current structure-aware corpus...")

    documents = parse_structure_corpus()

    print()
    print(
        f"Documents loaded: {len(documents)}"
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    documents = remove_duplicates(documents)

    print(
        f"Documents after removing duplicates: "
        f"{len(documents)}"
    )

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    print()
    print("Creating embeddings...")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    # --------------------------------------------------------
    # Build Chroma vectorstore
    # --------------------------------------------------------

    print()
    print("Building Chroma vectorstore...")

    VECTORSTORE_DIR.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
    )

    print()
    print("=" * 70)
    print("VECTORSTORE REBUILD COMPLETE")
    print("=" * 70)

    print()
    print(f"Documents: {len(documents)}")
    print(f"Location: {VECTORSTORE_DIR}")
    print(f"Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()