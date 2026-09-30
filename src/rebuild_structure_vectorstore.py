from pathlib import Path
import shutil

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from structure_aware_chunker import (
    parse_structure_corpus,
    remove_duplicates,
)


# ============================================================
# PATHS AND SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTORSTORE_DIR = (
    PROJECT_ROOT
    / "data"
    / "vectorstore"
    / "property_law_structure"
)

COLLECTION_NAME = "property_law_structure"
EMBEDDING_MODEL = "text-embedding-3-large"


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    # Load the OpenAI API key from .env
    load_dotenv(PROJECT_ROOT / ".env")

    print("Rebuilding structure-aware vectorstore...")

    # --------------------------------------------------------
    # Remove the old vectorstore
    # --------------------------------------------------------

    if VECTORSTORE_DIR.exists():
        print("Removing existing vectorstore...")
        shutil.rmtree(VECTORSTORE_DIR)

    # --------------------------------------------------------
    # Parse the NSW property law XML files
    # --------------------------------------------------------

    print("Parsing property law corpus...")

    documents = parse_structure_corpus()

    print(f"Documents loaded: {len(documents)}")

    # --------------------------------------------------------
    # Remove duplicate documents
    # --------------------------------------------------------

    documents = remove_duplicates(documents)

    print(f"Documents after removing duplicates: {len(documents)}")

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    print("Creating embeddings...")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    # --------------------------------------------------------
    # Create the Chroma vectorstore
    # --------------------------------------------------------

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

    print("Vectorstore rebuild complete.")
    print(f"Documents: {len(documents)}")
    print(f"Location: {VECTORSTORE_DIR}")


# Run the rebuild when this file is executed.
if __name__ == "__main__":
    main()