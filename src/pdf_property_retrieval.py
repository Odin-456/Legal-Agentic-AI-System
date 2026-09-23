import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pdf_property_law"
    / "C2024C00719_chunks.txt"
)

VECTORSTORE_DIR = (
    PROJECT_ROOT
    / "data"
    / "vectorstore"
    / "property_law_pdf"
)


COLLECTION_NAME = "property_law_pdf"


class PDFPropertyRetriever:

    def __init__(self):

        print(
            "Initialising PDF property-law retriever..."
        )

        if not CHUNKS_FILE.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {CHUNKS_FILE}"
            )

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv(
                "OPENAI_API_KEY"
            ),
        )

        self.vectorstore = None

        self._build_vectorstore()

    def _load_chunks(self):

        with open(
            CHUNKS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

        raw_chunks = text.split(
            "=" * 80
        )

        documents = []

        for raw_chunk in raw_chunks:

            raw_chunk = raw_chunk.strip()

            if not raw_chunk:
                continue

            if raw_chunk.startswith("CHUNK"):

                lines = raw_chunk.split(
                    "\n",
                    1
                )

                if len(lines) == 2:

                    chunk_number = (
                        lines[0]
                        .replace("CHUNK", "")
                        .strip()
                    )

                    chunk_text = lines[1].strip()

                    documents.append(
                        Document(
                            page_content=chunk_text,
                            metadata={
                                "source": "C2024C00719.pdf",
                                "document_type": "PDF",
                                "chunk": chunk_number,
                            },
                        )
                    )

        return documents

    def _build_vectorstore(self):

        documents = self._load_chunks()

        print(
            f"PDF chunks loaded: {len(documents)}"
        )

        if not documents:
            raise ValueError(
                "No PDF chunks were found."
            )

        VECTORSTORE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(
                VECTORSTORE_DIR
            ),
        )

        print(
            "PDF vector database created."
        )

    def retrieve(
        self,
        query,
        top_k=5
    ):

        results = (
            self.vectorstore.similarity_search(
                query,
                k=top_k
            )
        )

        return results


if __name__ == "__main__":

    retriever = PDFPropertyRetriever()

    question = input(
        "\nEnter a question: "
    ).strip()

    if question:

        results = retriever.retrieve(
            question,
            top_k=5
        )

        print()
        print("=" * 80)
        print("RETRIEVED PDF RESULTS")
        print("=" * 80)

        for i, result in enumerate(
            results,
            start=1
        ):

            print()
            print(
                f"RESULT {i}"
            )
            print("-" * 80)
            print(
                result.page_content[:2000]
            )