import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from retrieval import LegalRetriever


load_dotenv()


class LegalRAG:

    def __init__(self, top_k=5):
        print("Initialising Legal RAG...")

        self.top_k = top_k

        # Existing validated baseline retriever
        self.retriever = LegalRetriever()

        # OpenAI LLM
        self.llm = ChatOpenAI(
            model="gpt-5-mini",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        print("Legal RAG ready.")

    def retrieve(self, query):

        results = self.retriever.retrieve(
            query,
            top_k=self.top_k
        )

        documents = []

        for result in results:

            documents.append(
                Document(
                    page_content=result["text"],
                    metadata={
                        "citation": result["citation"],
                        "paragraph": result["paragraph"],
                        "subparagraph": result["subparagraph"],
                        "clause": result["clause"],
                        "score": result["score"],
                    }
                )
            )

        return documents

    def answer(self, query):

        documents = self.retrieve(query)

        context = "\n\n".join(
            f"[{doc.metadata['citation']} "
            f"{doc.metadata['paragraph'] or ''}]\n"
            f"{doc.page_content}"
            for doc in documents
        )

        prompt = f"""
You are a legal information assistant.

Answer the question using ONLY the provided eCFR
passages.

Do not use outside information.

If the passages do not provide enough information,
say so clearly.

Include the relevant eCFR citation in your answer.

Retrieved eCFR passages:

{context}

Question:

{query}
"""

        response = self.llm.invoke(prompt)

        return response.content


if __name__ == "__main__":

    rag = LegalRAG()

    query = (
        "What happens when land becomes agricultural "
        "land after being acquired?"
    )

    print("\nQUERY:")
    print(query)

    print("\nANSWER:")

    answer = rag.answer(query)

    print(answer)