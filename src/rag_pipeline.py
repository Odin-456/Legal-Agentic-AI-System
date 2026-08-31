from query_expansion import expand_query
from structure_retrieval import StructureAwareRetriever


class RAGPipeline:

    def __init__(self):
        self.retriever = StructureAwareRetriever()

    def retrieve(self, question, top_k=5):

        # 1. Expand the user's question
        expanded_query = expand_query(question)

        # 2. Retrieve relevant legal documents
        results = self.retriever.retrieve(
            question,
            expanded_query=expanded_query,
            top_k=top_k,
            max_related=10
        )

        return results


if __name__ == "__main__":

    pipeline = RAGPipeline()

    question = (
        "What must a foreign person do when land "
        "subsequently becomes agricultural land?"
    )

    results = pipeline.retrieve(question)

    print("=" * 80)
    print("RAG PIPELINE")
    print("=" * 80)

    for i, result in enumerate(results, start=1):

        metadata = result.get("metadata", {})

        print()
        print(f"Result {i}")
        print(
            f"Citation: "
            f"{metadata.get('citation', 'Unknown')}"
        )

        print(
            f"Paragraph: "
            f"{metadata.get('paragraph', '')}"
        )

        print(
            f"Score: "
            f"{result.get('score', 0):.4f}"
        )

        print(
            f"Type: "
            f"{result.get('retrieval_type', 'Unknown')}"
        )

        print(
            f"Text: "
            f"{result.get('text', '')[:500]}"
        )