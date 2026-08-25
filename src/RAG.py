from retrieval import LegalRetriever


class LegalRAG:
    """
    Retrieval-Augmented Generation layer.

    This class currently handles:
        1. Query retrieval
        2. Context construction

    The LLM generation component will be added after
    retrieval and context construction are verified.
    """

    def __init__(self, top_k=5):

        self.top_k = top_k

        print("Initialising Legal RAG...")

        self.retriever = LegalRetriever()

        print("Legal RAG ready.")


    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(self, query):

        return self.retriever.retrieve(
            query=query,
            top_k=self.top_k
        )


    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    def build_context(self, results):

        context_parts = []

        for result in results:

            citation = result.get(
                "citation",
                ""
            )

            paragraph = result.get(
                "paragraph"
            )

            text = result.get(
                "text",
                ""
            )

            location = citation

            if paragraph:
                location += paragraph

            context_parts.append(
                f"[{location}]\n{text}"
            )

        return "\n\n".join(
            context_parts
        )


    # ========================================================
    # RETRIEVE + CONTEXT
    # ========================================================

    def get_context(self, query):

        results = self.retrieve(query)

        context = self.build_context(
            results
        )

        return results, context


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    query = (
        "What happens when land becomes "
        "agricultural land after being acquired?"
    )

    rag = LegalRAG(
        top_k=5
    )

    results, context = rag.get_context(
        query
    )

    print()
    print("=" * 70)
    print("RAG RETRIEVAL TEST")
    print("=" * 70)

    print()
    print("QUERY:")
    print(query)

    print()
    print("RETRIEVED DOCUMENTS:")

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{rank}. "
            f"{result['citation']} "
            f"(score={result['score']:.4f})"
        )

    print()
    print("=" * 70)
    print("CONTEXT SENT TO LLM")
    print("=" * 70)

    print(context)
