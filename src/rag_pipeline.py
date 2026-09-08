from query_expansion import expand_query
from structure_retrieval import StructureAwareRetriever


class RAGPipeline:

    def __init__(self):

        self.retriever = (
            StructureAwareRetriever()
        )

    def retrieve(
        self,
        question,
        top_k=5,
        max_related=10,
    ):
        """
        Expand the question and retrieve relevant
        NSW property-law sections.
        """

        expanded_query = (
            expand_query(question)
        )

        results = (
            self.retriever.retrieve(
                question,
                expanded_query=expanded_query,
                top_k=top_k,
                max_related=max_related,
            )
        )

        return results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    pipeline = RAGPipeline()

    question = (
        "What happens if a landlord fails to carry out "
        "required repairs?"
    )

    results = pipeline.retrieve(
        question
    )

    print("=" * 80)
    print("NSW PROPERTY LAW RAG PIPELINE")
    print("=" * 80)

    print()
    print(f"Question: {question}")

    for i, result in enumerate(
        results,
        start=1,
    ):

        metadata = result.get(
            "metadata",
            {},
        )

        print()
        print("-" * 80)
        print(f"Result {i}")
        print("-" * 80)

        print(
            f"Act: "
            f"{metadata.get('act', 'Unknown')}"
        )

        print(
            f"Section: "
            f"{metadata.get('section', 'Unknown')}"
        )

        print(
            f"Heading: "
            f"{metadata.get('heading', 'Unknown')}"
        )

        subsection = metadata.get(
            "subsection",
            "",
        )

        if subsection:
            print(
                f"Subsection: "
                f"{subsection}"
            )

        paragraph = metadata.get(
            "paragraph",
            "",
        )

        if paragraph:
            print(
                f"Paragraph: "
                f"{paragraph}"
            )

        nested_paragraph = metadata.get(
            "nested_paragraph",
            "",
        )

        if nested_paragraph:
            print(
                f"Nested paragraph: "
                f"{nested_paragraph}"
            )

        hierarchy = metadata.get(
            "hierarchy",
            "",
        )

        if hierarchy:
            print(
                f"Hierarchy: "
                f"{hierarchy}"
            )

        print(
            f"Score: "
            f"{result.get('score', 0):.4f}"
        )

        print(
            f"Type: "
            f"{result.get('retrieval_type', 'Unknown')}"
        )

        print()
        print("Text:")

        print(
            result.get(
                "text",
                "",
            )[:1000]
        )