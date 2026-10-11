from structure_retrieval import StructureAwareRetriever


QUESTIONS = [
    "What responsibilities does a landlord have for repairs?",
    "Can a tenant end a rental agreement early?",
    "What are the responsibilities of a strata owners corporation?",
    "What happens when someone wants to transfer ownership of land?",
    "What does a community association have to do?",
]


def main():

    print("=" * 80)
    print("GRAPH RETRIEVAL TEST")
    print("=" * 80)

    retriever = StructureAwareRetriever(
        candidate_k=50,
        use_query_expansion=False,
        use_bm25=False,
        use_graph=True,
    )

    for question in QUESTIONS:

        print("\n")
        print("=" * 80)
        print("QUESTION:")
        print(question)
        print("=" * 80)

        results = retriever.retrieve(
            question,
            top_k=5
        )

        print("\nTop 5 results:")

        for index, result in enumerate(
            results,
            start=1
        ):

            document = result["document"]
            metadata = document.metadata

            print(
                f"{index}. "
                f"{metadata.get('act', '')} | "
                f"Section "
                f"{metadata.get('section', '')} | "
                f"{metadata.get('heading', '')}"
            )


if __name__ == "__main__":
    main()