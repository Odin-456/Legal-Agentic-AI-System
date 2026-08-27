import os

from dotenv import load_dotenv
from structure_retrieval import StructureAwareRetriever
from langchain_openai import ChatOpenAI


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

MODEL_NAME = "gpt-5-mini"
TOP_K = 5
MAX_RELATED = 10


# ============================================================
# INITIALISE
# ============================================================

print("Initialising Structure-Aware Legal Agent...")

retriever = StructureAwareRetriever()

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)


# ============================================================
# ANSWER QUESTION
# ============================================================

def answer_question(query):

    results = retriever.retrieve(
        query,
        top_k=TOP_K,
        max_related=MAX_RELATED,
    )

    if not results:
        return (
            "No relevant legal documents were found.",
            []
        )

    context_parts = []

    for result in results:

        metadata = result["metadata"]

        citation = metadata.get(
            "citation",
            "Unknown citation"
        )

        paragraph = metadata.get(
            "paragraph"
        )

        subparagraph = metadata.get(
            "subparagraph"
        )

        text = result["text"]

        location = citation

        if paragraph:
            location += paragraph

        if subparagraph:
            location += subparagraph

        context_parts.append(
            f"[{location}]\n{text}"
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are a legal research assistant.

Answer the user's question using ONLY the
provided eCFR context.

Rules:

1. Do not invent legal rules.
2. If the context does not contain enough
   information, say so clearly.
3. Use the most directly relevant regulation.
4. Cite the relevant CFR provision(s).
5. Do not treat unrelated USDA programs as
   answers to the question.
6. If multiple regulations are relevant,
   clearly distinguish them.
7. Give a concise, direct answer.

USER QUESTION:
{query}

eCFR CONTEXT:
{context}
"""

    response = llm.invoke(prompt)

    return response.content, results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    query = (
        "What happens when land becomes "
        "agricultural land after being acquired?"
    )

    print("\nQUERY:")
    print(query)

    answer, results = answer_question(
        query
    )

    print("\nSTRUCTURE-AWARE AGENT ANSWER:")
    print(answer)

    print("\nRETRIEVED SOURCES:")

    for result in results:

        metadata = result["metadata"]

        citation = metadata.get(
            "citation",
            "Unknown"
        )

        paragraph = metadata.get(
            "paragraph"
        )

        subparagraph = metadata.get(
            "subparagraph"
        )

        retrieval_type = result.get(
            "retrieval_type",
            "unknown"
        )

        location = citation

        if paragraph:
            location += paragraph

        if subparagraph:
            location += subparagraph

        print(
            f"- {location} "
            f"[{retrieval_type}]"
        )