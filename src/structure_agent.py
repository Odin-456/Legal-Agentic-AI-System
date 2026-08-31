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

    # --------------------------------------------------------
    # Retrieve relevant legal passages
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Build structured context
    # --------------------------------------------------------

    context_parts = []

    for i, result in enumerate(results, start=1):

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

        clause = metadata.get(
            "clause"
        )

        subclause = metadata.get(
            "subclause"
        )

        text = result["text"]

        # Build the most specific legal location
        location = citation

        if paragraph:
            location += paragraph

        if subparagraph:
            location += subparagraph

        if clause:
            location += clause

        if subclause:
            location += subclause

        retrieval_type = result.get(
            "retrieval_type",
            "unknown"
        )

        context_parts.append(
            f"SOURCE {i}\n"
            f"CITATION: {location}\n"
            f"RETRIEVAL TYPE: {retrieval_type}\n"
            f"TEXT: {text}"
        )

    context = "\n\n".join(context_parts)

    # --------------------------------------------------------
    # Grounded legal-answer prompt
    # --------------------------------------------------------

    prompt = f"""
You are a legal research assistant specialising
in the United States Code of Federal Regulations
(eCFR).

Your task is to answer the user's question using
ONLY the eCFR sources provided below.

STRICT GROUNDING RULES:

1. Use ONLY information explicitly contained in
   the provided eCFR context.

2. Do NOT use information from your general
   knowledge.

3. Do NOT use information from websites, regulations,
   programs, or documents that are not included in
   the provided context.

4. Every CFR citation in your answer MUST appear
   in the provided context.

5. If the question refers to a specific CFR section
   or paragraph, prioritize the source containing
   that exact section or paragraph.

6. Prefer the most directly relevant source over
   a related but less relevant source.

7. Do NOT introduce another CFR section merely
   because it may be related to the topic.

8. Do NOT substitute a different regulation for
   the regulation contained in the retrieved context.

9. If multiple sources are relevant, combine them
   only when they directly help answer the question.

10. If the provided context does not contain enough
    information to answer the question, say:

    "The provided eCFR context does not contain
    enough information to answer this question."

11. Do not guess or fill in missing information.

12. Keep the answer concise and directly answer
    the user's question.

13. After explaining the answer, provide the relevant
    CFR citation.

IMPORTANT:

The retrieved sources are the complete available
eCFR evidence for this question.

You must not rely on outside knowledge even if you
know a potentially relevant regulation.

USER QUESTION:
{query}

PROVIDED eCFR CONTEXT:
{context}

ANSWER:
"""

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

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

    for i, result in enumerate(results, start=1):

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

        clause = metadata.get(
            "clause"
        )

        subclause = metadata.get(
            "subclause"
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

        if clause:
            location += clause

        if subclause:
            location += subclause

        print(
            f"{i}. {location} "
            f"[{retrieval_type}]"
        )
