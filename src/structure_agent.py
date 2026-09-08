import os

from dotenv import load_dotenv
from structure_retrieval import StructureAwareRetriever
from query_expansion import expand_query
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

print(
    "Initialising Structure-Aware NSW Property-Law Agent..."
)

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

    expanded_query = (
        expand_query(query)
    )

    results = retriever.retrieve(
        query,
        expanded_query=expanded_query,
        top_k=TOP_K,
        max_related=MAX_RELATED,
    )

    if not results:

        return (
            "No relevant NSW property-law documents "
            "were found.",
            [],
        )

    context_parts = []

    for i, result in enumerate(
        results,
        start=1,
    ):

        metadata = result.get(
            "metadata",
            {},
        )

        act = metadata.get(
            "act",
            "Unknown Act",
        )

        section = metadata.get(
            "section",
            "Unknown section",
        )

        heading = metadata.get(
            "heading",
            "",
        )

        retrieval_type = result.get(
            "retrieval_type",
            "unknown",
        )

        context_parts.append(
            f"""
SOURCE {i}

ACT:
{act}

SECTION:
{section}

HEADING:
{heading}

RETRIEVAL TYPE:
{retrieval_type}

LEGAL TEXT:
{result.get("text", "")}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are an Australian property-law question-answering
assistant.

You answer questions using ONLY the NSW property-law
legislation provided in the retrieved context.

RULES:

1. Do not use information outside the retrieved context.

2. Do not invent legal rules.

3. Do not invent sections or Act names.

4. Answer the customer's question directly.

5. Explain the answer in clear ordinary language.

6. Preserve the meaning of the legislation.

7. If the retrieved documents do not contain enough
   information, say so instead of guessing.

8. Always identify the relevant Act and section when
   supported by the retrieved documents.

9. Do not provide personalised legal advice.

10. Make clear that the response is general information
    based on the retrieved NSW legislation.

CUSTOMER QUESTION:

{query}

RETRIEVED NSW PROPERTY-LAW DOCUMENTS:

{context}

ANSWER:
"""

    response = llm.invoke(
        prompt
    )

    return (
        response.content,
        results,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    query = (
        "What happens if a landlord fails to carry out "
        "required repairs?"
    )

    answer, results = (
        answer_question(
            query
        )
    )

    print()
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(answer)

    print()
    print("=" * 80)
    print("SOURCES")
    print("=" * 80)

    for index, result in enumerate(
        results,
        start=1,
    ):

        metadata = result.get(
            "metadata",
            {},
        )

        print(
            f"{index}. "
            f"{metadata.get('act', 'Unknown')} "
            f"§ {metadata.get('section', 'Unknown')} "
            f"- "
            f"{metadata.get('heading', '')}"
        )