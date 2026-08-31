from structure_retrieval import StructureAwareRetriever
from langchain_openai import ChatOpenAI


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "gpt-4o-mini"

TOP_K = 5


# ============================================================
# LLM
# ============================================================

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0
)


# ============================================================
# RAG QA
# ============================================================

def answer_question(query):
    """
    Retrieve relevant CFR sections and use the LLM
    to generate an answer based only on those sections.
    """

    # --------------------------------------------------------
    # 1. Create retriever
    # --------------------------------------------------------

    retriever = StructureAwareRetriever()

    # --------------------------------------------------------
    # 2. Retrieve relevant legal sections
    # --------------------------------------------------------

    results = retriever.retrieve(
        query,
        top_k=TOP_K,
        max_related=0
    )

    # --------------------------------------------------------
    # 3. Build legal context
    # --------------------------------------------------------

    context_parts = []

    for result in results:

        metadata = result.get(
            "metadata",
            {}
        )

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

        section_title = metadata.get(
            "section_title",
            ""
        )

        location = citation

        if paragraph:
            location += f" {paragraph}"

        if subparagraph:
            location += f" {subparagraph}"

        context_parts.append(
            f"""
Source: {location}
Section: {section_title}

Text:
{result.get("text", "")}
"""
        )

    context = "\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # 4. Create prompt
    # --------------------------------------------------------

    prompt = f"""
You are a legal question-answering assistant.

Answer the user's question using ONLY the legal
text provided in the context.

Do not use outside knowledge.
Do not invent legal requirements.
Do not make assumptions that are not supported
by the provided text.

Give a clear and concise answer.

After the answer, identify the relevant CFR
section(s) that support the answer.

USER QUESTION:
{query}

LEGAL CONTEXT:
{context}
"""

    # --------------------------------------------------------
    # 5. Ask the LLM
    # --------------------------------------------------------

    response = llm.invoke(
        prompt
    )

    return response.content


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    question = (
        "What must a foreign person do when land "
        "subsequently becomes agricultural land?"
    )

    print()
    print("=" * 80)
    print("LEGAL QUESTION")
    print("=" * 80)
    print(question)

    print()
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)

    answer = answer_question(
        question
    )

    print(answer)