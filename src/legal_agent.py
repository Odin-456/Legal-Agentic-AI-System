from structure_retrieval import StructureAwareRetriever
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI


# ============================================================
# INITIALISE RETRIEVER
# ============================================================

retriever = StructureAwareRetriever()


# ============================================================
# LEGAL SEARCH TOOL
# ============================================================

@tool
def search_legal_regulations(query: str) -> str:
    """
    Search the eCFR legal database for relevant regulations.

    Use this tool whenever the user asks a question about
    US federal regulations or legal requirements.
    """

    results = retriever.retrieve(
        query,
        top_k=5,
        max_related=5
    )

    if not results:
        return "No relevant legal regulations were found."

    context = []

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

        context.append(
            f"""
Source: {location}
Section: {section_title}

Text:
{result.get("text", "")}
"""
        )

    return "\n".join(context)


# ============================================================
# LLM
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# ============================================================
# AGENT
# ============================================================

agent = create_agent(
    model=llm,
    tools=[
        search_legal_regulations
    ],
    system_prompt="""
You are a legal research assistant.

Your job is to answer questions about US federal
regulations using the legal search tool.

When a question requires legal information:

1. Use the legal search tool.
2. Examine the retrieved regulations.
3. Answer using only the retrieved legal text.
4. Do not invent legal requirements.
5. Clearly identify the relevant CFR section.

If the retrieved information is insufficient,
say so instead of guessing.
"""
)


# ============================================================
# TEST AGENT
# ============================================================

if __name__ == "__main__":

    question = (
        "What must a foreign person do when land "
        "subsequently becomes agricultural land?"
    )

    print()
    print("=" * 80)
    print("LEGAL AGENT")
    print("=" * 80)

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        }
    )

    print()
    print("=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)

    print(
        result["messages"][-1].content
    )