import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from structure_retrieval import StructureAwareRetriever


load_dotenv()


MODEL_NAME = "gpt-5-mini"
TOP_K = 5
MAX_RELATED = 10


print("Initialising Structure-Aware NSW Property-Law Agent...")


retriever = StructureAwareRetriever()


def format_source(metadata):
    """
    Format NSW property-law metadata into a citation.
    """

    act = metadata.get(
        "act",
        "Unknown Act"
    )

    section = metadata.get(
        "section",
        "Unknown section"
    )

    citation = f"{act} § {section}"

    subsection = metadata.get("subsection")
    paragraph = metadata.get("paragraph")
    nested_paragraph = metadata.get("nested_paragraph")

    if subsection:
        citation += f" {subsection}"

    if paragraph:
        citation += f" {paragraph}"

    if nested_paragraph:
        citation += f" {nested_paragraph}"

    return citation


@tool
def search_property_law(query: str) -> str:
    """
    Search the NSW property-law legislation corpus.

    Use this tool whenever the user asks a question about
    NSW property law, including property ownership,
    registration, conveyancing, residential tenancy,
    strata schemes, or community land management.
    """

    results = retriever.retrieve(
        query,
        top_k=TOP_K,
        max_related=MAX_RELATED,
    )

    if not results:
        return (
            "No relevant NSW property-law legislation "
            "was found in the available corpus."
        )

    context_parts = []

    for i, result in enumerate(results, start=1):

        metadata = result.get(
            "metadata",
            {}
        )

        citation = format_source(metadata)

        heading = metadata.get(
            "heading",
            ""
        )

        text = result.get(
            "text",
            ""
        )

        context_parts.append(
            f"""
SOURCE {i}
ACT AND SECTION: {citation}
HEADING: {heading}

LEGAL TEXT:
{text}
"""
        )

    return "\n".join(context_parts)


llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)


agent = create_agent(
    model=llm,
    tools=[search_property_law],
    system_prompt="""
You are a NSW property-law research assistant.

Your job is to answer questions about NSW property law
using the supplied NSW property-law legislation.

The corpus contains:

- Real Property Act 1900
- Conveyancing Act 1919
- Residential Tenancies Act 2010
- Strata Schemes Management Act 2015
- Community Land Management Act 2021

When the user asks a legal question:

1. Use the property-law search tool.

2. Base the answer only on the retrieved NSW legislation.

3. Do not use outside legal knowledge to fill gaps.

4. Do not invent legal rules.

5. Do not invent Act names or section numbers.

6. Answer the question directly.

7. Explain the answer in clear, ordinary language.

8. Identify the relevant Act and section.

9. If several sections are relevant, identify the important
   sections separately.

10. If the retrieved material is insufficient, say so clearly.

11. Do not provide personalised legal advice.

12. Make clear that the response is general information
    based on the retrieved NSW legislation.

13. Do not refer to the legislation as CFR or US federal
    regulations.

14. Do not claim that a procedure is required unless the
    retrieved legislation supports that claim.

When appropriate, structure the answer as:

Short answer

What the legislation says

Important points

Sources

Keep the answer grounded in the retrieved legislation.
"""
)


if __name__ == "__main__":

    question = (
        "What happens if a landlord does not carry out "
        "required repairs?"
    )

    print()
    print("=" * 80)
    print("NSW PROPERTY-LAW AGENT")
    print("=" * 80)

    print()
    print(f"Question: {question}")

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    answer = result["messages"][-1].content

    print()
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(answer)