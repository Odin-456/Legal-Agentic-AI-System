import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from structure_retrieval import StructureAwareRetriever

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "gpt-5-mini"

TOP_K = 5

MAX_RELATED = 10


# ============================================================
# RETRIEVER
# ============================================================

print(
    "Initialising Structure-Aware NSW Property-Law Agent..."
)

retriever = StructureAwareRetriever()


# ============================================================
# SOURCE CITATION
# ============================================================

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

    subsection = metadata.get(
        "subsection"
    )

    paragraph = metadata.get(
        "paragraph"
    )

    nested_paragraph = metadata.get(
        "nested_paragraph"
    )

    if subsection:
        citation += f" {subsection}"

    if paragraph:
        citation += f" {paragraph}"

    if nested_paragraph:
        citation += f" {nested_paragraph}"

    return citation


# ============================================================
# PROPERTY-LAW SEARCH TOOL
# ============================================================


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

    for i, document in enumerate(
        results,
        start=1
    ):

        # The retriever returns LangChain Document objects.
        # Metadata is stored in document.metadata.
        metadata = document.metadata

        citation = format_source(
            metadata
        )

        heading = metadata.get(
            "heading",
            ""
        )

        context_parts.append(
            f"""
SOURCE {i}

ACT AND SECTION:
{citation}

HEADING:
{heading}

LEGAL TEXT:
{document.page_content}
"""
        )

    return "\n".join(
        context_parts
    )




# ============================================================
# LANGUAGE MODEL
# ============================================================

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
    api_key=os.getenv(
        "OPENAI_API_KEY"
    ),
)


# ============================================================
# AGENT
# ============================================================

agent = create_agent(
    model=llm,
    tools=[
        search_property_law
    ],
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

15. When the retrieved text contains the answer, use that
    text directly as the basis for the answer.

16. If the question specifies a section, subsection,
    paragraph, or nested paragraph, answer that specific
    provision first.

17. Do not discuss other subsections, paragraphs, or sections
    unless they are necessary to answer the question.

18. If the user asks about subsection (1), do not provide
    information from subsection (2) unless subsection (2) is
    necessary to understand the answer.

19. Keep the answer focused on the user's question and the
    retrieved legal text.

When appropriate, structure the answer as:

Short answer

What the legislation says

Important points

Sources

Keep the answer grounded in the retrieved legislation.
"""
)


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    question = (
        "What responsibilities does an association have "
        "under the Community Land Management Act 2021?"
    )

    print()
    print("=" * 80)
    print("NSW PROPERTY-LAW AGENT")
    print("=" * 80)

    print()
    print(
        f"Question: {question}"
    )

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

    answer = result[
        "messages"
    ][-1].content

    print()
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(answer)