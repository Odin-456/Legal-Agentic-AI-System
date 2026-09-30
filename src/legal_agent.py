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

# Create the structure-aware legal retriever
retriever = StructureAwareRetriever()


# ============================================================
# SOURCE CITATION
# ============================================================

# Convert legal metadata into a readable citation
def format_source(metadata):
    citation = f"{metadata.get('act', 'Unknown Act')} § {metadata.get('section', 'Unknown section')}"

    if metadata.get("subsection"):
        citation += f" {metadata['subsection']}"

    if metadata.get("paragraph"):
        citation += f" {metadata['paragraph']}"

    if metadata.get("nested_paragraph"):
        citation += f" {metadata['nested_paragraph']}"

    return citation


# ============================================================
# PROPERTY-LAW SEARCH TOOL
# ============================================================

# This tool searches the NSW property-law legislation
@tool
def search_property_law(query: str) -> str:
    """Search the NSW property-law legislation corpus."""

    # Retrieve the most relevant legal sections
    results = retriever.retrieve(
        query,
        top_k=TOP_K,
        max_related=MAX_RELATED,
    )

    # Return a message if nothing relevant is found
    if not results:
        return "No relevant NSW property-law legislation was found."

    context = []

    # Format each retrieved result for the language model
    for i, result in enumerate(results, 1):
        metadata = result.get("metadata", {})

        citation = format_source(metadata)
        heading = metadata.get("heading", "")

        # Get related chunks from the same legal section
        documents = result.get("related_documents", [])

        # Use the main document if there are no related chunks
        if not documents and result.get("document"):
            documents = [result["document"]]

        # Combine the legal text
        legal_text = "\n".join(
            document.page_content
            for document in documents
        )

        # Add the source and legal text to the context
        context.append(
            f"""
SOURCE {i}
ACT AND SECTION: {citation}
HEADING: {heading}
LEGAL TEXT:
{legal_text}
"""
        )

    return "\n".join(context)


# ============================================================
# LANGUAGE MODEL
# ============================================================

# Create the language model used to answer questions
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)


# ============================================================
# AGENT
# ============================================================

# Create the legal research agent
agent = create_agent(
    model=llm,
    tools=[search_property_law],
    system_prompt="""
You are a NSW property-law research assistant.

Answer questions using only the retrieved NSW property-law legislation.

The corpus contains:
- Real Property Act 1900
- Conveyancing Act 1919
- Residential Tenancies Act 2010
- Strata Schemes Management Act 2015
- Community Land Management Act 2021

Rules:
1. Use the property-law search tool for legal questions.
2. Base answers only on the retrieved legislation.
3. Do not use outside legal knowledge to fill gaps.
4. Do not invent legal rules, Act names, or section numbers.
5. Answer directly in clear, ordinary language.
6. Identify the relevant Act and section.
7. If several sections are relevant, identify them separately.
8. If the retrieved material is insufficient, say so.
9. Do not provide personalised legal advice.
10. State that the response is general information based on the retrieved NSW legislation.
11. Do not refer to the legislation as CFR or US federal regulations.
12. Do not claim a procedure is required unless the retrieved legislation supports it.
13. If a specific section, subsection, paragraph, or nested paragraph is requested, answer that provision first.
14. Do not discuss other provisions unless necessary to answer the question.
15. Keep the answer focused on the user's question.

When appropriate, use:
Short answer
What the legislation says
Important points
Sources
"""
)


# ============================================================
# MANUAL TEST
# ============================================================

# Run a simple test when this file is executed directly
if __name__ == "__main__":

    question = (
        "What responsibilities does an association have "
        "under the Community Land Management Act 2021?"
    )

    # Send the question to the agent
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    })

    # Print the final answer
    print(result["messages"][-1].content)