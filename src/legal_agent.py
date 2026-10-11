import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from structure_retrieval import StructureAwareRetriever

load_dotenv()

MODEL_NAME = "gpt-5-mini"
TOP_K = 5

print("Initialising Structure-Aware NSW Property-Law Agent...")

retriever = StructureAwareRetriever()

def format_source(metadata):
    """Format NSW property-law metadata into a citation."""
    act = metadata.get("act", "Unknown Act")
    section = metadata.get("section", "Unknown section")
    citation = f"{act} § {section}"

    if metadata.get("subsection"):
        citation += f" {metadata['subsection']}"
    if metadata.get("paragraph"):
        citation += f" {metadata['paragraph']}"
    if metadata.get("nested_paragraph"):
        citation += f" {metadata['nested_paragraph']}"

    return citation


@tool
def search_property_law(query: str) -> str:
    """
    Search the NSW property-law legislation corpus.

    Use this tool for questions about NSW property law,
    including property ownership, registration, conveyancing,
    residential tenancy, strata, and community land.
    """
    results = retriever.retrieve(query, top_k=TOP_K)

    if not results:
        return (
            "No relevant NSW property-law legislation "
            "was found in the available corpus."
        )

    context_parts = []

    for i, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        citation = format_source(metadata)
        heading = metadata.get("heading", "")
        related_documents = result.get("related_documents", [])
        text_parts = []

        for document in related_documents:
            text_parts.append(document.page_content)

        if not text_parts:
            document = result.get("document")
            if document is not None:
                text_parts.append(document.page_content)

        legal_text = "\n".join(text_parts)

        context_parts.append(
            f"""
SOURCE {i}

ACT AND SECTION:
{citation}

HEADING:
{heading}

LEGAL TEXT:
{legal_text}
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
9. If several sections are relevant, identify the important sections separately.
10. If the retrieved material is insufficient, say so clearly.
11. Do not provide personalised legal advice.
12. Make clear that the response is general information based on the retrieved NSW legislation.
13. Do not refer to the legislation as CFR or US federal regulations.
14. Do not claim that a procedure is required unless the retrieved legislation supports that claim.
15. When the retrieved text contains the answer, use that text directly as the basis for the answer.
16. If the question specifies a section, subsection, paragraph, or nested paragraph, answer that specific provision first.
17. Do not discuss other subsections, paragraphs, or sections unless they are necessary to answer the question.
18. If the user asks about subsection (1), do not provide information from subsection (2) unless subsection (2) is necessary to understand the answer.
19. Keep the answer focused on the user's question and the retrieved legal text.

Structure every answer using this format:

Short answer:

Give a short direct answer to the question in one paragraph.

Key points:

- **Registration:** Explain the point using the retrieved NSW legislation.
- **Vendor obligations:** Explain the point using the retrieved NSW legislation.
- **Other relevant requirement:** Explain the point using the retrieved NSW legislation.

Relevant legislation:

- **Act name, s X:** Briefly explain how this provision relates to the answer.
- **Act name, s X:** Briefly explain how this provision relates to the answer.

(Source: Act name, s X)

Formatting requirements:

- Put "Short answer:" on its own line.
- Leave a blank line after "Short answer:".
- Put "Key points:" on its own line.
- Leave a blank line after "Key points:".
- Put "Relevant legislation:" on its own line.
- Leave a blank line after "Relevant legislation:".
- Put every bullet point on its own separate line.
- Every bullet point must start with "- ".
- Start each bullet with a short bold topic, such as **Registration:** or **Vendor obligations:**.
- Do not use the words "Important point".
- Do not use "Important point" as a heading or bullet label.
- Never put multiple bullet points on the same line.
- Never put a heading and a bullet point on the same line.
- Leave a blank line between sections.
- Use **bold text** for important terms and topic names.
- Keep paragraphs short and readable.
- Do not put the entire answer into one paragraph.
- Do not include irrelevant legislation.
- Only cite sections that appear in the retrieved context.
- Do not invent section numbers or citations.
- Keep the answer concise and focused on the user's question.

The answer should follow a simple document-based RAG style:

Short answer first, followed by key points, relevant legislation,
and source citation.

Keep the answer grounded in the retrieved legislation.
"""
)

if __name__ == "__main__":
    question = (
        "What responsibilities does an association have "
        "under the Community Land Management Act 2021?"
    )

    print()
    print("=" * 60)
    print("NSW PROPERTY-LAW AGENT")
    print("=" * 60)
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
    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(answer)
