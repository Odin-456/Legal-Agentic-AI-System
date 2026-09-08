import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from rag_pipeline import RAGPipeline


load_dotenv()

MODEL_NAME = "gpt-5-mini"
TOP_K = 5


print("Initialising NSW Property-Law RAG QA...")


llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)


pipeline = RAGPipeline()


def format_source(metadata):
    """
    Format current NSW property-law metadata
    into a readable citation.
    """

    act = metadata.get("act", "Unknown Act")
    section = metadata.get("section", "Unknown section")

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


def answer_question(query):
    """
    Retrieve NSW property-law legislation and generate
    an answer using only the retrieved legal text.
    """

    results = pipeline.retrieve(
        query,
        top_k=TOP_K,
    )

    if not results:
        return (
            "I could not find relevant NSW property-law "
            "legislation in the available documents."
        )

    context_parts = []

    for i, result in enumerate(results, start=1):

        metadata = result.get("metadata", {})

        source = format_source(metadata)

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
CITATION: {source}
HEADING: {heading}

LEGAL TEXT:
{text}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are an Australian property-law question-answering
assistant.

Answer the user's question using ONLY the NSW property-law
legislation provided in the retrieved context.

The available legislation includes:

- Real Property Act 1900
- Conveyancing Act 1919
- Residential Tenancies Act 2010
- Strata Schemes Management Act 2015
- Community Land Management Act 2021

RULES:

1. Use only information supported by the retrieved legal text.

2. Do not use outside legal knowledge.

3. Do not invent legal rules.

4. Do not invent section numbers.

5. Do not invent facts.

6. Answer the user's question directly.

7. Explain the legislation in clear and simple language.

8. Preserve the meaning of the legislation.

9. Identify the relevant Act and section where supported
   by the retrieved text.

10. If the retrieved text is insufficient, clearly say so.

11. Do not pretend to provide personalised legal advice.

12. State that the answer is general information based on
    the retrieved NSW legislation.

USER QUESTION:

{query}

RETRIEVED NSW PROPERTY-LAW CONTEXT:

{context}

ANSWER:
"""

    response = llm.invoke(prompt)

    return response.content


if __name__ == "__main__":

    question = (
        "What happens if a landlord does not carry out "
        "required repairs?"
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

    answer = answer_question(question)

    print(answer)