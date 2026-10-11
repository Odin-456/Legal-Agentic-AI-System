from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


MODEL_NAME = "gpt-5-mini"


llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
)


def expand_query(question: str) -> str:
    """
    Convert a user's NSW property-law question into
    one concise search query.
    """

    prompt = f"""
You are a query expansion assistant for a NSW property-law
retrieval system.

Convert the user's question into ONE concise search query
for retrieving relevant sections from the following NSW
property-law legislation:

- Real Property Act 1900
- Conveyancing Act 1919
- Residential Tenancies Act 2010
- Strata Schemes Management Act 2015
- Community Land Management Act 2021

Rules:

1. Preserve the legal meaning of the user's question.

2. Keep important legal terms.

3. Do not invent facts.

4. Do not invent legislation or section numbers.

5. Do not answer the question.

6. Do not provide legal advice.

7. Return ONLY the expanded search query.

USER QUESTION:

{question}

EXPANDED SEARCH QUERY:
"""

    response = llm.invoke(prompt)

    expanded_query = response.content.strip()

    if not expanded_query:
        return question

    return expanded_query


if __name__ == "__main__":

    question = (
        "What happens if a landlord does not "
        "carry out required repairs?"
    )

    expanded = expand_query(question)

    print("Original:", question)
    print("Expanded:", expanded)