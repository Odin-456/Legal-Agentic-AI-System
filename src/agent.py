import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from retrieval import LegalRetriever


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

MODEL_NAME = "gpt-5-mini"
TOP_K = 5


# ============================================================
# RETRIEVER
# ============================================================

retriever = LegalRetriever()


# ============================================================
# LEGAL RETRIEVAL TOOL
# ============================================================

@tool
def search_legal_documents(query: str) -> str:
    """
    Search the eCFR legal document collection.

    Use this tool when a question requires information
    from the eCFR legal documents.
    """

    results = retriever.retrieve(
        query,
        top_k=TOP_K
    )

    if not results:
        return "No relevant legal documents were found."

    passages = []

    for result in results:

        citation = result["citation"]
        paragraph = result["paragraph"]
        text = result["text"]

        location = citation

        if paragraph:
            location += paragraph

        passages.append(
            f"[{location}]\n{text}"
        )

    return "\n\n".join(passages)


# ============================================================
# LANGUAGE MODEL
# ============================================================

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)


# ============================================================
# AGENT
# ============================================================

agent = create_agent(
    model=llm,
    tools=[search_legal_documents],
    system_prompt="""
You are a legal information assistant specialising in
the United States Code of Federal Regulations (eCFR).

When answering legal questions:

1. Use the search_legal_documents tool to retrieve
   relevant eCFR passages.

2. Base your answer only on the retrieved passages.

3. Do not invent legal information.

4. Include the relevant eCFR citation in your answer.

5. If the retrieved information is insufficient,
   clearly state that the available documents do not
   provide enough information.

Provide a concise and clear answer.
""",
)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    query = (
        "What happens when land becomes agricultural "
        "land after being acquired?"
    )

    print("\nQUERY:")
    print(query)

    print("\nAGENT ANSWER:")

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        }
    )

    print(
        result["messages"][-1].content
    )
