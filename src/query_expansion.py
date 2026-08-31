import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

MODEL_NAME = "gpt-5-mini"


# ============================================================
# INITIALISE LLM
# ============================================================

print("Initialising Query Expansion...")

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)


# ============================================================
# QUERY EXPANSION
# ============================================================

def expand_query(query):
    """
    Convert a natural-language legal question into a
    concise retrieval-focused legal search query.

    The expansion is deliberately conservative.

    The model should:
        - preserve the original legal meaning
        - preserve the legal subject
        - preserve the triggering condition
        - preserve the actor
        - preserve the requested action/obligation
        - preserve deadlines and time periods
        - add only close legal synonyms

    The model must NOT:
        - answer the question
        - invent legal rules
        - invent citations
        - introduce unrelated legal topics
        - change the legal trigger
        - introduce agencies or programs
    """

    prompt = f"""
You are a legal information retrieval assistant.

Convert the user's legal question into ONE concise
search query for retrieving the most relevant
eCFR regulation.

Be CONSERVATIVE.

The expanded query MUST preserve the legal meaning
of the original question.

RULES:

1. Keep the original legal subject.

2. Keep the original triggering event or condition.

3. Keep the original actor, person, organisation,
   or entity if one is stated.

4. Keep the original action, obligation,
   requirement, restriction, or consequence
   being asked about.

5. Keep any specific deadline or time period.

6. You may add a small number of close legal
   synonyms that could commonly appear in
   regulations.

7. Do NOT replace a legal condition with a
   different legal condition.

For example:

Original:
"land subsequently becomes agricultural land"

Do NOT replace this with:
"land is converted to agricultural use"

unless the original question specifically refers
to conversion of land use.

8. Do NOT introduce unrelated legal concepts,
   including:

   - zoning
   - taxation
   - land registry
   - eminent domain
   - compensation
   - planning permission
   - development
   - subdivision

   unless they are explicitly mentioned in the
   user's question.

9. Do NOT introduce agencies, government programs,
   or regulations that are not mentioned in the
   user's question.

10. Do NOT invent facts.

11. Do NOT answer the question.

12. Do NOT provide a legal citation.

13. Do NOT provide explanations.

14. Return ONLY the search query.

15. Keep the query concise.

USER QUESTION:
{query}
"""

    response = llm.invoke(prompt)

    expanded_query = response.content.strip()

    # --------------------------------------------------------
    # Remove accidental quotation marks
    # --------------------------------------------------------

    if (
        expanded_query.startswith('"')
        and expanded_query.endswith('"')
    ):
        expanded_query = expanded_query[1:-1]

    return expanded_query.strip()


# ============================================================
# TEST QUERIES
# ============================================================

if __name__ == "__main__":

    test_questions = [

        (
            "What happens when land becomes "
            "agricultural land after being acquired?"
        ),

        (
            "What must a foreign person do when "
            "land subsequently becomes agricultural land?"
        ),

        (
            "When must a report be filed for "
            "agricultural land?"
        ),

        (
            "foreign person land subsequently becomes "
            "agricultural land report within 90 days"
        ),

        (
            "What reporting requirement applies when "
            "land becomes agricultural land?"
        ),
    ]


    # ========================================================
    # RUN TESTS
    # ========================================================

    print()
    print("=" * 80)
    print("QUERY EXPANSION TEST")
    print("=" * 80)

    for question in test_questions:

        print()
        print("ORIGINAL:")
        print(question)

        expanded = expand_query(
            question
        )

        print()
        print("EXPANDED:")
        print(expanded)

        print("-" * 80)
