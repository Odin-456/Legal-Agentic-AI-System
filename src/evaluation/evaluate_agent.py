import sys
from pathlib import Path


# ============================================================
# PYTHON PATH
# ============================================================

SRC_DIR = Path(__file__).resolve().parent.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# IMPORT BASELINE AGENT
# ============================================================

from agent import agent


# ============================================================
# TEST DATASET
# ============================================================

TEST_QUERIES = [

    {
        "query": "What is the definition of agricultural land under 7 CFR part 781?",
        "expected_citation": "7 CFR 781.2",
    },

    {
        "query": "Who qualifies as a foreign person under 7 CFR 781.2?",
        "expected_citation": "7 CFR 781.2",
    },

    {
        "query": "What information must a foreign person report under 7 CFR 781.3(f)(1)?",
        "expected_citation": "7 CFR 781.3",
    },

    {
        "query": "What are the reporting requirements for foreign investment in agricultural land?",
        "expected_citation": "7 CFR 781.3",
    },

    {
        "query": "What information must be reported about foreign individuals or governments holding significant interest or substantial control?",
        "expected_citation": "7 CFR 781.3",
    },

    {
        "query": "What is the purpose of the regulations concerning foreign investment in agricultural land?",
        "expected_citation": "7 CFR 781.1",
    },

    {
        "query": "What is AFIDA under 7 CFR 781.2?",
        "expected_citation": "7 CFR 781.2",
    },

    {
        "query": "What does agricultural land include under the regulations?",
        "expected_citation": "7 CFR 781.2",
    },

    {
        "query": "What information must be provided about agricultural land in a report?",
        "expected_citation": "7 CFR 781.3",
    },

    {
        "query": "When must a foreign person file a report concerning agricultural land?",
        "expected_citation": "7 CFR 781.3",
    },

    {
        "query": "What happens when land becomes agricultural land after being acquired?",
        "expected_citation": "7 CFR 781.3",
    },

    {
        "query": "What information is required concerning the location and acreage of agricultural land?",
        "expected_citation": "7 CFR 781.3",
    },

    {
        "query": "What is the legal name and address information required for foreign persons?",
        "expected_citation": "7 CFR 781.3",
    },

    {
        "query": "What information must be reported concerning the use of agricultural land?",
        "expected_citation": "7 CFR 781.3",
    },

    {
        "query": "What is a foreign individual under 7 CFR 781.2?",
        "expected_citation": "7 CFR 781.2",
    },

    {
        "query": "What is the meaning of any interest under 7 CFR 781.2?",
        "expected_citation": "7 CFR 781.2",
    },

    {
        "query": "What are the requirements for reporting foreign ownership of agricultural land?",
        "expected_citation": "7 CFR 781.3",
    },

    {
        "query": "Which regulation implements the Agricultural Foreign Investment Disclosure Act?",
        "expected_citation": "7 CFR 781.1",
    },

    {
        "query": "What form must a foreign person file under the reporting requirements?",
        "expected_citation": "7 CFR 781.3",
    },

    {
        "query": "What agency receives reports concerning foreign investment in agricultural land?",
        "expected_citation": "7 CFR 781.3",
    },
]


# ============================================================
# RUN AGENT
# ============================================================

def run_agent(query):
    """
    Send a query to the baseline LangChain agent
    and return the final answer.
    """

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

    messages = result.get(
        "messages",
        []
    )

    if not messages:
        return ""

    return messages[-1].content


# ============================================================
# CITATION CHECK
# ============================================================

def citation_is_correct(
    answer,
    expected_citation
):
    """
    Check whether the expected CFR citation
    appears in the generated answer.
    """

    if not answer:
        return False

    return expected_citation in answer


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 60)
    print("BASELINE AGENT EVALUATION")
    print("=" * 60)

    total = len(
        TEST_QUERIES
    )

    correct = 0

    failed = []


    # ========================================================
    # PROCESS QUESTIONS
    # ========================================================

    for number, test_case in enumerate(
        TEST_QUERIES,
        start=1
    ):

        query = test_case[
            "query"
        ]

        expected = test_case[
            "expected_citation"
        ]

        print(
            f"\n{number:2d}. {query}"
        )


        try:

            answer = run_agent(
                query
            )


            # ------------------------------------------------
            # Check expected citation
            # ------------------------------------------------

            correct_citation = citation_is_correct(
                answer,
                expected
            )


            if correct_citation:

                correct += 1

                print(
                    f"    PASS | Expected: {expected}"
                )

            else:

                print(
                    f"    FAIL | Expected: {expected}"
                )

                failed.append(
                    {
                        "number": number,
                        "query": query,
                        "expected": expected,
                        "answer": answer,
                    }
                )


        except Exception as error:

            print(
                f"    ERROR | {error}"
            )

            failed.append(
                {
                    "number": number,
                    "query": query,
                    "expected": expected,
                    "answer": f"ERROR: {error}",
                }
            )


    # ========================================================
    # FINAL METRICS
    # ========================================================

    accuracy = (
        correct / total
    )


    print("\n" + "=" * 60)
    print("BASELINE AGENT RESULTS")
    print("=" * 60)


    print(
        f"Total questions: {total}"
    )


    print(
        f"Citation correct: {correct}"
    )


    print(
        f"Citation accuracy: {accuracy:.4f}"
    )


    print(
        f"Citation accuracy: {accuracy * 100:.2f}%"
    )


    # ========================================================
    # FAILED QUESTIONS
    # ========================================================

    print("\n" + "=" * 60)
    print("FAILED QUESTIONS")
    print("=" * 60)


    if not failed:

        print(
            "No failed questions."
        )

    else:

        for item in failed:

            print(
                f"\n{item['number']}. "
                f"{item['query']}"
            )

            print(
                f"Expected: "
                f"{item['expected']}"
            )

            print(
                "Answer:"
            )

            print(
                item["answer"]
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()