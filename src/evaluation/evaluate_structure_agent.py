import sys
from pathlib import Path

# ============================================================
# ADD src/ TO PYTHON PATH
# ============================================================

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# IMPORT STRUCTURE-AWARE AGENT
# ============================================================

from structure_agent import answer_question


# ============================================================
# TEST DATASET
# ============================================================

TEST_QUERIES = [

    {
        "query": "What is the definition of agricultural land under 7 CFR part 781?",
        "expected_citations": ["7 CFR 781.2"],
    },

    {
        "query": "Who qualifies as a foreign person under 7 CFR 781.2?",
        "expected_citations": ["7 CFR 781.2"],
    },

    {
        "query": "What information must a foreign person report under 7 CFR 781.3(f)(1)?",
        "expected_citations": ["7 CFR 781.3"],
    },

    {
        "query": "What are the reporting requirements for foreign investment in agricultural land?",
        "expected_citations": ["7 CFR 781.3"],
    },

    {
        "query": "What information must be reported about foreign individuals or governments holding significant interest or substantial control?",
        "expected_citations": ["7 CFR 781.3"],
    },

    {
        "query": "What is the purpose of the regulations concerning foreign investment in agricultural land?",
        "expected_citations": ["7 CFR 781.1"],
    },

    {
        "query": "What is AFIDA under 7 CFR 781.2?",
        "expected_citations": ["7 CFR 781.2"],
    },

    {
        "query": "What does agricultural land include under the regulations?",
        "expected_citations": ["7 CFR 781.2"],
    },

    {
        "query": "What information must be provided about agricultural land in a report?",
        "expected_citations": ["7 CFR 781.3"],
    },

    {
        "query": "When must a foreign person file a report concerning agricultural land?",
        "expected_citations": ["7 CFR 781.3"],
    },

    {
        "query": "What happens when land becomes agricultural land after being acquired?",
        "expected_citations": ["7 CFR 781.3"],
    },

    {
        "query": "What information is required concerning the location and acreage of agricultural land?",
        "expected_citations": ["7 CFR 781.3"],
    },

    {
        "query": "What is the legal name and address information required for foreign persons?",
        "expected_citations": ["7 CFR 781.3"],
    },

    {
        "query": "What information must be reported concerning the use of agricultural land?",
        "expected_citations": ["7 CFR 781.3"],
    },

    {
        "query": "What is a foreign individual under 7 CFR 781.2?",
        "expected_citations": ["7 CFR 781.2"],
    },

    {
        "query": "What is the meaning of any interest under 7 CFR 781.2?",
        "expected_citations": ["7 CFR 781.2"],
    },

    {
        "query": "What are the requirements for reporting foreign ownership of agricultural land?",
        "expected_citations": ["7 CFR 781.3"],
    },

    {
        "query": "Which regulation implements the Agricultural Foreign Investment Disclosure Act?",
        "expected_citations": ["7 CFR 781.1"],
    },

    {
        "query": "What form must a foreign person file under the reporting requirements?",
        "expected_citations": ["7 CFR 781.3"],
    },

    {
        "query": "What agency receives reports concerning foreign investment in agricultural land?",
        "expected_citations": ["7 CFR 781.3"],
    },
]


# ============================================================
# CITATION CHECK
# ============================================================

def citation_is_present(
    answer,
    expected_citations
):
    """
    Check whether at least one expected citation
    appears in the agent's answer.
    """

    answer_lower = answer.lower()

    for citation in expected_citations:

        if citation.lower() in answer_lower:
            return True

    return False


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 60)
    print("STRUCTURE-AWARE AGENT EVALUATION")
    print("=" * 60)

    total_questions = len(
        TEST_QUERIES
    )

    passed = 0

    failed = []

    for number, test_case in enumerate(
        TEST_QUERIES,
        start=1
    ):

        query = test_case["query"]

        expected_citations = (
            test_case["expected_citations"]
        )

        print(
            f"\n{number:2d}. {query}"
        )

        try:

            # ----------------------------------------------
            # Run structure-aware agent
            # ----------------------------------------------

            answer, retrieved_results = (
                answer_question(query)
            )

            # ----------------------------------------------
            # Check citation in generated answer
            # ----------------------------------------------

            citation_correct = (
                citation_is_present(
                    answer,
                    expected_citations
                )
            )

            if citation_correct:

                passed += 1

                print(
                    f"    PASS | Expected: "
                    f"{', '.join(expected_citations)}"
                )

            else:

                failed.append(
                    {
                        "number": number,
                        "query": query,
                        "expected_citations":
                            expected_citations,
                        "answer": answer,
                    }
                )

                print(
                    f"    FAIL | Expected: "
                    f"{', '.join(expected_citations)}"
                )

        except Exception as error:

            failed.append(
                {
                    "number": number,
                    "query": query,
                    "expected_citations":
                        expected_citations,
                    "error": str(error),
                }
            )

            print(
                f"    ERROR | {error}"
            )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    accuracy = (
        passed / total_questions
    )

    print("\n" + "=" * 60)
    print("STRUCTURE-AWARE AGENT RESULTS")
    print("=" * 60)

    print(
        f"Total questions: "
        f"{total_questions}"
    )

    print(
        f"Citation correct: "
        f"{passed}"
    )

    print(
        f"Citation accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Citation accuracy: "
        f"{accuracy * 100:.2f}%"
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
                "Expected: "
                f"{', '.join(item['expected_citations'])}"
            )

            if "error" in item:

                print(
                    "Error: "
                    f"{item['error']}"
                )

            else:

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
