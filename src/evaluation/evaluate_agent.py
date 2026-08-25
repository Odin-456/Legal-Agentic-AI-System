import json
from pathlib import Path

from agent import agent


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_PATH = Path(
    "data/evaluation/agent_results.json"
)


# ============================================================
# TEST QUESTIONS
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
# RUN AGENT
# ============================================================

def run_agent(query):

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

    return result["messages"][-1].content


# ============================================================
# MAIN
# ============================================================

def main():

    results = []

    print("=" * 60)
    print("AGENT EVALUATION")
    print("=" * 60)

    for number, test in enumerate(
        TEST_QUERIES,
        start=1
    ):

        query = test["query"]

        print(f"\n[{number}/20]")
        print(f"Question: {query}")

        try:

            answer = run_agent(query)

            print("Answer:")
            print(answer)

            results.append(
                {
                    "id": number,
                    "query": query,
                    "expected_citations": test[
                        "expected_citations"
                    ],
                    "answer": answer,
                }
            )

        except Exception as error:

            print(
                f"ERROR: {error}"
            )

            results.append(
                {
                    "id": number,
                    "query": query,
                    "expected_citations": test[
                        "expected_citations"
                    ],
                    "answer": None,
                    "error": str(error),
                }
            )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("AGENT EVALUATION COMPLETE")
    print("=" * 60)

    print(
        f"Results saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
