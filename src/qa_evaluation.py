from legal_agent import search_legal_regulations


QUESTIONS = [
    "What must a foreign person do when land subsequently becomes agricultural land?",
    "When must a foreign person report an interest in agricultural land?",
    "What happens when agricultural land later ceases to be agricultural?",
    "Where must reports concerning agricultural land be filed?",
    "What information must be included about agricultural land in the report?",
]


def main():

    print("=" * 80)
    print("QA EVALUATION")
    print("=" * 80)

    for i, question in enumerate(QUESTIONS, start=1):

        print()
        print("-" * 80)
        print(f"Question {i}")
        print(f"Question: {question}")
        print("-" * 80)

        answer = search_legal_regulations.invoke({
            "query": question
        })

        print()
        print("ANSWER:")
        print(answer)


if __name__ == "__main__":
    main()