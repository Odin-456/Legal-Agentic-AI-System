from legal_agent import agent


def main():

    print("=" * 80)
    print("LEGAL AGENTIC AI SYSTEM")
    print("=" * 80)
    print()
    print("Ask a question about US federal regulations.")
    print("Type 'exit' to quit.")
    print()

    while True:

        question = input("Legal question: ").strip()

        if question.lower() == "exit":
            print("Goodbye.")
            break

        if not question:
            print("Please enter a question.")
            continue

        print()
        print("-" * 80)
        print("SEARCHING LEGAL DATABASE...")
        print("-" * 80)

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            }
        )

        answer = result["messages"][-1].content

        print()
        print("=" * 80)
        print("ANSWER")
        print("=" * 80)
        print()
        print(answer)
        print()


if __name__ == "__main__":
    main()