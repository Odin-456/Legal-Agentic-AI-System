import json
import os
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv


# ============================================================
# Configuration
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY was not found in the .env file."
    )

client = OpenAI(api_key=api_key)

QUESTIONS_FILE = Path(
    "data/evaluation/qa_evaluation_questions.json"
)

OUTPUT_FILE = Path(
    "data/evaluation/chatgpt_baseline_results.json"
)

MODEL = "gpt-5-mini"


# ============================================================
# Load evaluation questions
# ============================================================

def load_questions():

    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation questions not found: {QUESTIONS_FILE}"
        )

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    questions = []

    # --------------------------------------------------------
    # Current dataset format
    #
    # {
    #     "dataset_name": "...",
    #     "description": "...",
    #     "question_count": 59,
    #     "acts": [...],
    #     "questions": [
    #         {
    #             "id": 1,
    #             "act": "...",
    #             "section": "...",
    #             "question": "..."
    #         }
    #     ]
    # }
    # --------------------------------------------------------

    if isinstance(data, dict) and "questions" in data:

        question_list = data["questions"]

        if not isinstance(question_list, list):
            raise ValueError(
                "The 'questions' field must be a list."
            )

        for item in question_list:

            if not isinstance(item, dict):
                continue

            question = item.get("question", "")

            if not question:
                continue

            questions.append(
                {
                    "id": item.get("id"),
                    "act": item.get("act"),
                    "section": item.get("section"),
                    "question": question,
                }
            )

    # --------------------------------------------------------
    # Alternative format:
    #
    # [
    #     {"id": 1, "question": "..."}
    # ]
    # --------------------------------------------------------

    elif isinstance(data, list):

        for index, item in enumerate(
            data,
            start=1
        ):

            if not isinstance(item, dict):
                continue

            question = item.get("question", "")

            if not question:
                continue

            questions.append(
                {
                    "id": item.get("id", index),
                    "act": item.get("act"),
                    "section": item.get("section"),
                    "question": question,
                }
            )

    else:

        raise ValueError(
            "Unsupported evaluation-question JSON format."
        )

    return questions


# ============================================================
# Generate answer using GPT-5-mini only
# ============================================================

def answer_question(question):

    system_prompt = """
You are a general-purpose AI assistant answering questions
about Australian property law.

Answer the user's question using your general knowledge.

Important:
- Do not retrieve documents.
- Do not use a provided legislation corpus.
- Do not use the project's NSW property-law database.
- Answer as accurately and clearly as possible.
- If you are uncertain, say so.
- Do not invent legislation sections or legal authorities.
"""

    response = client.responses.create(
        model=MODEL,
        instructions=system_prompt,
        input=question,
    )

    return response.output_text.strip()


# ============================================================
# Main evaluation
# ============================================================

def main():

    print("=" * 70)
    print("CHATGPT BASELINE EVALUATION")
    print("=" * 70)

    print(f"Model: {MODEL}")
    print(f"Questions file: {QUESTIONS_FILE}")
    print()

    questions = load_questions()

    print(
        f"Evaluation questions: {len(questions)}"
    )
    print()

    # --------------------------------------------------------
    # Make sure we are using the complete dataset
    # --------------------------------------------------------

    if len(questions) != 59:

        raise ValueError(
            f"Expected 59 evaluation questions, "
            f"but found {len(questions)}."
        )

    results = []

    # --------------------------------------------------------
    # Generate answers
    # --------------------------------------------------------

    for index, item in enumerate(
        questions,
        start=1
    ):

        question_id = item["id"]
        act = item["act"]
        section = item["section"]
        question = item["question"]

        print("=" * 70)
        print(
            f"QUESTION {index}/{len(questions)}"
        )
        print(f"ID: {question_id}")
        print(f"Act: {act}")
        print(f"Section: {section}")
        print("-" * 70)
        print(question)
        print()

        try:

            answer = answer_question(question)

            print("ANSWER")
            print("-" * 70)
            print(answer)
            print()

            results.append(
                {
                    "id": question_id,
                    "act": act,
                    "section": section,
                    "question": question,
                    "answer": answer,
                    "model": MODEL,
                    "system": (
                        "GPT-5-mini general-purpose "
                        "baseline without retrieval"
                    ),
                }
            )

        except Exception as e:

            print("ERROR")
            print(str(e))
            print()

            results.append(
                {
                    "id": question_id,
                    "act": act,
                    "section": section,
                    "question": question,
                    "answer": "",
                    "model": MODEL,
                    "system": (
                        "GPT-5-mini general-purpose "
                        "baseline without retrieval"
                    ),
                    "error": str(e),
                }
            )

    # ========================================================
    # Save results
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # Final summary
    # ========================================================

    successful = sum(
        1
        for result in results
        if result.get("answer")
    )

    failed = len(results) - successful

    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Total questions: {len(results)}"
    )

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed: {failed}"
    )

    print()

    print(
        f"Results saved to: {OUTPUT_FILE}"
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()

