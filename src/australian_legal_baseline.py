import json
import os
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "isaacus/open-australian-legal-llm"

QUESTIONS_FILE = "data/evaluation/qa_evaluation_questions.json"

OUTPUT_FILE = "data/evaluation/australian_legal_baseline_results.json"

MAX_NEW_TOKENS = 250


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


# ============================================================
# LOAD QUESTIONS
# ============================================================

def load_questions(path):
    """
    Load the 59 evaluation questions from the
    'questions' array in the JSON dataset.
    """

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            "Expected the evaluation dataset to be a JSON object."
        )

    questions = data.get("questions", [])

    if not isinstance(questions, list):
        raise ValueError(
            "The 'questions' field must be a list."
        )

    cleaned_questions = []

    for item in questions:

        if not isinstance(item, dict):
            continue

        question = item.get("question")

        if not question:
            continue

        cleaned_questions.append(
            {
                "id": item.get("id"),
                "act": item.get("act"),
                "section": item.get("section"),
                "question": question,
            }
        )

    return cleaned_questions


# ============================================================
# CLEAN ANSWER
# ============================================================

def clean_answer(answer):

    answer = answer.strip()

    # Remove excessive blank lines
    while "\n\n\n" in answer:
        answer = answer.replace("\n\n\n", "\n\n")

    return answer


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(model, tokenizer, question):

    prompt = f"""You are an Australian legal assistant.

Answer the following question about Australian law.

Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():

        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Only decode newly generated tokens
    generated_ids = output_ids[0][input_ids.shape[1]:]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return clean_answer(answer)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("OPEN AUSTRALIAN LEGAL LLM EVALUATION")
    print("=" * 70)

    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print()

    # --------------------------------------------------------
    # Load evaluation questions
    # --------------------------------------------------------

    print("Loading evaluation questions...")

    questions = load_questions(QUESTIONS_FILE)

    print(f"Evaluation questions: {len(questions)}")
    print()

    if len(questions) != 59:

        print(
            f"WARNING: Expected 59 questions, "
            f"but found {len(questions)}."
        )

        print("Please check the evaluation dataset.")

        return

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Tokenizer loaded.")
    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=(
            torch.float16
            if DEVICE in ["mps", "cuda"]
            else torch.float32
        ),
    )

    model.to(DEVICE)
    model.eval()

    print("Model loaded.")
    print()

    # --------------------------------------------------------
    # Evaluate all 59 questions
    # --------------------------------------------------------

    results = []

    total = len(questions)

    for index, item in enumerate(questions, start=1):

        question_id = item["id"]
        act = item["act"]
        section = item["section"]
        question = item["question"]

        print("=" * 70)
        print(f"QUESTION {index}/{total}")
        print(f"ID: {question_id}")
        print(f"Act: {act}")
        print(f"Section: {section}")
        print("=" * 70)

        print(question)
        print()

        start_time = time.time()

        try:

            answer = generate_answer(
                model,
                tokenizer,
                question,
            )

            elapsed = time.time() - start_time

            print("ANSWER")
            print("-" * 70)
            print(answer)
            print()
            print(
                f"Generation time: "
                f"{elapsed:.2f} seconds"
            )

            results.append(
                {
                    "id": question_id,
                    "act": act,
                    "section": section,
                    "question": question,
                    "answer": answer,
                    "model": MODEL_NAME,
                    "system": (
                        "Open Australian Legal LLM "
                        "without retrieval"
                    ),
                    "generation_time_seconds": round(
                        elapsed,
                        2,
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
                    "model": MODEL_NAME,
                    "system": (
                        "Open Australian Legal LLM "
                        "without retrieval"
                    ),
                    "error": str(e),
                }
            )

        print()

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    successful = sum(
        1
        for result in results
        if result.get("answer")
    )

    failed = total - successful

    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(f"Total questions: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()

    print("Results saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()