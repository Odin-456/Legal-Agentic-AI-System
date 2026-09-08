import json
import os
import re


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_FILE = "data/evaluation/chatgpt_baseline_results.json"

OUTPUT_FILE = "data/evaluation/chatgpt_baseline_evaluation.json"


# ============================================================
# NORMALISE TEXT
# ============================================================

def normalise(text):
    """
    Normalise text for reliable matching.
    """

    if text is None:
        return ""

    text = str(text).lower()

    text = text.replace("–", "-")
    text = text.replace("—", "-")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# ACT MATCHING
# ============================================================

def act_matches(answer, expected_act):
    """
    Check whether the answer explicitly mentions
    the expected Act.
    """

    answer_normalised = normalise(answer)
    act_normalised = normalise(expected_act)

    return act_normalised in answer_normalised


# ============================================================
# SECTION MATCHING
# ============================================================

def section_matches(answer, expected_section):
    """
    Check for legal references such as:

        s 42
        s. 42
        section 42
        section 42(1)
        sec 42
    """

    answer = normalise(answer)

    section = str(expected_section).strip()

    patterns = [
        rf"\bs\.?\s*{re.escape(section)}\b",
        rf"\bsection\s+{re.escape(section)}\b",
        rf"\bsec\.?\s*{re.escape(section)}\b",
    ]

    for pattern in patterns:

        if re.search(pattern, answer):
            return True

    return False


# ============================================================
# ANSWER PRESENCE
# ============================================================

def answer_present(answer):

    if not answer:
        return False

    return len(answer.strip()) > 0


# ============================================================
# EVALUATE ONE RESULT
# ============================================================

def evaluate_result(result):

    answer = result.get("answer", "")

    expected_act = result.get("act", "")

    expected_section = result.get("section", "")

    act_correct = act_matches(
        answer,
        expected_act
    )

    section_correct = section_matches(
        answer,
        expected_section
    )

    act_and_section_correct = (
        act_correct and section_correct
    )

    return {
        "id": result.get("id"),
        "act": expected_act,
        "section": expected_section,
        "question": result.get("question"),
        "answer": answer,
        "act_correct": act_correct,
        "section_correct": section_correct,
        "act_and_section_correct": act_and_section_correct,
        "answer_present": answer_present(answer),
    }


# ============================================================
# CALCULATE OVERALL METRICS
# ============================================================

def calculate_metrics(evaluated):

    total = len(evaluated)

    if total == 0:
        return {}

    act_correct = sum(
        item["act_correct"]
        for item in evaluated
    )

    section_correct = sum(
        item["section_correct"]
        for item in evaluated
    )

    both_correct = sum(
        item["act_and_section_correct"]
        for item in evaluated
    )

    answers_present = sum(
        item["answer_present"]
        for item in evaluated
    )

    return {
        "total_questions": total,

        "act_accuracy": round(
            act_correct / total,
            4
        ),

        "section_accuracy": round(
            section_correct / total,
            4
        ),

        "act_and_section_accuracy": round(
            both_correct / total,
            4
        ),

        "answer_generation_rate": round(
            answers_present / total,
            4
        ),

        "act_correct_count": act_correct,

        "section_correct_count": section_correct,

        "act_and_section_correct_count": both_correct,

        "answers_present_count": answers_present,
    }


# ============================================================
# CALCULATE RESULTS BY ACT
# ============================================================

def calculate_act_metrics(evaluated):

    acts = {}

    for item in evaluated:

        act = item["act"]

        if act not in acts:
            acts[act] = []

        acts[act].append(item)

    act_metrics = {}

    for act, items in acts.items():

        total = len(items)

        act_correct = sum(
            item["act_correct"]
            for item in items
        )

        section_correct = sum(
            item["section_correct"]
            for item in items
        )

        both_correct = sum(
            item["act_and_section_correct"]
            for item in items
        )

        act_metrics[act] = {
            "questions": total,

            "act_accuracy": round(
                act_correct / total,
                4
            ),

            "section_accuracy": round(
                section_correct / total,
                4
            ),

            "act_and_section_accuracy": round(
                both_correct / total,
                4
            ),
        }

    return act_metrics


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    metrics,
    act_metrics,
    evaluated
):

    print()
    print("=" * 70)
    print("GPT-5-MINI BASELINE EVALUATION RESULTS")
    print("=" * 70)

    print()

    print("OVERALL")
    print("-" * 70)

    print(
        f"Total questions: "
        f"{metrics['total_questions']}"
    )

    print(
        f"Act accuracy: "
        f"{metrics['act_accuracy']:.2%}"
    )

    print(
        f"Section accuracy: "
        f"{metrics['section_accuracy']:.2%}"
    )

    print(
        f"Act + section accuracy: "
        f"{metrics['act_and_section_accuracy']:.2%}"
    )

    print(
        f"Answer generation rate: "
        f"{metrics['answer_generation_rate']:.2%}"
    )

    print()

    print("RESULTS BY ACT")
    print("-" * 70)

    for act, values in act_metrics.items():

        print()
        print(act)

        print(
            f"  Questions: "
            f"{values['questions']}"
        )

        print(
            f"  Act accuracy: "
            f"{values['act_accuracy']:.2%}"
        )

        print(
            f"  Section accuracy: "
            f"{values['section_accuracy']:.2%}"
        )

        print(
            f"  Act + section accuracy: "
            f"{values['act_and_section_accuracy']:.2%}"
        )

    # --------------------------------------------------------
    # Incorrect results
    # --------------------------------------------------------

    incorrect = [
        item
        for item in evaluated
        if not item["act_and_section_correct"]
    ]

    print()
    print("INCORRECT ACT + SECTION RESULTS")
    print("-" * 70)

    print(
        f"Incorrect: "
        f"{len(incorrect)}/{len(evaluated)}"
    )

    print()

    for item in incorrect:

        print(
            f"Q{item['id']} | "
            f"{item['act']} | "
            f"s {item['section']}"
        )

        print(
            f"Question: "
            f"{item['question']}"
        )

        print(
            f"Act correct: "
            f"{item['act_correct']}"
        )

        print(
            f"Section correct: "
            f"{item['section_correct']}"
        )

        print()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EVALUATING GPT-5-MINI BASELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load results
    # --------------------------------------------------------

    print()
    print("Loading results...")

    if not os.path.exists(RESULTS_FILE):

        print()
        print("ERROR:")
        print(
            f"Could not find: {RESULTS_FILE}"
        )

        return

    with open(
        RESULTS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        results = json.load(f)

    print(
        f"Results loaded: {len(results)}"
    )

    if len(results) != 59:

        print()
        print(
            "WARNING: Expected 59 results."
        )

        print(
            f"Found: {len(results)}"
        )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    evaluated = []

    for result in results:

        evaluated_result = evaluate_result(
            result
        )

        evaluated.append(
            evaluated_result
        )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        evaluated
    )

    act_metrics = calculate_act_metrics(
        evaluated
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_results(
        metrics,
        act_metrics,
        evaluated
    )

    # --------------------------------------------------------
    # Save evaluation
    # --------------------------------------------------------

    output = {

        "model": "gpt-5-mini",

        "evaluation_type": (
            "Act and section identification"
        ),

        "system": (
            "GPT-5-mini without retrieval"
        ),

        "metrics": metrics,

        "results_by_act": act_metrics,

        "questions": evaluated,
    }

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("=" * 70)
    print("EVALUATION SAVED")
    print("=" * 70)

    print()
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()

