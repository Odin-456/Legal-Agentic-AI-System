import json
import requests
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from structure_retrieval import StructureAwareRetriever


# --------------------------------------------------
# Setup
# --------------------------------------------------

load_dotenv()

QUESTIONS_FILE = Path(
    "data/evaluation/qa_evaluation_questions.json"
)

GPT_RESULTS_FILE = Path(
    "data/evaluation/final_experiment_results.json"
)

OUTPUT_FILE = Path(
    "data/evaluation/llm_model_comparison_results.json"
)

EMBEDDING_MODEL = "text-embedding-3-large"

TOP_K = 5
STRUCTURE_CANDIDATES = 50

QWEN_MODEL = "qwen3:8b"
LLAMA_MODEL = "llama3.1:8b"

OLLAMA_URL = "http://localhost:11434/api/chat"


# --------------------------------------------------
# System prompt
# --------------------------------------------------

SYSTEM_PROMPT = """
You are a NSW property law question answering system.

Answer the user's question using ONLY the NSW property-law
legislation provided in the context.

Rules:
- Use only the supplied legislative context.
- Do not use outside legal knowledge.
- Do not invent Acts, sections, subsections, or legal rules.
- Identify the relevant Act and section when possible.
- If the supplied context is insufficient, clearly say so.
- Do not provide personalised legal advice.
- Answer concisely and accurately.
"""


# --------------------------------------------------
# Load evaluation questions
# --------------------------------------------------

with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
    dataset = json.load(file)

questions = dataset["questions"]

print(f"Loaded {len(questions)} evaluation questions.")


# --------------------------------------------------
# Load existing GPT-5-mini results
# --------------------------------------------------

with open(GPT_RESULTS_FILE, "r", encoding="utf-8") as file:
    gpt_results = json.load(file)

gpt_summary = gpt_results["summary"]["structure_aware"]


# --------------------------------------------------
# Structure-aware retriever
# --------------------------------------------------
#
# IMPORTANT:
# This is the ONLY retriever used in this experiment.
#
# It loads:
#
# data/vectorstore/property_law_structure
#
# It does NOT use:
# - property_law_baseline
# - query expansion
#
# Your structure_retrieval.py must also have
# query expansion removed.
# --------------------------------------------------

structure_retriever = StructureAwareRetriever(
    candidate_k=STRUCTURE_CANDIDATES
)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def normalize_text(value):
    if value is None:
        return ""

    return " ".join(str(value).lower().split())


def normalize_act(act):
    return normalize_text(act)


def normalize_section(section):
    return normalize_text(section)


def get_metadata(document):
    return document.metadata or {}


def get_page_content(document):
    return document.page_content or ""


def document_is_relevant(
    document,
    expected_act,
    expected_section,
):
    metadata = get_metadata(document)

    actual_act = normalize_act(
        metadata.get("act")
    )

    actual_section = normalize_section(
        metadata.get("section")
    )

    return (
        actual_act == normalize_act(expected_act)
        and
        actual_section == normalize_section(expected_section)
    )


# --------------------------------------------------
# Retrieve using ONLY structure-aware retrieval
# --------------------------------------------------

def retrieve_documents(question):
    results = structure_retriever.retrieve(
        question,
        top_k=TOP_K,
    )

    documents = []

    for result in results:

        document = result["document"]

        if document not in documents:
            documents.append(document)

        for related_document in result.get(
            "related_documents",
            []
        ):
            if related_document not in documents:
                documents.append(related_document)

    return documents


# --------------------------------------------------
# Build legal context
# --------------------------------------------------

def build_context(documents):
    context_parts = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        metadata = get_metadata(document)

        context_parts.append(
            f"""
[SOURCE {index}]
Act: {metadata.get("act", "")}
Section: {metadata.get("section", "")}
Heading: {metadata.get("heading", "")}
Subsection: {metadata.get("subsection", "")}
Paragraph: {metadata.get("paragraph", "")}
Nested paragraph: {metadata.get("nested_paragraph", "")}

Legal text:
{get_page_content(document)}
"""
        )

    return "\n".join(context_parts)


# --------------------------------------------------
# Ask local Ollama model
# --------------------------------------------------

def ask_ollama(
    model,
    question,
    context,
):

    user_prompt = f"""
Use the following NSW property-law legislative context
to answer the question.

CONTEXT:
{context}

QUESTION:
{question}
"""

    payload = {
        "model": model,

        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],

        "stream": False,

        "think": False,

        "options": {
            "temperature": 0,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=300,
    )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"]


# --------------------------------------------------
# GPT-5-mini evaluator
# --------------------------------------------------
#
# GPT-5-mini is ONLY used here for evaluating
# Qwen3 8B and Llama 3.1 8B answers.
# --------------------------------------------------

evaluator = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
)


def evaluate_answer(
    question,
    answer,
    documents,
    expected_act,
    expected_section,
):

    context = build_context(documents)

    evaluator_prompt = f"""
Evaluate the following NSW property-law question and answer.

QUESTION:
{question}

EXPECTED ACT:
{expected_act}

EXPECTED SECTION:
{expected_section}

RETRIEVED CONTEXT:
{context}

ANSWER:
{answer}

Evaluate the answer using ONLY the retrieved legal context.

Return JSON only with these fields:

{{
  "answer_correctness": 0,
  "groundedness": 0,
  "completeness": 0,
  "citation_accuracy": 0,
  "reasoning_score": 1,
  "answers_question": true,
  "reasoning_correct": true,
  "total_claims": 0,
  "unsupported_claims": 0,
  "explanation": ""
}}

Rules:

1. answer_correctness:
   Score 0-100 based on whether the answer is legally
   correct according to the supplied context.

2. groundedness:
   Score 0-100 based on whether the claims are supported
   by the retrieved legislative context.

3. completeness:
   Score 0-100 based on whether the answer addresses
   the important parts of the question.

4. citation_accuracy:
   Score 0-100 based on whether cited Acts, sections,
   and sources are accurate.

5. reasoning_score:
   Score from 1-5.

6. answers_question:
   True only if the answer actually answers the question.

7. reasoning_correct:
   Must be false if:
   - reasoning_score is below 4
   - answers_question is false
   - the expected Act and section were not retrieved.

8. total_claims:
   Count the substantive claims made in the answer.

9. unsupported_claims:
   Count substantive claims that cannot be supported
   by the retrieved legislative context.

10. A refusal saying "context is insufficient" is NOT
    correct if the retrieved context contains enough
    information to answer the question.

Return valid JSON only.
"""

    response = evaluator.invoke(
        evaluator_prompt
    )

    content = response.content

    if isinstance(content, list):

        content = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        )

    content = content.strip()

    if content.startswith("```"):

        content = content.replace(
            "```json",
            ""
        )

        content = content.replace(
            "```",
            ""
        )

        content = content.strip()

    return json.loads(content)


# --------------------------------------------------
# Run one local model
# --------------------------------------------------

def run_model(model_name):

    results = []

    for index, item in enumerate(
        questions,
        start=1,
    ):

        question_id = item["id"]

        question = item["question"]

        expected_act = item["act"]

        expected_section = item["section"]

        print(
            f"[{model_name}] "
            f"{index}/{len(questions)} "
            f"Question {question_id}"
        )

        # ------------------------------------------
        # Structure-aware retrieval ONLY
        # ------------------------------------------

        documents = retrieve_documents(
            question
        )

        # Only the first TOP_K documents are used
        # for retrieval evaluation.
        top_documents = documents[:TOP_K]

        retrieved_relevant = any(
            document_is_relevant(
                document,
                expected_act,
                expected_section,
            )
            for document in top_documents
        )

        # ------------------------------------------
        # Build context
        # ------------------------------------------

        context = build_context(
            top_documents
        )

        # ------------------------------------------
        # Generate answer with local model
        # ------------------------------------------

        answer = ask_ollama(
            model_name,
            question,
            context,
        )

        # ------------------------------------------
        # Evaluate answer using GPT-5-mini
        # ------------------------------------------

        evaluation = evaluate_answer(
            question,
            answer,
            top_documents,
            expected_act,
            expected_section,
        )

        # ------------------------------------------
        # Reasoning correctness
        # ------------------------------------------

        if not retrieved_relevant:
            evaluation["reasoning_correct"] = False

        results.append(
            {
                "id": question_id,
                "question": question,
                "expected_act": expected_act,
                "expected_section": expected_section,

                "retrieved_relevant": (
                    retrieved_relevant
                ),

                "answer": answer,

                "evaluation": evaluation,
            }
        )

    return results


# --------------------------------------------------
# Calculate retrieval + answer metrics
# --------------------------------------------------

def calculate_metrics(results):

    total = len(results)

    precision_at_1 = []
    precision_at_5 = []

    recall_at_1 = []
    recall_at_5 = []

    answer_correctness = []
    groundedness = []
    completeness = []
    citation_accuracy = []
    reasoning_accuracy = []

    hallucination_rates = []

    for result in results:

        expected_act = result[
            "expected_act"
        ]

        expected_section = result[
            "expected_section"
        ]

        # ------------------------------------------
        # Retrieval
        # ------------------------------------------

        documents = retrieve_documents(
            result["question"]
        )

        top_documents = documents[:TOP_K]

        relevant_count_at_1 = 0
        relevant_count_at_5 = 0

        if len(top_documents) >= 1:

            if document_is_relevant(
                top_documents[0],
                expected_act,
                expected_section,
            ):

                relevant_count_at_1 = 1

        for document in top_documents:

            if document_is_relevant(
                document,
                expected_act,
                expected_section,
            ):

                relevant_count_at_5 += 1

        # P@1
        if len(top_documents) > 0:

            precision_at_1.append(
                relevant_count_at_1
                / len(top_documents[:1])
            )

        # P@5
        if len(top_documents) > 0:

            precision_at_5.append(
                relevant_count_at_5
                / len(top_documents)
            )

        # R@1
        recall_at_1.append(
            1
            if relevant_count_at_1 > 0
            else 0
        )

        # R@5
        recall_at_5.append(
            1
            if relevant_count_at_5 > 0
            else 0
        )

        # ------------------------------------------
        # Answer evaluation
        # ------------------------------------------

        evaluation = result[
            "evaluation"
        ]

        answer_correctness.append(
            evaluation[
                "answer_correctness"
            ]
        )

        groundedness.append(
            evaluation[
                "groundedness"
            ]
        )

        completeness.append(
            evaluation[
                "completeness"
            ]
        )

        citation_accuracy.append(
            evaluation[
                "citation_accuracy"
            ]
        )

        reasoning_accuracy.append(
            1
            if evaluation[
                "reasoning_correct"
            ]
            else 0
        )

        # ------------------------------------------
        # Hallucination
        # ------------------------------------------

        total_claims = evaluation[
            "total_claims"
        ]

        unsupported_claims = evaluation[
            "unsupported_claims"
        ]

        if total_claims > 0:

            hallucination_rates.append(
                unsupported_claims
                / total_claims
            )

    return {

        "P@1": (
            sum(precision_at_1)
            / len(precision_at_1)
            if precision_at_1
            else 0
        ),

        "P@5": (
            sum(precision_at_5)
            / len(precision_at_5)
            if precision_at_5
            else 0
        ),

        "R@1": (
            sum(recall_at_1)
            / total
        ),

        "R@5": (
            sum(recall_at_5)
            / total
        ),

        "answer_correctness": (
            sum(answer_correctness)
            / len(answer_correctness)
        ),

        "groundedness": (
            sum(groundedness)
            / len(groundedness)
        ),

        "completeness": (
            sum(completeness)
            / len(completeness)
        ),

        "citation_accuracy": (
            sum(citation_accuracy)
            / len(citation_accuracy)
        ),

        "reasoning_accuracy": (
            sum(reasoning_accuracy)
            / len(reasoning_accuracy)
        ),

        "hallucination_rate": (
            sum(hallucination_rates)
            / len(hallucination_rates)
            if hallucination_rates
            else 0
        ),
    }


# --------------------------------------------------
# Run Qwen3 8B
# --------------------------------------------------

print("\n" + "=" * 70)
print("RUNNING QWEN3 8B")
print("=" * 70)

qwen_results = run_model(
    QWEN_MODEL
)


# --------------------------------------------------
# Run Llama 3.1 8B
# --------------------------------------------------

print("\n" + "=" * 70)
print("RUNNING LLAMA 3.1 8B")
print("=" * 70)

llama_results = run_model(
    LLAMA_MODEL
)


# --------------------------------------------------
# Calculate metrics
# --------------------------------------------------

print("\nCalculating Qwen3 8B metrics...")

qwen_metrics = calculate_metrics(
    qwen_results
)

print("Calculating Llama 3.1 8B metrics...")

llama_metrics = calculate_metrics(
    llama_results
)


# --------------------------------------------------
# Save results
# --------------------------------------------------

output = {

    "experiment": {

        "description": (
            "Comparison of GPT-5-mini, Qwen3 8B "
            "and Llama 3.1 8B using the same "
            "structure-aware retriever without "
            "query expansion."
        ),

        "embedding_model": (
            EMBEDDING_MODEL
        ),

        "vectorstore": (
            "data/vectorstore/property_law_structure"
        ),

        "retrieval": (
            "structure_aware"
        ),

        "query_expansion": False,

        "top_k": TOP_K,

        "structure_candidates": (
            STRUCTURE_CANDIDATES
        ),

        "question_count": len(questions),

        "evaluator": "gpt-5-mini",
    },

    "summary": {

        "GPT-5-mini": gpt_summary,

        "Qwen3-8B": qwen_metrics,

        "Llama-3.1-8B": llama_metrics,
    },

    "results": {

        "Qwen3-8B": qwen_results,

        "Llama-3.1-8B": llama_results,
    },
}


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        output,
        file,
        indent=2,
        ensure_ascii=False,
    )


# --------------------------------------------------
# Print final comparison
# --------------------------------------------------

print("\n")
print("=" * 70)
print("FINAL LLM MODEL COMPARISON")
print("=" * 70)

print("\nGPT-5-mini")
print(
    json.dumps(
        gpt_summary,
        indent=2,
    )
)

print("\nQwen3 8B")
print(
    json.dumps(
        qwen_metrics,
        indent=2,
    )
)

print("\nLlama 3.1 8B")
print(
    json.dumps(
        llama_metrics,
        indent=2,
    )
)

print("\n" + "=" * 70)

print(
    "Results saved to:"
)

print(
    OUTPUT_FILE
)