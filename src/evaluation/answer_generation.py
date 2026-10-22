"""Context construction and answer generation for GPT and local Ollama models."""
import requests

from experiment_config import (
    OLLAMA_TIMEOUT,
    OLLAMA_URL,
    SYSTEM_PROMPT,
    answer_llm,
)


def get_metadata(document):
    if hasattr(document, "metadata"):
        return document.metadata or {}
    if isinstance(document, dict):
        return document.get("metadata", document) or {}
    return {}


def get_page_content(document):
    if hasattr(document, "page_content"):
        return document.page_content or ""
    if isinstance(document, dict):
        return document.get("page_content", document.get("text", "")) or ""
    return str(document)


def build_context(documents):
    context_parts = []
    for index, document in enumerate(documents, start=1):
        metadata = get_metadata(document)
        act = metadata.get("act", "")
        section = metadata.get("section", "")
        heading = metadata.get("heading", "")
        subsection = metadata.get("subsection", "")
        paragraph = metadata.get("paragraph", "")
        nested_paragraph = metadata.get("nested_paragraph", "")
        text = get_page_content(document)
        context_parts.append(
            f"""
[SOURCE {index}]
Act: {act}
Section: {section}
Heading: {heading}
Subsection: {subsection}
Paragraph: {paragraph}
Nested paragraph: {nested_paragraph}

Legal text:
{text}
"""
        )
    return "\n".join(context_parts)


def generate_answer(question, documents):
    context = build_context(documents)
    prompt = f"""
{SYSTEM_PROMPT}

QUESTION:
{question}

RETRIEVED LEGAL CONTEXT:
{context}

Provide a concise and accurate answer.
"""
    response = answer_llm.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    return str(content).strip()


def generate_local_answer(model_name, question, documents):
    context = build_context(documents)
    user_prompt = f"""
Use the following NSW property-law legislative context
to answer the question.

CONTEXT:
{context}

QUESTION:
{question}

Answer concisely and accurately using only the context.
"""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    content = data.get("message", {}).get("content", "")
    if not content:
        raise ValueError(f"Ollama returned an empty answer for {model_name}.")
    return content.strip()
