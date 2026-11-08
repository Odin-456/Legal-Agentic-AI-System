import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from property_law_parser import parse_property_law_corpus
load_dotenv()

OUTPUT_FILE = Path("data/evaluation/qa_evaluation_questions.json")
MODEL_NAME = "gpt-5-mini"

# Sections used in the evaluation dataset
TARGET_SECTIONS = {
    "Real Property Act 1900": [
        
        "1", "14", "32", "41", "42", "43",
        "45C", "45D", "46C", "74F", "118",
        "53", "56", "57", "58", "65",
        "74H", "74T", "93", "97"
    ],

    "Conveyancing Act 1919": [
    
        "23B", "38", "41", "54", "54A", "55",
        "66G", "73", "75", "109", "111", "113",
        "30", "52A", "66K", "66S",
        "66T", "88K", "89", "129"
    ],

    "Residential Tenancies Act 2010": [

        "21", "22", "24", "41", "51", "52",
        "55", "63", "64", "87", "100", "187",
        "50", "57", "73B", "82",
        "88", "105B", "115", "168"
    ],

    "Strata Schemes Management Act 2015": [

        "9", "10", "11", "30", "31", "32",
        "44", "50", "106", "135", "139", "154",
        "73", "74", "86", "108",
        "110", "137B", "160", "182"
    ],

    "Community Land Management Act 2021": [
  
        "5", "6", "8", "9", "25", "30",
        "31", "32", "34", "40", "50", "100",
        "53", "77", "78", "84",
        "91", "109", "127", "129A"
    ]
}

def load_target_sections():
    """Load the legislation and select the required sections."""
    sections = parse_property_law_corpus()
    selected = []
    for act in TARGET_SECTIONS:
        section_numbers = TARGET_SECTIONS[act]
        for section in sections:
            if section["act"] == act:
                if section["section"] in section_numbers:
                    selected.append(section)
    return selected


def generate_questions(llm, act, sections):
    """Generate one natural-language question for each section."""
    section_data = []
    for section in sections:
        data = {
            "section": section["section"],
            "heading": section["heading"],
            "text": section["text"],
        }
        section_data.append(data)
    prompt = f"""
Generate one natural-language question for each section below.

Act: {act}

Requirements:
- Generate exactly one question for each section.
- Base each question on the supplied legal text.
- Do not mention the section number.
- Do not copy the heading directly.
- Make the questions sound like questions from a real person.
- Do not invent information.
- Return only JSON.

Use this format:

[
    {{
        "section": "section number",
        "question": "question?"
    }}
]

Sections:

{json.dumps(section_data, indent=2)}
"""
    response = llm.invoke(prompt)

    questions = json.loads(response.content)

    return questions


def main():

    # Load the selected legal sections
    sections = load_target_sections()
    print(f"Selected sections: {len(sections)}")

    # Create the question-generation model
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.2,
    )

    questions = []

    # Generate questions for each Act
    for act in TARGET_SECTIONS:
        act_sections = []
        for section in sections:
            if section["act"] == act:
                act_sections.append(section)
        generated_questions = generate_questions(
            llm,
            act,
            act_sections,
        )
        for question in generated_questions:
            new_question = {
                "id": len(questions) + 1,
                "act": act,
                "section": question["section"],
                "question": question["question"],
            }
            questions.append(new_question)

    # Save the evaluation dataset
    dataset = {
        "question_count": len(questions),
        "questions": questions,
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
            dataset,
            file,
            indent=2,
            ensure_ascii=False,
        )
        
    print(f"Created {len(questions)} questions.")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()