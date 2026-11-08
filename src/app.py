import sys
from pathlib import Path
import markdown
from flask import Flask, render_template, request

SRC_DIR = Path(__file__).resolve().parent

from legal_agent import agent

app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def home():
    question = ""
    answer = ""
    error = ""

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if not question:
            error = "Please enter a legal question."
        else:
            try:
                result = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": question,
                            }
                        ]
                    }
                )

                answer = result["messages"][-1].content

                answer = markdown.markdown(
                    answer,
                    extensions=["extra", "nl2br"]
                )

            except Exception as e:
                error = f"An error occurred: {e}"

    return render_template(
        "index.html",
        question=question,
        answer=answer,
        error=error,
    )


if __name__ == "__main__":

    print("NSW PROPERTY-LAW AGENTIC AI SYSTEM")
    print()
    print("Web interface starting...")
    print("Open http://127.0.0.1:5000 in your browser.")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )