# Legal Document Retrieval and Question Answering

This repository contains a lightweight Agentic AI system for retrieving information from NSW property legislation and answering legal questions. It uses XML parsing, hierarchical document chunking, Chroma vector databases, structure-aware retrieval and a LangChain agent powered by GPT-5-mini.

## 1. Project Configuration

The project uses Python 3.11.4 and organises its source code, legislation data, vector databases and evaluation results under the repository root.

The main Python scripts are stored in `src/`. Dataset files and generated outputs are stored in `data/`.

The expected directory structure is:

```text
Legal-Agentic-AI-System/
|-- .env
|-- .gitignore
|-- .python-version
|-- README.md
|-- requirements.txt
|
|-- data/
|   |-- raw/
|   |   `-- australian_property_law/
|   |       `-- *.xml
|   |
|   |-- vectorstore/
|   |   |-- property_law_baseline/
|   |   `-- property_law_structure/
|   |
|   |-- evaluation/
|   |   |-- qa_evaluation_questions.json
|   |   `-- final_experiment_results.json
|   |
|   `-- exploration_figures/
|       |-- dataset_exploration_summary.csv
|       `-- figure_*.png
|
`-- src/
    |-- property_law_parser.py
    |-- property_law_chunker.py
    |-- structure_aware_chunker.py
    |-- build_vectorstore.py
    |-- structure_retrieval.py
    |-- legal_agent.py
    |-- app.py
    |-- data_exploration.py
    |-- create_qa_evaluation_questions.py
    |-- final_experiment_evaluation.py
    `-- templates/
        `-- index.html
```

The raw XML legislation files must be placed inside `data/raw/australian_property_law/` before running the preprocessing and vector database construction scripts. The vector database folders are generated during the build process.

Install the dependencies using the repository's `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The system requires an OpenAI API key for text embeddings and GPT-5-mini. Create a `.env` file in the repository root:

```env
OPENAI_API_KEY=your_openai_api_key
```

## 2. Legislation Preprocessing

The system uses five NSW property Acts:

- Real Property Act 1900
- Conveyancing Act 1919
- Residential Tenancies Act 2010
- Strata Schemes Management Act 2015
- Community Land Management Act 2021

The preprocessing pipeline converts the original XML legislation into documents suitable for semantic search and structure-aware retrieval.

### XML Legislation Parsing

**Script:** `src/property_law_parser.py`

**Input:** Original NSW property legislation XML files stored in `data/raw/australian_property_law/`.

**Processing:**
- Reads the XML files using Python's `xml.etree.ElementTree`.
- Extracts legislative text, section numbers and headings.
- Cleans whitespace and processes nested legislative lists.
- Handles repealed provisions during parsing.
- Organises the extracted content into structured legislative documents.

**Output:** Parsed legislation documents containing the text and section information required by the chunking scripts.

### Baseline Document Chunking

**Script:** `src/property_law_chunker.py`

**Input:** Parsed legislation documents.

**Processing:**
- Creates one LangChain Document for each legislative section.
- Adds the Act name, section number and heading to the document content.
- Attaches metadata identifying the Act, section, heading and source type.

**Output:** Section-level documents used as the baseline RAG retrieval dataset.

### Structure-Aware Document Chunking

**Script:** `src/structure_aware_chunker.py`

**Input:** Parsed NSW property legislation.

**Processing:**
- Divides legislation according to its original legal hierarchy.
- Creates documents at section, subsection, paragraph and nested-paragraph levels where applicable.
- Records hierarchical metadata, including the Act, section, heading, subsection and paragraph references.
- Identifies repealed provisions.
- Removes duplicate documents based on their hierarchy metadata and content.

**Output:** Fine-grained legislative documents that preserve the relationships between sections and their lower-level provisions.

## 3. Vector Database Construction and Retrieval

### Vector Database Construction

**Script:** `src/build_vectorstore.py`

The script builds two separate Chroma vector databases to support the comparison between baseline RAG and structure-aware retrieval.

**Input:** Baseline section-level documents and structure-aware legislative documents generated from the preprocessing pipeline.

**Processing:**
- Generates document embeddings using OpenAI's `text-embedding-3-large` model.
- Creates a baseline collection named `property_law_baseline`.
- Creates a structure-aware collection named `property_law_structure`.
- Saves both collections locally using Chroma.

Run the script from the repository root:

```bash
python src/build_vectorstore.py
```

**Output:**

```text
data/
`-- vectorstore/
    |-- property_law_baseline/
    `-- property_law_structure/
```

The two vector databases are used separately by the baseline retrieval and structure-aware retrieval methods.

**Important:** The build script deletes the existing vector database directories before rebuilding them. Run it only when you intend to regenerate both databases.

### Structure-Aware Retrieval

**Script:** `src/structure_retrieval.py`

**Input:** A user's legal question and the structure-aware Chroma vector database.

**Processing:**

The `StructureAwareRetriever` combines semantic similarity with legal document structure to rank relevant legislative provisions.

1. Identifies relevant legal terms and corrects selected query terms.
2. Extracts explicit references to Acts, sections and subsections from the question.
3. Retrieves an initial set of 50 candidate documents using semantic similarity.
4. Calculates a structure score using matching text, headings, Acts, sections, subsections and paragraph metadata.
5. Combines the semantic and structure scores using a 60% semantic and 40% structure weighting.
6. Selects the top five unique Act-section combinations.
7. Retrieves up to 10 related chunks from the selected sections and orders them according to their legal hierarchy.

The structure score uses the following weights:

| Matching component | Weight |
|---|---:|
| Text | 0.50 |
| Heading | 0.22 |
| Act | 0.10 |
| Section | 0.08 |
| Subsection | 0.05 |
| Paragraph | 0.03 |
| Nested paragraph | 0.02 |

**Output:** Ranked legislative provisions and related chunks that provide additional context for answering the user's question.

## 4. Agentic AI System and Web Application

### Legal Question Answering Agent

**Script:** `src/legal_agent.py`

**Input:** A user's legal question and the legislative context returned by the structure-aware retriever.

**Processing:**
- Loads the OpenAI API key from the `.env` file.
- Initialises GPT-5-mini using the LangChain framework with a temperature of 0.
- Defines the `search_property_law(query)` tool to connect the agent with the structure-aware retriever.
- Uses LangChain's `create_agent()` to create a single-agent architecture.
- Formats retrieved provisions with their Act names and legislative references.
- Uses a system prompt to restrict answers to the five NSW property Acts included in the corpus.

The agent is instructed not to introduce external legal knowledge or invent legislative references. If the retrieved context is insufficient, it must state that the available information is insufficient. The system provides legal information rather than personalised legal advice.

**Output:** Concise, context-grounded answers with references to the retrieved legislative provisions.

### Flask Web Application

**Script:** `src/app.py`

**Input:** A legal question submitted through the web interface.

**Processing:**
- Receives the question through a Flask POST request.
- Passes the question to the LangChain agent.
- Retrieves the agent's response.
- Converts the Markdown response into HTML.
- Displays the answer using the `src/templates/index.html` template.

**Output:** A browser-based interface for submitting NSW property-law questions and viewing the generated answers.

Start the application from the repository root:

```bash
python src/app.py
```

Open the application in a browser:

```text
http://127.0.0.1:5000
```

## 5. Evaluation Experiments

The evaluation pipeline compares baseline retrieval and structure-aware retrieval, as well as different language models using the same structure-aware retrieval context.

### Evaluation Question Dataset

**Script:** `src/create_qa_evaluation_questions.py`

The evaluation question dataset is stored in:

```text
data/evaluation/qa_evaluation_questions.json
```

This dataset is used as the input for the final experiment evaluation.

### Experiment Execution

**Script:** `src/final_experiment_evaluation.py`

**Input:**
- `data/evaluation/qa_evaluation_questions.json`
- Baseline Chroma vector database.
- Structure-aware Chroma vector database.
- OpenAI API access.
- Ollama with Qwen3 8B and Llama 3.1 8B available locally.

**Processing:**

Four experiments are conducted:

| Experiment | Retrieval method | Language model |
|---|---|---|
| 1 | Baseline RAG | GPT-5-mini |
| 2 | Structure-aware RAG | GPT-5-mini |
| 3 | Structure-aware RAG | Qwen3 8B |
| 4 | Structure-aware RAG | Llama 3.1 8B |

The three language-model experiments using structure-aware retrieval share the same retrieved context, allowing the language models to be compared under the same retrieval conditions.

The evaluation measures:

- **Retrieval performance:** Precision@1, Precision@5, Recall@1, Recall@5 and Hit@5.
- **Answer quality:** Correctness, groundedness and completeness.
- **Citation accuracy:** Whether the cited legislative sources match the retrieved provisions.
- **Reasoning accuracy:** Whether the answer correctly addresses the question and retrieves the target section, using a reasoning score threshold of 4 out of 5.
- **Hallucination rate:** The proportion of substantive answer claims that are unsupported by the retrieved context.

GPT-5-mini is used as the answer model for the GPT experiment and as the evaluation model for assessing the generated answers.

Run the evaluation from the repository root:

```bash
python src/final_experiment_evaluation.py
```

For the local language-model experiments, install and start Ollama, then download the required models:

```bash
ollama pull qwen3:8b
ollama pull llama3.1:8b
```

**Output:** Experiment results are saved in:

```text
data/evaluation/final_experiment_results.json
```

The evaluation script also saves progress after each question so that intermediate results are retained during execution.

## 6. Notes

- The original XML legislation files must be supplied in `data/raw/australian_property_law/`.
- Run commands from the repository root so that the relative file paths work correctly.
- Build the Chroma vector databases before running the application or final evaluation.
- An OpenAI API key is required for embeddings, GPT-5-mini and the evaluation process.
- Ollama and the required local models are needed for the Qwen3 8B and Llama 3.1 8B experiments.
- Rebuilding the vector databases removes the existing generated database directories.
- Do not commit `.env`, API keys or other private credentials.