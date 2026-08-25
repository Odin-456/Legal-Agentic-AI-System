# Development of an Agentic AI System for Legal Document Retrieval and Question Answering

## Overview

This project focuses on developing a lightweight Agentic AI system for legal document retrieval and question answering.

The system combines Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and the LangChain framework to improve the accuracy and reliability of legal information retrieval.

The project aims to develop a single-agent architecture capable of retrieving relevant legal information, analysing legal documents, and generating contextually grounded responses.

---

## Research Objectives

This project investigates:

1. How an agentic AI system can be designed using LangChain to support legal document retrieval and question answering.

2. How structure-aware retrieval can improve accuracy, contextual understanding, and factual grounding in legal AI systems.

---

## Project Structure
├── .venv/
├── .env
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│
├── notebooks/
│
├── results/
│
└── src/
    ├── ingestion.py
    ├── parsing.py
    ├── embeddings.py
    ├── retrieval.py
    ├── rag.py
    ├── agent.py
    └── evaluation.py

---

## Technologies

The project uses:

- Python
- LangChain
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- Vector Databases
- Hugging Face Models
- Git/GitHub

---

## Environment Setup

### 1. Create virtual environment

```bash
python3 -m venv .venv