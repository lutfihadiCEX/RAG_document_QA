# 🧠 RAG Document QA System



A Retrieval-Augmented Generation (RAG) system for document question-answering using local LLMs. 100% free, no API keys required!



---



## 🔎 Overview



This project demonstrates a complete RAG pipeline that allows users to:

-Upload documents (PDF, TXT, DOCX)

-Ask natural language questions about the content

-Get accurate answers with source citations

-Run everything locally with no cloud costs



Key Features:

-Local LLM execution (Llama 3.2, Mistral, Phi-3)

-Vector similarity search with FAISS

-Document chunking with context preservation

-Source citation and transparency

-Web UI with Streamlit



---



## ⚡ Quick Start & Setup

Prerequisites

-Python 3.10+

-Ollama installed

-8GB+ RAM recommended



Installation



1. Clone the repository

```bash

git clone https://github.com/lutfihadiCEX/RAG_document_QA.git

cd rag-document-qa

```

2. Create and activate the Conda environment

```bash

conda create -n rag python=3.10 -y
conda activate rag

```

3. Install dependencies

```bash

pip install -r requirements.txt

```



4. Install and start Ollama

```bash

# Install Ollama from https://ollama.ai



# Pull a model

ollama pull llama3.2



# Start Ollama service (keep terminal open)

ollama serve

```



5. Run the application

```bash

streamlit run ui/streamlit\_app.py

```

If you already have an environment set up (e.g. base), you can skip creating a new one — but using a dedicated environment avoids version conflicts.

---

## 👨‍💻 Usage

Via Web UI
1.Upload Documents: Click "Upload Documents" in sidebar
2.Process: Click "Process Documents" button
3.Ask Questions: Type questions in the chat interface
4.View Sources: Expand answers to see source citations

---

## ☰ Tech Stack

-LangChain - Document loading, chunking, and RAG pipeline

-FAISS - Vector similarity search for document retrieval

-Sentence-Transformers - Embedding generation

-Ollama - Local LLM runtime (Llama 3.2, Mistral, Phi-3)

-Streamlit - Front-end UI

---

## Acknowledgments

-LangChain for the RAG framework
-Ollama for local LLM inference
-Hugging Face for embeddings
-FAISS for vector search

---

## Author
Lutfihadi
For research and demonstration purposes as part of AI/ML portfolio projects

---

## Future improvements

- Add multi-document context memory  
- Integrate document summarization using LangChain chains  
- Dockerize for easier deployment  
- Extend support for other languages documents

