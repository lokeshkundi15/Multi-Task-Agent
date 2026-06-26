# Multi-Source Autonomous Agentic RAG Chatbot

[cite_start]An advanced, industry-standard Multi-Task GenAI application built with **LangChain**, **MongoDB Atlas Vector Search**, and **FAISS**[cite: 1, 130]. This autonomous agent intelligently routes user queries across private documents (PDFs) and local vector documentation, with a seamless fallback to the LLM's internal knowledge.

---

## 🚀 Architecture & Key Features
- **Intelligent Routing**: Automatically evaluates user intent and pulls data from MongoDB Vector Search first, falls back to FAISS, or handles it using core LLM capabilities.
- **Production Fallback Loop**: Ensures the system never halts or answers generic text when private data is missing, providing seamless answers via a prioritized cascade logic.
- **Multi-Source Knowledge**:
  - **Private Knowledge Base**: Seeds and queries cloud-managed MongoDB Atlas using vector search.
  - **Local Documentation**: Uses FAISS in-memory store to manage and search auxiliary documentation.

---

## 🛠️ Tech Stack
- **Framework**: LangChain, LangChain Core
- **LLM Engine**: Google Gemini-2.5-Flash (via OpenRouter API)
- **Vector Stores**: MongoDB Atlas Vector Search, FAISS
- **Embeddings Model**: OpenAI Text-Embedding-3-Small (via OpenRouter)
- **Data Parser**: PyPDFLoader (Local PDF ingestion)

---

## 💻 Environment Setup

### 1. Create a Virtual Environment
**MacOS/Linux**:
```bash
python3 -m venv env

Windows:

Bash
python -m venv env


2. Activate the Virtual Environment
MacOS/Linux:

Bash
source env/bin/activate
Windows:

Bash
.\env\Scripts\activate
3. Install Dependencies

Bash
pip install -r requirements.txt

Environment Configuration
Create a .env file in the root directory and configure the following parameters:

Code snippet

OPENAI_API_KEY="your_openrouter_api_key_here"
DB_NAME="elearrning"
COLLECTION_NAME="monodb-training"
ATLAS_VECTOR_SEARCH_INDEX_NAME="vector_index"
ATLAS_CONNECTION_STRING="your_mongodb_atlas_connection_string_here"

Note: Make sure .env is added to your .gitignore file before pushing this repository to GitHub to prevent exposing your API keys.

Execution Flow

Step 1: Ingest and Index the Document
Run the following command to load your local document (AI_Job_Roadmap.pdf), slice it into optimized text chunks, and generate vector embeddings directly inside your MongoDB Atlas cluster:

Bash
python retrievers/vector_search_index.py

Step 2: Launch the Agentic CLI Dashboard
Once indexing is ready, launch the core autonomous agent chatbot environment:

Bash
python main.py

Usage
Press 1 to interact with the AI Agent.

Ask questions regarding your private PDF data or vector search architecture concepts.

Type x at any time to break the loop and return back to the main interactive options menu.

Repository Directory Hierarchy
Plaintext
.
├── .env
├── .gitignore
├── requirements.txt
├── README.md
├── main.py
├── agent.py
├── AI_Job_Roadmap.pdf
└── retrievers/
    ├── vector_search_index.py
    └── web_search.py