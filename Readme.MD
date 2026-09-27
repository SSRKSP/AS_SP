# Document Q&A with RAG
## Why RAG?

A normal LLM may not know the contents of a user's uploaded document.
RAG first retrieves relevant information from the document and then
provides that information to the LLM as context.

We chose RAG because it helps the system generate answers grounded in
the uploaded document and allows us to provide page-level sources.

Our verification layer further checks whether the generated answer is
supported by the retrieved content.
#### Dataset / Sample Document

The sample document used for testing was created by our team and is
included in this repository under `sample dataset`.

## Demo Video

▶ Watch the project demonstration in https://drive.google.com/drive/folders/1V-LgBlj_j9BigLq_dQtKqt3WHJBlV_5p

## How it works?

Upload a PDF, ask questions about it, and get answers grounded in the document — with
page-level sources and a built-in verification layer (confidence rating + auto-answered
follow-up questions).



1. **Upload** — your PDF is loaded and split into overlapping text chunks (so no sentence
   gets cut off between chunks).
2. **Embed** — each chunk is converted into a vector (a list of numbers representing its
   meaning) using a free, local embedding model (`sentence-transformers`). No API key
   needed for this step.
3. **Store** — all vectors are saved in a local vector database (Chroma). This indexing
   happens once per document.
4. **Ask** — your question is also embedded, and the vector database finds the most
   relevant chunks by comparing vectors (fast — no LLM involved in this matching step).
5. **Generate** — those chunks + your question are sent to an LLM hosted by
   [Groq](https://groq.com) (e.g. Llama 3.3), which writes the final answer.
6. **Verify** — a second pass rates how well-supported the answer is and proactively
   answers likely follow-up questions, grounded only in the retrieved text.

Steps 1–3 happen once per document. Step 4 onward is fast because the heavy lifting
(indexing) is already done.

## Setup

```bash
git clone <your-repo-url>
cd rag-doc-qa

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt

cp .env.example .env         # then paste your real Groq API key into .env
```

Get a free Groq API key at https://console.groq.com 

## Run

```bash
streamlit run app.py
```

This opens the app in your browser (usually `http://localhost:8501`). Upload a PDF,
click "Process document," then ask questions.

## Project structure

```
rag-doc-qa/
├── app.py              # Streamlit app: upload, RAG pipeline, verification layer
├── requirements.txt    # Python dependencies
├── .env.example        # Template for your API key (copy to .env, never commit .env)
└── .gitignore          # Keeps secrets and generated files out of GitHub
```

## Pushing to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Document Q&A RAG app"
git branch -M main
git remote add origin <your-empty-github-repo-url>
git push -u origin main
```

Because `.env` is in `.gitignore`, your API key never gets uploaded. Anyone who clones
the repo just needs to create their own `.env` from `.env.example` and run the two
commands above — the app works immediately.
