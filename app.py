import os
import tempfile
import uuid

import docx2txt
import pytesseract
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageOps
from pdf2image import convert_from_path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

load_dotenv()

QA_PROMPT = PromptTemplate(
    template="""You must answer using ONLY the information in the context below.
Do not use any outside knowledge, and do not fill in gaps from what you already know
about the topic — even if you're confident it's correct.
If the context has only partial or fragmentary information related to the question,
use what is there and clearly say what's missing or unclear — do not refuse just
because it's incomplete.
Only reply exactly "The document does not contain this information." if the context
has nothing at all relevant to the question.
If your answer includes any mathematical notation, write it using $ for inline math
and $$ for a math expression on its own line (standard Markdown-LaTeX style) —
never use \\[ \\] or \\( \\).

Context:
{context}

Question: {question}
Answer (using only the context above):""",
    input_variables=["context", "question"],
)

# On Windows, pytesseract and pdf2image need to be told exactly where these
# programs were installed. Update these paths if you installed them elsewhere.
_TESSERACT_PATH = r"C:\rag-doc-qa\tesseract.exe"
if os.path.exists(_TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_PATH

_POPPLER_PATH = r"C:\poppler\poppler-26.07.0\Library\bin"

st.set_page_config(page_title="Document Q&A (RAG)", layout="wide")
st.title("📄 Document Q&A with RAG")
st.caption("Upload a PDF, ask questions, get answers grounded in the document with sources.")


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    # Runs locally, no API key needed. Converts text into vectors (lists of numbers).
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it to a .env file (see .env.example).")
        st.stop()
    return ChatGroq(groq_api_key=api_key, model_name="openai/gpt-oss-120b")


def preprocess_for_ocr(image):
    """Improve OCR accuracy on typed/printed text: grayscale, boost contrast, and
    upscale ONLY if the image is small (upscaling large images just wastes time).
    This does NOT help with handwriting — only printed/typed text."""
    image = image.convert("L")  # grayscale
    width, height = image.size
    if min(width, height) < 1000:  # small/low-res — upscaling actually helps here
        image = image.resize((width * 2, height * 2), Image.LANCZOS)
    image = ImageOps.autocontrast(image)  # boost contrast
    return image


def load_any_file(uploaded_file, tmp_path):
    """Extract text from a PDF, DOCX, or image file, and return it as LangChain Documents."""
    suffix = uploaded_file.name.lower().split(".")[-1]

    if suffix == "pdf":
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        # If a PDF has no real extractable text, it's likely scanned/image-based.
        # Fall back to OCR: turn each page into an image, then read the text off it.
        if not any(doc.page_content.strip() for doc in documents):
            page_images = convert_from_path(tmp_path, poppler_path=_POPPLER_PATH)
            documents = []
            for i, page_image in enumerate(page_images):
                text = pytesseract.image_to_string(preprocess_for_ocr(page_image), config="--psm 6")
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"page": str(i + 1), "source": uploaded_file.name},
                    )
                )
        else:
            for doc in documents:
                doc.metadata["source"] = uploaded_file.name
        return documents

    elif suffix == "docx":
        text = docx2txt.process(tmp_path)
        return [Document(page_content=text, metadata={"page": "1", "source": uploaded_file.name})]

    elif suffix in ("txt", "md"):
        with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return [Document(page_content=text, metadata={"page": "1", "source": uploaded_file.name})]

    elif suffix in ("png", "jpg", "jpeg"):
        image = Image.open(tmp_path)
        text = pytesseract.image_to_string(preprocess_for_ocr(image), config="--psm 6")
        return [Document(page_content=text, metadata={"page": "1", "source": uploaded_file.name})]

    else:
        raise ValueError(f"Unsupported file type: .{suffix}")


def process_documents(uploaded_files):
    """Read multiple uploaded files, split each into overlapping chunks, and embed
    everything together into one shared vector store so questions can pull from any of them."""
    all_chunks = []
    debug_info = []  # per-file: how much text we actually managed to extract
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for uploaded_file in uploaded_files:
        suffix = "." + uploaded_file.name.lower().split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        documents = load_any_file(uploaded_file, tmp_path)
        char_count = sum(len(doc.page_content.strip()) for doc in documents)
        debug_info.append({"name": uploaded_file.name, "chars_extracted": char_count})

        chunks = splitter.split_documents(documents)
        all_chunks.extend(chunks)

    embedding_model = get_embedding_model()
    # A fresh, unique collection name every time guarantees this vectorstore only ever
    # contains what was just uploaded — no leftover data from a previous run can leak in.
    vectorstore = Chroma.from_documents(
        all_chunks, embedding_model, collection_name=f"session_{uuid.uuid4().hex}"
    )
    return vectorstore, len(all_chunks), debug_info


def normalize_math(text):
    """Convert \\[ \\] and \\( \\) style LaTeX delimiters into $ / $$ so Streamlit renders them."""
    text = text.replace("\\[", "$$").replace("\\]", "$$")
    text = text.replace("\\(", "$").replace("\\)", "$")
    return text


def summarize_documents(llm, vectorstore, max_chunks=15):
    """Answers 'what does this contain' style questions, which plain similarity search
    handles poorly since there's no single chunk that matches a vague, whole-document question."""
    raw = vectorstore.get(limit=max_chunks)
    texts = raw.get("documents", [])
    combined = "\n\n".join(texts)
    prompt = f"""Summarize what the following document content covers — the main topics,
sections, and what a reader would learn from it. Base this only on the text below.

Content:
{combined}

Summary:"""
    return llm.invoke(prompt).content


def verify_answer(llm, query, answer, source_docs):
    """The novel feature: rate confidence + proactively answer likely follow-up questions,
    grounded only in the retrieved source text."""
    context = "\n\n".join(doc.page_content for doc in source_docs)
    prompt = f"""You are reviewing an AI-generated answer for accuracy and completeness.

Question: {query}
Answer given: {answer}

Source text (the only thing you may use):
{context}

Do two things:
1. Rate CONFIDENCE as one of: "Directly stated in source", "Inferred from source", or "Not clearly supported".
2. List 2-3 natural follow-up questions a curious reader would likely ask next, and answer
   each one using ONLY the source text above. If the source doesn't cover a follow-up, say so.

If you include any mathematical notation, write it using $ for inline math and $$ for a
math expression on its own line — never use \\[ \\] or \\( \\).

Format your reply with clear headers: CONFIDENCE, then FOLLOW-UP QUESTIONS."""
    return llm.invoke(prompt).content


# --- Session state setup ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "memory" not in st.session_state:
    st.session_state.memory = None
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []  # list of dicts: question, answer, sources, verification

# --- Sidebar: upload & process ---
with st.sidebar:
    st.header("1. Upload your document(s)")
    uploaded_files = st.file_uploader(
        "Upload PDF(s), DOCX, TXT, MD, or images",
        type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Process document(s)", type="primary"):
        with st.spinner(f"Reading, chunking, and embedding {len(uploaded_files)} file(s)..."):
            vectorstore, chunk_count, debug_info = process_documents(uploaded_files)
            st.session_state.vectorstore = vectorstore
            st.session_state.chunk_count = chunk_count
            # New batch of documents = fresh conversation. Wipe any memory from before.
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True, output_key="answer"
            )
            st.session_state.chat_display = []
        st.success(
            f"Processed {len(uploaded_files)} file(s) into {chunk_count} chunks. Ask questions below!"
        )
        with st.expander("Debug: text extracted per file"):
            for info in debug_info:
                if info["chars_extracted"] < 20:
                    st.warning(f"⚠️ {info['name']}: only {info['chars_extracted']} characters extracted — likely failed to read this file properly.")
                else:
                    st.write(f"✅ {info['name']}: {info['chars_extracted']} characters extracted")

    st.divider()
    st.caption(
        "How it works: your PDF is split into overlapping text chunks, each turned into "
        "a vector (embedding), and stored in a local vector database (Chroma). This happens "
        "once per document — asking questions afterward is fast."
    )

# --- Main: ask questions ---
st.header("2. Ask questions")

if st.session_state.vectorstore is None:
    st.info("Upload and process a PDF from the sidebar to get started.")
else:
    if st.button("📋 What does this document contain? (overview)"):
        llm = get_llm()
        with st.spinner("Reading through the document for an overview..."):
            summary = summarize_documents(llm, st.session_state.vectorstore)
        st.session_state.chat_display.append(
            {
                "question": "What does this document contain? (overview)",
                "answer": summary,
                "sources": [],
                "verification": "(Overview mode reads a spread of the document directly, "
                "not via similarity search, so no single-chunk confidence check applies.)",
            }
        )
        st.rerun()

    # Show the conversation so far, oldest first
    for turn in st.session_state.chat_display:
        st.subheader("Q: " + turn["question"])
        st.markdown(normalize_math(turn["answer"]))
        for doc in turn["sources"]:
            page = doc.metadata.get("page", "unknown")
            source_name = doc.metadata.get("source", "document")
            with st.expander(f"{source_name} — Page {page}"):
                st.markdown(doc.page_content)
        with st.expander("Confidence & anticipated follow-ups"):
            st.markdown(normalize_math(turn["verification"]))
        st.divider()

    query = st.text_input("Your question about the document:", key=f"q_{len(st.session_state.chat_display)}")

    if query:
        llm = get_llm()
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 6})
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=st.session_state.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        )

        with st.spinner("Searching the document and generating an answer..."):
            result = qa_chain.invoke({"question": query})
            verification = verify_answer(
                llm, query, result["answer"], result["source_documents"]
            )

        st.session_state.chat_display.append(
            {
                "question": query,
                "answer": result["answer"],
                "sources": result["source_documents"],
                "verification": verification,
            }
        )
        st.rerun()