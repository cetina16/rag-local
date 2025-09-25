import os
import re
import shutil
import streamlit as st
from transformers import pipeline

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# -------------------
# Settings
# -------------------
INDEX_PATH = "faiss_index"  # Folder to persist FAISS index

# Regex patterns for repeated headers/footers in PDFs
HEADER_PATTERNS = [
    r"Reinforcement Learning \(RL\) Guide \| Unsloth Documentation",
    r"Reward Functions\s*/\s*Verifiers",
    r"\b\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*(AM|PM)?\b",  # dates/times
    r"^\s*📖.*$",  # lines starting with book emoji
]
header_regexes = [re.compile(p, re.IGNORECASE) for p in HEADER_PATTERNS]

# -------------------
# Utility: clean text
# -------------------
def _clean_page_text(text: str) -> str:
    lines = []
    for ln in text.splitlines():
        if any(rx.search(ln) for rx in header_regexes):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()

# -------------------
# Load and split documents
# -------------------
def load_documents(folder_path="docs"):
    docs = []
    for fname in os.listdir(folder_path):
        path = os.path.join(folder_path, fname)
        if fname.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())
        elif fname.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
    return docs

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    cleaned_docs = []
    for d in documents:
        cleaned_docs.append(
            Document(
                page_content=_clean_page_text(d.page_content),
                metadata=d.metadata
            )
        )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(cleaned_docs)

# -------------------
# Build or load FAISS vector store (persistent)
# -------------------
@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(INDEX_PATH):
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        raw_docs = load_documents("docs")
        split_docs = split_documents(raw_docs)
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local(INDEX_PATH)
    return vectorstore

vectorstore = get_vectorstore()

# -------------------
# QA Prompt
# -------------------
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a precise assistant. Answer the question ONLY using the context. "
        "If the answer is not in the context, say \"I don't know\".\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
)

# -------------------
# Load QA chain
# -------------------
@st.cache_resource
def load_qa_chain():
    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256,
        do_sample=False
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    retriever = vectorstore.as_retriever(
        search_type="mmr",  # maximal marginal relevance
        search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.5}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    return qa_chain

qa_chain = load_qa_chain()

# -------------------
# Streamlit UI
# -------------------
st.title("📚 LangChain RAG Assistant")
st.write("Ask questions about your documents in the `docs/` folder.")

query = st.text_input("Your question:")
if query:
    result = qa_chain(query)
    answer = result["result"]
    retrieved_docs = result["source_documents"]

    st.markdown("### ✅ Answer")
    st.write(answer)

    st.markdown("### 📖 References")
    for idx, doc in enumerate(retrieved_docs, start=1):
        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", None)
        page_info = f" (page {page})" if page is not None else ""
        preview = doc.page_content[:300].replace("\n", " ")
        st.markdown(f"**[{idx}] {src}{page_info}**: {preview}…")
