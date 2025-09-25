import streamlit as st
from sentence_transformers import SentenceTransformer # provides pretrained embedding models to convert text into vectors
from transformers import pipeline
import faiss # a vector database (fast similarity search)
import os
from PyPDF2 import PdfReader # extracts text from PDF files

# -------------------
# Chunk text with metadata (doc_name + chunk_id)
# -------------------
def chunk_text(text, doc_name, chunk_size=500, overlap=50):
    chunks = [] 
    # overlap -> to avoid cutting off context mid-sentence
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i+chunk_size]
        chunks.append({"doc": doc_name, "text": chunk})
    return chunks

# -------------------
# Load and chunk all docs
# -------------------
def load_chunks(folder_path="docs"):
    all_chunks = []
    for fname in os.listdir(folder_path):
        path = os.path.join(folder_path, fname)
        if fname.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            all_chunks.extend(chunk_text(text, fname))
        elif fname.endswith(".pdf"):
            pdf = PdfReader(path)
            text = "".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            all_chunks.extend(chunk_text(text, fname))
    # Returns a list of all chunks from all files. 
    return all_chunks

# -------------------
# Build vector DB
# -------------------
@st.cache_resource # ensures this step runs only once, even if Streamlit reruns the script.
def build_index():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    # Uses FAISS to create a vector index where we can quickly search for relevant chunks.
    chunks = load_chunks("docs")
    texts = [c["text"] for c in chunks]

    embeddings = embedder.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1]) # IndexFlatL2 uses Euclidean distance for similarity search.
    index.add(embeddings)
    return embedder, index, chunks

embedder, index, chunks = build_index()

# -------------------
# Retrieval with references
# -------------------
def retrieve(query, k=1):
    q_emb = embedder.encode([query])   # Embeds the query into a vector
    distances, indices = index.search(q_emb, k) # Searches the FAISS index for the top k=3 closest chunks.
    results = []
    for i in indices[0]:
        results.append(chunks[i])  # each result has {"doc": ..., "text": ...}
    # Returns the corresponding chunk dictionaries.
    return results

# -------------------
# QA Model
# -------------------
@st.cache_resource  # ensures this step runs only once, even if Streamlit reruns the script.
def load_qa_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

qa_model = load_qa_model()

# -------------------
# Generate answer + references
# -------------------
def answer_query(query):
    retrieved = retrieve(query)
    context = " ".join([r["text"] for r in retrieved])
    prompt = f"Answer the question based on the context:\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"
    result = qa_model(prompt, max_length=200)
    return result[0]["generated_text"], retrieved

# -------------------
# Streamlit UI
# -------------------
st.title("📚 RAG Assistant with References")
st.write("Ask questions about your documents in the `docs/` folder.")

query = st.text_input("Your question:")
if query:
    answer, retrieved = answer_query(query)
    st.markdown("### ✅ Answer")
    st.write(answer)

    st.markdown("### 📖 References")
    for idx, r in enumerate(retrieved, start=1):
        st.markdown(f"**[{idx}] {r['doc']}**: {r['text'][:200]}...")
