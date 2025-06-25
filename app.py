import streamlit as st
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from transformers import pipeline

# --- Load and preprocess dataset ---
@st.cache_resource
def load_documents():
    dataset = load_dataset("lvwerra/codeparrot-clean-train", split="train[:1000]")
    texts = [item['content'] for item in dataset]
    documents = [Document(page_content=t) for t in texts]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

# --- Build ChromaDB ---
@st.cache_resource
def build_retriever(split_docs):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(embedding_function=embeddings)
    
    # Insert documents in one go if not already added
    if len(db.get()["ids"]) == 0:
        db.add_documents(split_docs)
    return db.as_retriever()

# --- Load lightweight model ---
@st.cache_resource
def load_generator():
    return pipeline("text2text-generation", model="google/flan-t5-small")

# --- Core QA logic ---
def rag_qa(query, retriever, generator):
    context_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in context_docs[:3]])  # Limit context
    prompt = f"Answer this question based on the following context:\n{context}\n\nQuestion: {query}"
    result = generator(prompt, max_length=512, temperature=0.1)
    return result[0]['generated_text'], context

# --- UI ---
st.title("💬 RAG Coding Assistant")
st.markdown("Ask programming questions. Answers are grounded in real code examples.")

query = st.text_input("Enter your coding question:")
if query:
    with st.spinner("Generating answer..."):
        docs = load_documents()
        retriever = build_retriever(docs)
        generator = load_generator()
        answer, context = rag_qa(query, retriever, generator)

    st.subheader("Answer:")
    st.write(answer)

    with st.expander("🔍 Show Retrieved Context"):
        st.code(context)
