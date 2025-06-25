import streamlit as st
from langchain_community.vectorstores import Chroma
import chromadb  # To disable telemetry
chromadb.config.settings.TELEMETRY_ENABLED = False

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import load_dataset
import os
import requests

# Load HF token from env
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load embedding function
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Load dataset (small portion for demo)
dataset = load_dataset("codeparrot/codeparrot-clean", split="train[:100]")
texts = [item['content'] for item in dataset]

# Split documents
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.create_documents(texts)

# Create vectorstore (Chroma) with LangChain interface
db = Chroma.from_documents(split_docs, embedding_function=embedding_model)

# Setup retriever
retriever = db.as_retriever()

# Call HuggingFace Inference API
def call_hf_api(prompt):
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_length": 512}}

    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()[0]["generated_text"]

# RAG QA logic
def rag_qa(query):
    context_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = f"Answer the following question based on context:\n\n{context}\n\nQuestion: {query}"
    return call_hf_api(prompt)

# Streamlit UI
st.title("💬 RAG-based AI Coding Assistant")
st.write("Ask me a programming question!")

query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            try:
                answer = rag_qa(query)
                st.success(answer)
            except Exception as e:
                st.error(f"Error: {e}")
