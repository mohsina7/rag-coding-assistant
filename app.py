import streamlit as st
from transformers import pipeline
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datasets import load_dataset
import os

# Load HF token
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# BGE Embedding
embedding_model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Load dataset
dataset = load_dataset("codeparrot/codeparrot-clean", split="train[:1000]")
texts = [item["content"] for item in dataset]

# Chunk texts
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.create_documents(texts)

# Create Chroma vector store
db = Chroma(embedding_function=embedding_model)
for i in range(0, len(split_docs), 500):
    db.add_documents(split_docs[i:i+500])

# Retriever
retriever = db.as_retriever()

# Generator model
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    token=HF_TOKEN
)

# RAG QA
def rag_qa(query):
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Answer this question based on the following context:\n{context}\n\nQuestion: {query}"
    result = generator(prompt, max_length=512)
    return result[0]["generated_text"]

# UI
st.title("💬 RAG-based AI Coding Assistant")
query = st.text_input("Ask your coding question:")

if st.button("Get Answer"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = rag_qa(query)
            st.success(answer)
