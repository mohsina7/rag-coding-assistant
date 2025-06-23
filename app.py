import streamlit as st
from transformers import pipeline
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datasets import load_dataset
import os

# Load HF token from env
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load dataset
dataset = load_dataset("codeparrot/codeparrot-clean", split="train[:1000]")
texts = [item['content'] for item in dataset]

# Split documents
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.create_documents(texts)

# Insert to Chroma in batches
db = Chroma(embedding_function=embedding_model)
batch_size = 500

for i in range(0, len(split_docs), batch_size):
    end_i = min(i + batch_size, len(split_docs))
    print(f"Inserting batch {i} to {end_i} ...")
    db.add_documents(split_docs[i:end_i])

# Setup retriever
retriever = db.as_retriever()

# Load model pipeline
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    token=HF_TOKEN
)

# QA function
def rag_qa(query):
    context_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = f"Answer this question based on the following context:\n{context}\n\nQuestion: {query}"
    result = generator(prompt, max_length=512)
    return result[0]['generated_text']

# Streamlit UI
st.title("💬 RAG-based AI Coding Assistant")
st.write("Ask me a programming question!")

query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            answer = rag_qa(query)
            st.success(answer)
