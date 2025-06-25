import streamlit as st
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import requests

# Use Streamlit secrets for HF API token securely
HF_API_TOKEN = st.secrets.get("HF_API_TOKEN", None)
if HF_API_TOKEN is None:
    st.error("Error: HF_API_TOKEN not set in Streamlit secrets. Please add it to deploy.")
    st.stop()

HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

CODE_SNIPPETS = [
    "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)",
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    "def is_prime(num):\n    if num <= 1:\n        return False\n    for i in range(2, int(num ** 0.5) + 1):\n        if num % i == 0:\n            return False\n    return True",
    "def reverse_string(s):\n    return s[::-1]",
    "def sum_list(lst):\n    total = 0\n    for num in lst:\n        total += num\n    return total",
]

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
    model.eval()
    return tokenizer, model

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings.cpu().numpy()

@st.cache_resource(show_spinner=False)
def build_faiss_index(snippets):
    # We load the embedding model internally here to avoid passing unhashable params
    tokenizer, model = load_embedding_model()
    embeddings = np.vstack([get_embedding(s, tokenizer, model) for s in snippets])
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)
    return index

def retrieve_relevant_snippets(query, tokenizer, model, index, snippets, top_k=3):
    query_emb = get_embedding(query, tokenizer, model).reshape(1, -1)
    distances, indices = index.search(query_emb, top_k)
    return [snippets[i] for i in indices[0]]

def query_flan_t5(prompt, api_token=HF_API_TOKEN):
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens":150, "return_full_text": False}}
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result[0]['generated_text']
    else:
        st.error(f"Hugging Face API Error {response.status_code}: {response.text}")
        return None

def main():
    st.title("🚀 RAG Code Explainer - Python Code Assistance")

    st.markdown(
        """
        Paste your Python code snippet or question about Python code.
        The system will retrieve relevant examples and generate explanations or suggestions.
        """
    )

    # Load tokenizer and model once (outside cached functions)
    tokenizer, model = load_embedding_model()
    # Build FAISS index without passing model/tokenizer here
    index = build_faiss_index(CODE_SNIPPETS)

    user_input = st.text_area("Enter Python code snippet or question:", height=150)

    if st.button("Explain / Suggest"):
        if not user_input.strip():
            st.warning("Please provide some input.")
            return

        with st.spinner("Retrieving relevant snippets..."):
            relevant_snippets = retrieve_relevant_snippets(user_input, tokenizer, model, index, CODE_SNIPPETS)

        st.subheader("🔍 Retrieved Code Snippets")
        for i, snippet in enumerate(relevant_snippets, 1):
            st.code(snippet, language="python")

        context = "\n\n".join([f"Example {i}:\n{snippet}" for i, snippet in enumerate(relevant_snippets, 1)])

        prompt = (
            f"Here are some Python code examples:\n{context}\n\n"
            f"User input:\n{user_input}\n\n"
            "Based on the above examples, provide a clear explanation, suggestions, or fixes."
        )

        with st.spinner("Generating response from Hugging Face FLAN-T5..."):
            answer = query_flan_t5(prompt)

        if answer:
            st.subheader("💡 Explanation / Suggestions")
            st.write(answer)

if __name__ == "__main__":
    main()