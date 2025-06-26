# RAG Coding Assistant 🤖

A Retrieval-Augmented Generation (RAG) based AI assistant for answering coding questions.

Built with:
- LangChain
- HuggingFace Transformers
- Sentence Transformers
- ChromaDB
- Streamlit

### Features:
✅ Ingests code dataset  
✅ Splits and embeds to Chroma vector store  
✅ Retrieves relevant chunks on user question  
✅ Generates answer via LLM (Flan-T5)  

---

## Setup

```bash
# Create venv
python3 -m venv venv
source venv/bin/activate

# Install
pip install -r requirements.txt

# Run
streamlit run app.py
