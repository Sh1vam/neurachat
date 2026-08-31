# 🧠 NeuraChat Ultra

A Claude-like AI assistant with a production-grade UI, built on **FastAPI + Groq + RAG + Tool Calling**.

---

## 🚀 Quick Start

```bash
# 1. Navigate to project folder
cd neurachat

# 2. Install and run (one command)
python setup_and_run.py

# 3. Open browser → http://localhost:8000
```

---

## ✨ Features

| Feature | Status | Notes |
|---------|--------|-------|
| Multi-model chat (streaming) | ✅ | Llama 4, Llama 3.3, Mixtral, Gemma |
| Token calculator | ✅ | Real-time tracking + cost estimate |
| RAG — Document Q&A | ✅ | PDF, DOCX, TXT, PY, JS, CSV, MD |
| Image analysis (Vision) | ✅ | Llama 4 Scout/Maverick |
| Image generation | ✅ | Pollinations.ai (free, no key) |
| PowerPoint generation | ✅ | AI-structured slides, 4 themes |
| PDF report generation | ✅ | ReportLab, multi-section |
| Word document generation | ✅ | python-docx |
| Jupyter Notebook generation | ✅ | nbformat |
| Chart/Graph generation | ✅ | matplotlib, 4 chart types |
| Conversation history | ✅ | In-memory, per-session |
| Advanced RAG (ChromaDB) | 🔧 Optional | `pip install chromadb sentence-transformers` |
| Local model inference | 🔧 Optional | `pip install llama-cpp-python` |

---

## 📁 Project Structure

```
neurachat/
├── main.py              ← FastAPI backend (heavily commented)
├── requirements.txt     ← Python dependencies
├── setup_and_run.py     ← One-command setup
├── README.md            ← This file
├── outputs/             ← Generated files (PPT, PDF, etc.)
├── chroma_db/           ← Vector DB (created when ChromaDB installed)
└── static/
    └── index.html       ← Frontend (Claude-like UI)
```

---

## 🔌 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Server health + capabilities |
| GET | `/api/models` | List available models |
| POST | `/api/chat` | Chat (streaming SSE or JSON) |
| POST | `/api/upload` | Upload document to RAG |
| GET | `/api/documents` | List RAG documents |
| DELETE | `/api/documents/{id}` | Remove document |
| POST | `/api/analyze-image` | Vision analysis |
| POST | `/api/generate/pptx` | Create PowerPoint |
| POST | `/api/generate/pdf` | Create PDF |
| POST | `/api/generate/docx` | Create Word doc |
| POST | `/api/generate/notebook` | Create Jupyter notebook |
| POST | `/api/generate/chart` | Create chart image |
| POST | `/api/generate/image` | Generate image (Pollinations) |
| POST | `/api/ai-generate` | AI-structured file generation |
| GET | `/api/download/{file}` | Download generated file |
| GET | `/api/outputs` | List generated files |
| GET | `/api/tokens/stats` | Token usage stats |
| POST | `/api/tokens/reset` | Reset token counter |

---

## 🔮 Upgrade Path

### Level 1 (Current): Core App
- FastAPI + Groq + Streaming + RAG + File Generation ✅

### Level 2: Add ChromaDB
```bash
pip install chromadb sentence-transformers
# Restart server — ChromaDB auto-detected
```

### Level 3: Add LlamaIndex
```bash
pip install llama-index llama-index-llms-groq llama-index-embeddings-huggingface
```

### Level 4: Local model (no GPU)
```bash
pip install llama-cpp-python
# Download a GGUF model from Hugging Face
# llm = Llama(model_path="model.gguf", n_ctx=4096, n_threads=8, n_gpu_layers=0)
```

### Level 5: Fine-tune on Google Colab (free T4 GPU)
```python
# In Colab:
pip install transformers peft accelerate bitsandbytes trl datasets
# Use QLoRA to fine-tune Llama 3.2 3B on your data
# Export to GGUF → run locally with llama-cpp-python
```

---

## 💡 Key Concepts Explained in Code

Every concept is documented with multi-line comments in `main.py`:

- **FastAPI** — HTTP routing, Pydantic validation, CORS
- **Async/Await** — Non-blocking I/O, handling 1000s of simultaneous users
- **Streaming SSE** — Real-time token delivery to browser
- **RAG** — Simple (full doc injection) + Advanced (ChromaDB vector search)
- **Embeddings** — Text → vector → semantic similarity search
- **Token counting** — tiktoken, cost calculation, context window management
- **Vision models** — Base64 image encoding, multi-modal message format
- **File generation** — PPT, PDF, DOCX, .ipynb, charts
- **Tool calling** — Structured JSON output → file creation pipeline

Created with the help of Claude
