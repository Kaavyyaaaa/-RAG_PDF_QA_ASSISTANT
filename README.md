# RAG PDF Q&A ASSISTANT

A local, privacy-friendly, open-source tool to **ask questions about your PDFs** using Retrieval-Augmented Generation (RAG) and free, local AI models.  
Upload any PDF, ask questions, and get intelligent, context-aware answers—**all on your own machine**.

---

## 🚀 Features

- **PDF Upload:** Drag and drop one or more PDFs for instant processing.
- **Local Embeddings:** Uses strong, free embedding models for semantic search.
- **RAG Pipeline:** Retrieves the most relevant chunks from your PDFs and generates answers using a local LLM.
- **No Internet Required:** All processing is local; your data never leaves your computer.
- **Chat History:** See your previous questions and answers.
- **Source Attribution:** See which PDF and chunk each answer is based on.
- **Dark/Light Mode Support:** Readable UI in both themes.
- **Free & Open Source:** No paid APIs, no vendor lock-in.

---

## 🖥️ Demo

<!-- Add your own screenshot if you want -->
<!-- ![Demo Screenshot](demo_screenshot.png) -->

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Kaavyyaaaa/-RAG_PDF_QA_ASSISTANT.git
cd RAG_PDF_QA_ASSISTANT
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## ⚙️ Usage

### 1. Start the App

```bash
streamlit run app.py
```

### 2. Open in Your Browser

Go to [http://localhost:8501](http://localhost:8501)  
Upload your PDFs, ask questions, and get answers!

---

## 🧠 How It Works

1. **PDF Processing:** Extracts and chunks text from uploaded PDFs.
2. **Embedding:** Generates vector embeddings for each chunk using `all-MiniLM-L6-v2` (or another free model).
3. **Storage:** Stores embeddings and metadata in a local ChromaDB database.
4. **Retrieval:** When you ask a question, retrieves the most relevant chunks using semantic search.
5. **Answer Generation:** Passes the context to a local LLM (default: `google/flan-t5-base`) to generate a detailed answer.
6. **Source Display:** Shows which PDF and chunk the answer is based on, along with a confidence score.

---

## 🛠️ Configuration

- **Embedding Model:** Change in `embedding_manager.py` (`EMBEDDING_MODEL_NAME`).
- **LLM Model:** Change in `llm_handler.py` (`DEFAULT_MODEL`).
- **Chunk Size/Overlap:** Tune in `pdf_processor.py` (function: `chunk_text`).
- **Database:** Uses local ChromaDB (no cloud).

---

## 📝 Example Questions

- What is the main topic of the document?
- Summarize the key findings.
- List the steps described in the manual.
- What are the limitations mentioned?
- Who are the authors?

---

## 🖤 Why Local/Private?

- **Privacy:** Your documents and questions never leave your computer.
- **No API Costs:** 100% free and open source.
- **Customizable:** Swap out models, tweak chunking, or add features as you wish.

---

## 🧩 Dependencies

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- [sentence-transformers](https://www.sbert.net/)
- [ChromaDB](https://www.trychroma.com/)
- [transformers](https://huggingface.co/transformers/)
- [loguru](https://github.com/Delgan/loguru)
- [streamlit-extras](https://github.com/arnaudmiribel/streamlit-extras)

See `requirements.txt` for the full list.

---

## 🙏 Acknowledgements

- [HuggingFace](https://huggingface.co/) for open-source models.
- [ChromaDB](https://www.trychroma.com/) for vector storage.
- [Streamlit](https://streamlit.io/) for the UI framework.

---

## 📄 License

This project is licensed under the MIT License.

---

## 💡 Future Ideas

- Support for scanned PDFs (OCR)
- Multi-language support
- More powerful local LLMs
- Export Q&A history

---

**Questions or suggestions?**  
Open an issue or pull request on [GitHub](https://github.com/Kaavyyaaaa/-RAG_PDF_QA_ASSISTANT.git)!
