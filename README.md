# RAG-Based Document Q\&A System

A Retrieval-Augmented Generation (RAG) system that enables users to upload documents and ask intelligent questions with real-time responses, citations, and summarization.

---

## Features

* Multi-format document support (PDF, DOCX, CSV, XLSX, PNG)
* Context-aware Q\&A with chat history
* Real-time streaming answers (typing effect)
* Citations with page numbers and snippets
* Document summarization (short, medium, detailed)
* OCR support for image-based PDFs and PNGs
* Semantic vector search with ChromaDB
* Modern UI with dark/light mode

---

## System Flow

1. **Document Ingestion**

   * User uploads file
   * File type detected
   * Text extracted (OCR fallback for images/scanned PDFs)

2. **Processing**

   * Text split into chunks
   * Metadata (page numbers, source info) attached

3. **Storage**

   * Embeddings generated using Sentence Transformers
   * Stored in ChromaDB vector database

4. **Question Answering**

   * User question converted to embedding
   * Similar chunks retrieved from ChromaDB
   * Google Gemini LLM generates answer with citations

5. **Response Delivery**

   * Tokens streamed in real time to the frontend
   * Citations shown alongside answers

---


**Flow:**

```
📂 Upload Document → Text Extraction / OCR → Text Chunking + Metadata → Embeddings (Sentence Transformers) → ChromaDB (Vector Store) → 🔎 Query + Similarity Search → 🧠 Google Gemini LLM → 📑 Answer + Citations (Streaming to UI)
```

*Export diagram as PNG and include in README:*

```mermaid
graph TB
    U[User] --> F[Next.js Frontend]
    F --> FU[File Upload] --> B[FastAPI Backend]
    F --> CUI[Chat UI] --> B

    B --> DP[Document Processing]
    DP --> OCR[OCR Extraction]
    DP --> TXT[Text Chunking]
    TXT --> EMB[Embedding Generation]
    EMB --> DB[ChromaDB Vector Store]

    CUI --> ASK[Question Endpoint]
    ASK --> QP[Query Processing]
    QP --> RET[Context Retrieval]
    RET --> LLM[Google Gemini LLM]
    LLM --> RESP[Answer + Citations]
    RESP --> STREAM[Streaming Response]

    CUI --> SUMM[Summarization Endpoint]
    SUMM --> LLM


---

## Technology Stack

### Backend

* FastAPI (Python 3.12+)
* Google Gemini 2.5 Flash LLM
* ChromaDB (vector database)
* Sentence Transformers (all-MiniLM-L6-v2)
* pdfplumber, PyMuPDF, python-docx, pandas, openpyxl
* pytesseract + pdf2image for OCR

### Frontend

* Next.js 13.5 (TypeScript + React 18)
* Tailwind CSS + shadcn/ui + Radix UI
* react-markdown with KaTeX
* Fetch API for backend communication

---

## Project Structure

```
RAG assignment v1/
├── Backend/
│   ├── main.py              # FastAPI entry point
│   ├── rag.py               # Core RAG logic
│   ├── pdf_utils.py         # Document processing
│   ├── requirements.txt     # Python dependencies
│   └── media/               # Uploaded docs
│
├── Frontend/
│   ├── app/qa/page.tsx      # Q&A interface
│   ├── components/ChatUI.tsx
│   ├── components/FileUploader.tsx
│   └── lib/api-config.ts
```

---

## Setup Instructions

### Prerequisites

* Python 3.12+
* Node.js 18+
* npm or yarn
* Tesseract OCR (for image-based docs)

### Backend

```bash
cd Backend
pip install -r requirements.txt

# .env file
GOOGLE_API_KEY=your_gemini_api_key
CHROMA_DB_DIR=.chroma
MEDIA_ROOT=media
```

### Frontend

```bash
cd Frontend
npm install

# .env.local file
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Run

```bash
# Backend
cd Backend
python main.py   # http://localhost:8000

# Frontend
cd Frontend
npm run dev      # http://localhost:3000
```

---

## API Endpoints

* **POST /upload** – Upload and process documents
* **POST /ask** – Q\&A with full response
* **POST /ask\_stream** – Streaming answers
* **POST /summarize** – Document summarization (short, medium, detailed)
* **GET /health** – Health check and DB status

---

## Extra Features Implemented

* OCR fallback for scanned/image-based files
* Streaming answers with typing animation
* Multi-level summarization
* Page-level citations with snippets
* Responsive frontend with dark/light theme
* Chat history context retention

---

## Future Enhancements

* Multi-language OCR and LLM support
* Usage analytics and insights
* Shared docs and collaborative Q\&A
* Mobile companion app
* Scaling with Postgres/Redis

---

## License

MIT License
