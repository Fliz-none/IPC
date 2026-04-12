# IPC - Hỏi đáp Tài liệu Học tập bằng AI

Ứng dụng RAG trích xuất nội dung từ PDF và trả lời câu hỏi dựa trên tài liệu.

## Stack

- **Embedding**: BGE-M3 (1024-dim, multilingual)
- **Database**: PostgreSQL + pgvector (vector + full-text hybrid search)
- **LLM**: Gemini API (mặc định) / Ollama local
- **UI**: Streamlit
- **Deploy**: Docker Compose

## Khởi chạy

```bash
git clone <repo-url>
cd IPC
docker compose up -d --build
```

3 container:
- `ipc_postgres` - PostgreSQL + pgvector (port 5432)
- `ipc_ollama` - Ollama LLM server (port 11434)
- `ipc_app` - Streamlit UI (port 8501)

Mở **http://localhost:8501** -> nhập Gemini API Key -> upload PDF -> hỏi đáp.

Lấy Gemini API Key miễn phí: https://aistudio.google.com/apikey

## Cấu trúc

```
IPC/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── config.py
├── app.py
├── core/
│   ├── pdf_extract.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── generator.py
│   └── pipeline.py
├── prompts/
│   └── qa_prompt.py
└── scripts/
    └── init.sql
```

## GPU (optional)

Uncomment phần `deploy` trong `docker-compose.yml` service `ollama` để dùng NVIDIA GPU.
