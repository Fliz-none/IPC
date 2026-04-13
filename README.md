# IPC - AI-Powered Study Document Q&A System (RAG)

An advanced **RAG (Retrieval-Augmented Generation)** application designed to extract content from PDFs and answer questions with high precision using Hybrid Search and Reranking architectures.

## 🚀 Key Features

-   **Parent-Child Chunking Strategy**: Optimizes retrieval by indexing small chunks while providing full-sized parent chunks to the LLM for superior context.
-   **Hybrid Search**: Combines Semantic Vector Search with Keyword-based Full-text Search via the **RRF (Reciprocal Rank Fusion)** algorithm in PostgreSQL.
-   **Cross-Encoder Reranking**: Utilizes a specialized reranker model to re-score candidates, ensuring the most relevant information is always prioritized.
-   **Multi-Provider Fallback**: Seamlessly switches between **Gemini**, **Groq**, and **OpenRouter** in case of API quota limits.
-   **Streaming Response**: Real-time answer generation for a smooth user experience.
-   **Source Citation**: Provides precise references including page numbers and source text snippets from the original PDF.

## 🛠 Tech Stack

-   **Embedding**: `BGE-M3` (Multilingual, 1024-dim) - State-of-the-art for multilingual retrieval.
-   **Reranker**: `MMARCO MiniLM` (Cross-Encoder).
-   **Database**: **Supabase** (PostgreSQL + `pgvector`).
-   **LLM**: Google Gemini API (Primary), Groq (Llama 3.3), OpenRouter.
-   **UI**: Streamlit.

---

## 🏗 System Architecture

```mermaid
graph TD
    A[PDF Upload] --> B[PyMuPDF Extract]
    B --> C[Parent-Child Chunking]
    C --> D[BGE-M3 Embedding]
    D --> E[(Supabase pgvector)]
    
    F[User Question] --> G[BGE-M3 Embed Query]
    G --> H[Hybrid Search + RRF]
    H --> I[Cross-Encoder Reranking]
    I --> J[Prompt Construction]
    J --> K[LLM Generation]
    K --> L[Streaming Output + Sources]
```

---

## ⚙️ Installation & Setup

### 1. Supabase Setup (One-time)

1.  Visit [Supabase](https://supabase.com) -> Create a new Project (Region: **Southeast Asia** recommended).
2.  Go to **SQL Editor** -> Paste the contents of `scripts/init.sql` -> Click **Run**.
3.  Navigate to **Project Settings > Database** -> Copy the **Connection string (URI)**.
    *   *Note: Use port `6543` (Transaction Pooler) for optimal connection management on Cloud environments.*

### 2. Environment Configuration

Create a `.env` file from the template:
```bash
cp .env.example .env
```

Update it with your credentials:
```env
DATABASE_URL=postgresql://postgres.[YOUR_PROJECT_ID]:[PASSWORD]@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres
# API Keys can also be entered directly in the App Sidebar or kept in .env
GEMINI_API_KEY=your_api_key_here
```

### 3. Run with Docker (Recommended)

The system will automatically download the Embedding and Reranker models to cache (~1.5GB).

```bash
docker compose up -d --build
```
Access the app at: **http://localhost:8501**

---

## ☁️ Deployment (Streamlit Cloud)

This project is fully compatible with **Streamlit Community Cloud**:

1.  Push your code to GitHub (ensure `.env` is ignored via `.gitignore`).
2.  Connect your Repo to Streamlit Cloud.
3.  In **Settings > Secrets**, paste your `.env` content in TOML format:
    ```toml
    DATABASE_URL = "your_supabase_url"
    GEMINI_API_KEY = "your_key"
    ```
4.  The app will automatically install dependencies and start.

---

## 📂 Project Structure

-   `core/chunker.py`: Implementation of the Parent-Child chunking strategy.
-   `core/vectorstore.py`: Persistence logic and Hybrid Search implementation in PostgreSQL.
-   `core/retriever.py`: Retrieval pipeline and Reranking logic.
-   `core/generator.py`: LLM orchestration with Auto-fallback.
-   `app.py`: Streamlit frontend.
