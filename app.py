import os
import tempfile

import streamlit as st

from core.pipeline import ingest_pdf, ask_stream, get_documents, remove_document
from core.vectorstore import (
    create_chat_session,
    list_chat_sessions,
    get_chat_messages,
    save_chat_message,
    update_session_title,
    delete_chat_session,
    save_api_key,
    get_api_key,
)

NO_INFO_PHRASES = [
    "không chứa đủ thông tin",
    "không đủ thông tin",
    "không tìm thấy thông tin",
]


def _has_answer(text: str) -> bool:
    text_lower = text.lower()
    return not any(phrase in text_lower for phrase in NO_INFO_PHRASES)


st.set_page_config(
    page_title="IPC - Hỏi đáp Tài liệu",
    page_icon="📚",
    layout="wide",
)

# --- Session state init ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

# --- Sidebar ---
with st.sidebar:
    # Gemini Settings
    st.header("Cài đặt Gemini")

    saved_key = get_api_key("gemini")
    api_key = st.text_input(
        "API Key",
        type="password",
        value=saved_key,
        help="Lấy key miễn phí tại: https://aistudio.google.com/apikey",
    )
    if api_key and api_key != saved_key:
        try:
            from google import genai
            test_client = genai.Client(api_key=api_key)
            test_client.models.generate_content(
                model="gemini-2.0-flash",
                contents="hi",
                config=genai.types.GenerateContentConfig(max_output_tokens=5),
            )
            save_api_key("gemini", api_key)
            st.success("Key hợp lệ, đã lưu!")
        except Exception as e:
            st.error(f"Key không hợp lệ: {e}")
            api_key = saved_key
    model = st.selectbox(
        "Model",
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash",
         "gemini-1.5-flash", "gemini-1.5-pro"],
    )

    st.divider()

    # Chat history
    st.header("Lịch sử Chat")

    if st.button("+ Cuộc trò chuyện mới", use_container_width=True):
        st.session_state.current_session_id = None
        st.session_state.messages = []
        st.rerun()

    sessions = list_chat_sessions()
    for sess in sessions:
        col1, col2 = st.columns([4, 1])
        label = f"{sess['title'][:30]}"
        is_active = st.session_state.current_session_id == sess["id"]
        if col1.button(
            f"{'> ' if is_active else ''}{label}",
            key=f"sess_{sess['id']}",
            use_container_width=True,
        ):
            st.session_state.current_session_id = sess["id"]
            st.session_state.messages = get_chat_messages(sess["id"])
            st.rerun()
        if col2.button("X", key=f"del_sess_{sess['id']}"):
            delete_chat_session(sess["id"])
            if st.session_state.current_session_id == sess["id"]:
                st.session_state.current_session_id = None
                st.session_state.messages = []
            st.rerun()

    st.divider()

    # Document management
    st.header("Quản lý Tài liệu")

    upload_mode = st.radio("Nguồn file", ["Upload", "Đường dẫn local"], horizontal=True)

    stage_labels = {
        "extract": "Đang trích xuất text từ PDF...",
        "chunk": "Đang chia nhỏ tài liệu...",
        "embed": "Đang tạo embeddings...",
        "done": "Hoàn tất!",
    }

    def on_progress(stage, current, total, _bar, _text):
        label = stage_labels.get(stage, stage)
        if total > 0 and stage == "embed":
            pct = current / total
            _text.text(f"{label} ({current}/{total} chunks)")
        else:
            pct = 0.05 if stage == "extract" else 0.1 if stage == "chunk" else 1.0
            _text.text(label)
        _bar.progress(min(pct, 1.0))

    if upload_mode == "Upload":
        uploaded_file = st.file_uploader("Tải lên PDF", type=["pdf"])
        if uploaded_file and st.button("Xử lý tài liệu"):
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            progress_bar = st.progress(0)
            status_text = st.empty()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                count = ingest_pdf(
                    tmp_path, uploaded_file.name,
                    progress_callback=lambda s, c, t: on_progress(s, c, t, progress_bar, status_text),
                )
                progress_bar.progress(1.0)
                status_text.empty()
                st.success(f"Đã lưu: {uploaded_file.name} ({count} chunks, {file_size_mb:.1f}MB)")
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Lỗi: {e}")
            finally:
                os.unlink(tmp_path)
    else:
        local_path = st.text_input(
            "Đường dẫn file PDF",
            placeholder=r"C:\Users\lucif\Downloads\document.pdf",
        )
        if local_path and st.button("Xử lý tài liệu"):
            # Auto-convert Windows path to Docker mount path
            # D:\folder\file.pdf -> /data/local/folder/file.pdf
            resolved = local_path.strip().strip('"').strip("'")
            if len(resolved) >= 2 and resolved[1] == ":":
                # Windows absolute path: C:\... or D:\...
                drive_letter = resolved[0].upper()
                rest = resolved[2:].replace("\\", "/")
                resolved = f"/data/local/{drive_letter}{rest}"

            if not os.path.isfile(resolved):
                st.error(f"File không tồn tại: {local_path}\n(Docker path: {resolved})")
            elif not resolved.lower().endswith(".pdf"):
                st.error("Chỉ hỗ trợ file PDF.")
            else:
                file_size_mb = os.path.getsize(resolved) / (1024 * 1024)
                filename = os.path.basename(resolved)
                progress_bar = st.progress(0)
                status_text = st.empty()
                try:
                    count = ingest_pdf(
                        resolved, filename,
                        progress_callback=lambda s, c, t: on_progress(s, c, t, progress_bar, status_text),
                    )
                    progress_bar.progress(1.0)
                    status_text.empty()
                    st.success(f"Đã lưu: {filename} ({count} chunks, {file_size_mb:.1f}MB)")
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Lỗi: {e}")

    st.divider()
    st.subheader("Tài liệu đã lưu")
    docs = get_documents()
    if not docs:
        st.info("Chưa có tài liệu nào.")
    for doc in docs:
        col1, col2 = st.columns([3, 1])
        col1.text(f"{doc['name'][:30]} ({doc['chunks']} chunks)")
        if col2.button("Xóa", key=f"del_{doc['hash']}"):
            remove_document(doc["hash"])
            st.rerun()

    st.divider()
    st.caption(f"Gemini | {model}")

# --- Main chat area ---
st.title("IPC - Hỏi đáp Tài liệu Học tập")

if not api_key:
    st.warning("Nhập API Key ở sidebar để bắt đầu hỏi đáp.")

if not get_documents():
    st.info("Hãy tải lên ít nhất một tài liệu PDF ở thanh bên trái để bắt đầu.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources") and _has_answer(msg["content"]):
            with st.expander("Nguồn tham khảo"):
                for src in msg["sources"]:
                    page = src["metadata"].get("page_number", "?")
                    source_file = src["metadata"].get("source_file", "?")
                    st.caption(f"📄 Trang {page} - {source_file}")
                    text = src.get("parent_text", src["document"])
                    st.text(text[:500] + "..." if len(text) > 500 else text)
                    st.divider()

# Chat input
if prompt := st.chat_input("Đặt câu hỏi về tài liệu..."):
    if not api_key:
        st.error("Vui lòng nhập API Key ở sidebar.")
        st.stop()

    # Auto-create session on first message
    if st.session_state.current_session_id is None:
        title = prompt[:50] + ("..." if len(prompt) > 50 else "")
        session_id = create_chat_session(title)
        st.session_state.current_session_id = session_id
    else:
        session_id = st.session_state.current_session_id

    # Save & show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_message(session_id, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate & save assistant response
    with st.chat_message("assistant"):
        try:
            stream, sources = ask_stream(
                prompt, model=model, api_key=api_key
            )
            response = st.write_stream(stream)
        except Exception as e:
            response = f"Lỗi: {e}"
            st.error(response)
            sources = []

    # Save to DB
    save_sources = sources if _has_answer(response) else None
    save_chat_message(session_id, "assistant", response, save_sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources,
    })

    if sources and _has_answer(response):
        with st.expander("Nguồn tham khảo"):
            for src in sources:
                page = src["metadata"].get("page_number", "?")
                source_file = src["metadata"].get("source_file", "?")
                st.caption(f"📄 Trang {page} - {source_file}")
                text = src.get("parent_text", src["document"])
                st.text(text[:500] + "..." if len(text) > 500 else text)
                st.divider()
