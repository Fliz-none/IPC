import streamlit as st

from core.generator import PROVIDERS
from core.vectorstore import save_api_key, get_api_key

st.set_page_config(
    page_title="IPC - Cài đặt",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Cài đặt API Keys")
st.caption("Thêm nhiều key = tự động chuyển đổi khi hết quota")

st.divider()

# --- LLM Provider Keys ---
st.header("LLM Providers")
st.info("Cần ít nhất 1 key để hỏi đáp. Thêm nhiều key để auto-fallback khi hết quota.")

for p in PROVIDERS:
    col1, col2 = st.columns([3, 1])
    saved = get_api_key(p["name"])

    with col1:
        key_input = st.text_input(
            f"{p['name'].upper()}",
            type="password",
            value=saved,
            key=f"key_{p['name']}",
        )
    with col2:
        st.caption(f"[Lấy key]({p['key_help']})")
        if saved:
            st.success("Đã lưu", icon="✅")

    if key_input and key_input != saved:
        if key_input.startswith(p["key_prefix"]) and len(key_input) > 20:
            save_api_key(p["name"], key_input)
            st.success(f"{p['name'].upper()} key đã lưu!")
            st.rerun()
        else:
            st.error(f"Key phải bắt đầu bằng `{p['key_prefix']}...`")

st.divider()

# --- Cohere (Embedding) ---
st.header("Cohere (Embedding)")
st.info("**Bắt buộc** - dùng để chuyển văn bản thành vector cho tìm kiếm.")

col1, col2 = st.columns([3, 1])
saved_cohere = get_api_key("cohere")
with col1:
    cohere_input = st.text_input(
        "COHERE",
        type="password",
        value=saved_cohere,
        key="key_cohere",
    )
with col2:
    st.caption("[Lấy key](https://dashboard.cohere.com/api-keys)")
    if saved_cohere:
        st.success("Đã lưu", icon="✅")

if cohere_input and cohere_input != saved_cohere:
    if len(cohere_input) > 20:
        save_api_key("cohere", cohere_input)
        st.success("Cohere key đã lưu!")
        st.rerun()

st.divider()

# --- Summary ---
st.header("Trạng thái")
all_providers = []
for p in PROVIDERS:
    k = get_api_key(p["name"])
    status = "✅ Sẵn sàng" if k else "❌ Chưa có"
    all_providers.append({"Provider": p["name"].upper(), "Vai trò": "LLM Generation", "Trạng thái": status})

cohere_status = "✅ Sẵn sàng" if get_api_key("cohere") else "❌ Chưa có (bắt buộc)"
all_providers.append({"Provider": "COHERE", "Vai trò": "Embedding", "Trạng thái": cohere_status})

st.table(all_providers)
