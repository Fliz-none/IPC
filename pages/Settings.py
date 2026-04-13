import streamlit as st

from core.generator import PROVIDERS
from core.vectorstore import save_api_key, get_api_key, delete_api_key

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

# --- Cohere Keys (Embedding - multi-key) ---
st.header("Cohere (Embedding)")
st.info("**Bắt buộc** - thêm nhiều key để xoay vòng khi hết quota. Mỗi key free 1000 req/tháng.")

MAX_COHERE_KEYS = 5

# Load existing keys
cohere_keys = []
for i in range(1, MAX_COHERE_KEYS + 1):
    key_name = f"cohere_{i}"
    k = get_api_key(key_name)
    if k:
        cohere_keys.append((key_name, k))

# Display existing keys
if cohere_keys:
    st.caption(f"{len(cohere_keys)} key(s) đã lưu - tự động xoay vòng")
    for key_name, key_val in cohere_keys:
        col1, col2 = st.columns([4, 1])
        col1.text(f"{key_name}: {key_val[:8]}...{key_val[-4:]}")
        if col2.button("Xóa", key=f"del_{key_name}"):
            delete_api_key(key_name)
            st.rerun()

# Add new key
st.caption("[Lấy key tại dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys)")
new_cohere = st.text_input(
    "Thêm Cohere key mới",
    type="password",
    placeholder="Dán key tại đây...",
    key="new_cohere_key",
)
if new_cohere and st.button("Lưu key"):
    if len(new_cohere) > 20:
        next_idx = len(cohere_keys) + 1
        if next_idx <= MAX_COHERE_KEYS:
            save_api_key(f"cohere_{next_idx}", new_cohere)
            st.success(f"Cohere key {next_idx} đã lưu!")
            st.rerun()
        else:
            st.error(f"Tối đa {MAX_COHERE_KEYS} keys.")
    else:
        st.error("Key không hợp lệ.")

st.divider()

# --- Summary ---
st.header("Trạng thái")
all_providers = []
for p in PROVIDERS:
    k = get_api_key(p["name"])
    status = "✅ Sẵn sàng" if k else "❌ Chưa có"
    all_providers.append({"Provider": p["name"].upper(), "Vai trò": "LLM Generation", "Trạng thái": status})

cohere_count = len(cohere_keys)
cohere_status = f"✅ {cohere_count} key(s)" if cohere_count > 0 else "❌ Chưa có (bắt buộc)"
all_providers.append({"Provider": "COHERE", "Vai trò": "Embedding", "Trạng thái": cohere_status})

st.table(all_providers)
