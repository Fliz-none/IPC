# === CORE RULES (không đổi - đảm bảo accuracy) ===
CORE_RULES = """Quy tắc bắt buộc:
- Chỉ sử dụng thông tin từ phần "Ngữ cảnh" để trả lời. KHÔNG bịa thêm thông tin.
- Luôn trích dẫn số trang nguồn (ví dụ: [Trang 5]).
- Nếu ngữ cảnh không đủ thông tin, nói rõ: "Tài liệu không chứa đủ thông tin để trả lời câu hỏi này."
"""

# === DEFAULT USER STYLE (user có thể thay đổi) ===
DEFAULT_STYLE = """Bạn là trợ lý học tập thông minh.
- Trả lời bằng ngôn ngữ giống với câu hỏi.
- Trả lời đầy đủ, rõ ràng và có cấu trúc."""

USER_PROMPT_TEMPLATE = """Ngữ cảnh:
{context}

Câu hỏi: {question}"""


def build_system_prompt(user_style: str = "") -> str:
    """Combine core rules (fixed) + user style (customizable)."""
    style = user_style.strip() if user_style else DEFAULT_STYLE
    return f"{style}\n\n{CORE_RULES}"


def build_context(chunks: list[dict]) -> str:
    parts = []
    seen_parents = set()
    for chunk in chunks:
        page = chunk["metadata"].get("page_number", "?")
        source = chunk["metadata"].get("source_file", "?")
        parent = chunk.get("parent_text", chunk["document"])
        parent_key = parent[:100]
        if parent_key in seen_parents:
            continue
        seen_parents.add(parent_key)
        parts.append(f"[Trang {page} - {source}]\n{parent}")
    return "\n\n---\n\n".join(parts)


def format_prompt(question: str, chunks: list[dict], user_style: str = "") -> list[dict]:
    context = build_context(chunks)
    return [
        {"role": "system", "content": build_system_prompt(user_style)},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                context=context, question=question
            ),
        },
    ]
