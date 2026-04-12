SYSTEM_PROMPT = """Bạn là trợ lý học tập thông minh. Trả lời câu hỏi CHỈ dựa trên nội dung tài liệu được cung cấp bên dưới.

Quy tắc:
- Chỉ sử dụng thông tin từ phần "Ngữ cảnh" để trả lời. KHÔNG bịa thêm thông tin.
- Luôn trích dẫn số trang nguồn (ví dụ: [Trang 5]).
- Nếu ngữ cảnh không đủ thông tin để trả lời, hãy nói rõ: "Tài liệu không chứa đủ thông tin để trả lời câu hỏi này."
- Trả lời bằng ngôn ngữ giống với câu hỏi.
- Trả lời đầy đủ, rõ ràng và có cấu trúc."""

USER_PROMPT_TEMPLATE = """Ngữ cảnh:
{context}

Câu hỏi: {question}"""


def build_context(chunks: list[dict]) -> str:
    """Build context string from retrieved chunks, using parent_text for richer context."""
    parts = []
    seen_parents = set()
    for chunk in chunks:
        page = chunk["metadata"].get("page_number", "?")
        source = chunk["metadata"].get("source_file", "?")
        # Use parent_text (expanded context) instead of child chunk
        parent = chunk.get("parent_text", chunk["document"])
        # Deduplicate overlapping parent texts
        parent_key = parent[:100]
        if parent_key in seen_parents:
            continue
        seen_parents.add(parent_key)
        parts.append(f"[Trang {page} - {source}]\n{parent}")
    return "\n\n---\n\n".join(parts)


def format_prompt(question: str, chunks: list[dict]) -> list[dict]:
    """Format the full message list for the LLM."""
    context = build_context(chunks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                context=context, question=question
            ),
        },
    ]
