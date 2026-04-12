from google import genai

from config import LLM_TEMPERATURE, GEMINI_API_KEY, GEMINI_MODEL
from prompts.qa_prompt import format_prompt

_client = None


def _get_client(api_key: str = ""):
    global _client
    key = api_key or GEMINI_API_KEY
    if _client is None and key:
        _client = genai.Client(api_key=key)
    elif api_key and api_key != GEMINI_API_KEY:
        _client = genai.Client(api_key=api_key)
    return _client


def generate_answer_stream(
    query: str,
    chunks: list[dict],
    model: str = "",
    api_key: str = "",
    **kwargs,
):
    """Stream answer using Gemini."""
    messages = format_prompt(query, chunks)
    client = _get_client(api_key)
    model = model or GEMINI_MODEL

    system_msg = ""
    user_msg = ""
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            user_msg = m["content"]

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=user_msg,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_msg,
            temperature=LLM_TEMPERATURE,
        ),
    ):
        if chunk.text:
            yield chunk.text
