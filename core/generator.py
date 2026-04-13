"""Multi-provider LLM generator with auto-fallback on quota errors."""

from google import genai
from openai import OpenAI

from config import LLM_TEMPERATURE
from prompts.qa_prompt import format_prompt

# Provider configs: (name, base_url, default_model, key_prefix)
PROVIDERS = [
    {
        "name": "gemini",
        "type": "gemini",
        "default_model": "gemini-2.5-flash",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        "key_prefix": "AIza",
        "key_help": "https://aistudio.google.com/apikey",
    },
    {
        "name": "groq",
        "type": "openai_compat",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it", "mixtral-8x7b-32768"],
        "key_prefix": "gsk_",
        "key_help": "https://console.groq.com/keys",
    },
    {
        "name": "openrouter",
        "type": "openai_compat",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "google/gemini-2.0-flash-exp:free",
        "models": [
            "google/gemini-2.0-flash-exp:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "qwen/qwen3-8b:free",
        ],
        "key_prefix": "sk-or-",
        "key_help": "https://openrouter.ai/keys",
    },
]

# Cohere is for reranking only, not in PROVIDERS (LLM generation)
COHERE_CONFIG = {
    "name": "cohere",
    "key_prefix": "...",  # Cohere keys don't have a fixed prefix
    "key_help": "https://dashboard.cohere.com/api-keys",
}

QUOTA_ERROR_CODES = {429, 503}
QUOTA_ERROR_PHRASES = [
    "quota", "rate_limit", "rate limit", "exceeded", "overloaded",
    "resource_exhausted", "unavailable", "high demand", "503", "429",
    "too many requests", "temporarily", "try again later",
]


def _is_quota_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(phrase in msg for phrase in QUOTA_ERROR_PHRASES)


def _messages_to_parts(messages: list[dict]) -> tuple[str, str]:
    system_msg = ""
    user_msg = ""
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            user_msg = m["content"]
    return system_msg, user_msg


# --- Gemini ---

def _gemini_stream(messages: list[dict], model: str, api_key: str):
    client = genai.Client(api_key=api_key)
    system_msg, user_msg = _messages_to_parts(messages)
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


# --- OpenAI-compatible (Groq, OpenRouter) ---

def _openai_stream(messages: list[dict], model: str, api_key: str, base_url: str):
    client = OpenAI(api_key=api_key, base_url=base_url)
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=LLM_TEMPERATURE,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


# --- Auto-fallback stream ---

def generate_answer_stream(
    query: str,
    chunks: list[dict],
    keys: dict = None,
    preferred_provider: str = "gemini",
    model: str = "",
    user_style: str = "",
    **kwargs,
):
    """Stream answer with auto-fallback across providers on quota errors."""
    messages = format_prompt(query, chunks, user_style=user_style)
    keys = keys or {}

    # Build ordered provider list: preferred first, then others
    ordered = []
    for p in PROVIDERS:
        if p["name"] == preferred_provider:
            ordered.insert(0, p)
        else:
            ordered.append(p)

    errors = []
    for provider in ordered:
        key = keys.get(provider["name"], "")
        if not key:
            continue

        use_model = model if (model and provider["name"] == preferred_provider) else provider["default_model"]

        try:
            if provider["type"] == "gemini":
                collected = []
                for token in _gemini_stream(messages, use_model, key):
                    collected.append(token)
                    yield token
                return  # Success
            else:
                collected = []
                for token in _openai_stream(messages, use_model, key, provider["base_url"]):
                    collected.append(token)
                    yield token
                return  # Success
        except Exception as e:
            if _is_quota_error(e):
                errors.append(f"{provider['name']}: quota exceeded")
                continue  # Try next provider
            else:
                raise e

    if errors:
        yield f"\n\nTất cả providers đều hết quota: {', '.join(errors)}"
    else:
        yield "Chưa có API key nào được cấu hình."
