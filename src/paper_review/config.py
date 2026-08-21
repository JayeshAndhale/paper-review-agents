from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_groq import ChatGroq

load_dotenv()


class Settings(BaseSettings):
    groq_api_key: str
    llm_model: str
    llm_model_cheap: str
    embedding_model: str
    chroma_path: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()


def get_llm(tier: str, schema: type[BaseModel] | None = None, max_tokens: int = 1024):
    """max_tokens defaults small -- Groq's free tier caps prompt + max_tokens
    together per request (e.g. 8000 TPM), so a blanket large budget makes even
    trivial calls (e.g. classify_section's one-word answer) get rejected as
    'too large' before generation ever starts. Call sites that genuinely need
    more room (long drafts, many extracted claims) should pass a bigger value
    explicitly rather than raising the shared default.

    reasoning_effort is pinned to "low" -- these gpt-oss models reason by
    default, and for structured-output calls that hidden reasoning can eat
    the whole max_tokens budget before the model ever emits the required
    tool call, which Groq then rejects as 'did not call a tool' rather than
    truncating gracefully."""
    model_string = settings.llm_model if tier == "strong" else settings.llm_model_cheap
    provider, model = model_string.split("/", 1)

    if provider == "groq":
        llm = ChatGroq(
            model=model,
            api_key=settings.groq_api_key,
            max_tokens=max_tokens,
            reasoning_effort="low",
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if schema is not None:
        # function_calling (the default) relies on tool_choice=required, which
        # these gpt-oss reasoning models don't reliably honor on Groq -- they
        # sometimes answer in plain text instead of calling the tool, which
        # Groq then rejects outright rather than falling back gracefully.
        # json_schema uses Groq's dedicated structured-output API instead.
        llm = llm.with_structured_output(schema, method="json_schema")

    return llm.with_retry(wait_exponential_jitter=True, stop_after_attempt=5)