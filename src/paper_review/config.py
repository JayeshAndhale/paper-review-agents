from dotenv import load_dotenv
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


def get_llm(tier: str):
    model_string = settings.llm_model if tier == "strong" else settings.llm_model_cheap
    provider, model = model_string.split("/", 1)

    if provider == "groq":
        llm = ChatGroq(model=model, api_key=settings.groq_api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return llm.with_retry(wait_exponential_jitter=True, stop_after_attempt=5)