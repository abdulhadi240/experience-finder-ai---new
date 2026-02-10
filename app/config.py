from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """Loads and validates application settings from environment variables."""

    # Required keys
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    model: str = "gpt-4o"
    zep_api_key: str = Field(alias="ZEP_API_KEY")

    # Optional keys
    supabase_url: str | None = Field(default=None, alias="SUPABASE_URL")
    supabase_key: str | None = Field(default=None, alias="SUPABASE_KEY")
    supabase_project_id: str | None = Field(default=None, alias="SUPABASE_PROJECT_ID")
    supabase_service_role_key: str | None = Field(default=None, alias="SUPABASE_SERVICE_ROLE_KEY")
    google_maps_api_key: str | None = Field(default=None, alias="GOOGLE_MAPS_API_KEY")
    perplexity_api_key: str | None = Field(default=None, alias="PERPLEXITY_API_KEY")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",        # no prefix
        extra="ignore",       # allow unknown vars
        case_sensitive=True   # interpret env vars exactly as uppercase
    )

settings = Settings()
