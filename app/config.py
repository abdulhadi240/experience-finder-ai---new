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

    # ── Credits API (service-to-service, internal-only) ────────────
    credits_base_url: str = Field(default="https://staging.hiptraveler.com", alias="CREDITS_BASE_URL")
    credits_callback_secret: str | None = Field(default=None, alias="CREDITS_CALLBACK_SECRET")
    # Short on purpose: this call sits in front of every chat message. The 30s
    # used for RAG would let a stalled credits service hang the whole stream.
    credits_timeout_seconds: float = Field(default=3.0, alias="CREDITS_TIMEOUT_SECONDS")
    # Kill switch: set false to fail OPEN during a credits-service incident
    # without a redeploy. Default true — the gate is a security boundary.
    credits_enforce: bool = Field(default=True, alias="CREDITS_ENFORCE")
    # Units to hand back on refund. The reserve response carries no cost field,
    # so 1 unit per billable action is an assumption pending the refund contract.
    credits_refund_cost: int = Field(default=1, alias="CREDITS_REFUND_COST")

    # ── Agent deadlines ───────────────────────────────────────────
    # Without these a hung agent never resolves: the user waits forever and the
    # credit is never handed back. A timeout is a refundable hard-fail.
    agent_timeout_seconds: float = Field(default=120.0, alias="AGENT_TIMEOUT_SECONDS")
    # Max gap between streamed tokens (also covers the wait for the first one).
    agent_token_timeout_seconds: float = Field(default=90.0, alias="AGENT_TOKEN_TIMEOUT_SECONDS")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",        # no prefix
        extra="ignore",       # allow unknown vars
        case_sensitive=True   # interpret env vars exactly as uppercase
    )

settings = Settings()
