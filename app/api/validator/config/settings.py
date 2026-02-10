"""
Configuration and settings
"""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    perplexity_api_key: str = os.getenv("PERPLEXITY_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    
    supabase_url: str | None = os.getenv("SUPABASE_URL")
    supabase_key: str | None = os.getenv("SUPABASE_KEY")
    supabase_project_id: str | None = os.getenv("SUPABASE_PROJECT_ID")
    supabase_service_role_key: str | None = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    google_maps_api_key: str | None = os.getenv("GOOGLE_MAPS_API_KEY")
    
    # Model configurations
    openai_model: str = "gpt-4o"
    openai_research_model: str = "gpt-4o"
    perplexity_model: str = "sonar-pro"
    gemini_model: str = "gemini-1.5-pro"
    
    # Research settings
    max_research_tokens: int = 2000
    research_timeout: int = 60  # seconds
    min_validation_score: int = 2
    
    # Priority sources
    priority_sources: list = ["tripadvisor.com", "yelp.com"]
    
    class Config:
        env_file = ".env"
        extra = "ignore"  


settings = Settings()