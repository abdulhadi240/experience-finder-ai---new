"""
Service layer for OpenAI API interactions
"""
import os
import asyncio
import traceback
from dotenv import load_dotenv
from typing import Dict, Any

from openai import OpenAI
from app.api.validator.models.schemas import QueryClassification
from app.api.validator.config.prompt import SYSTEM_PROMPT


class OpenAIService:
    """Service class to handle OpenAI API calls and query classification"""

    def __init__(self):
        """Initialize OpenAI client"""
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        self.client = OpenAI(api_key=self.api_key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini-search-preview")

    async def classify_query(self, query: str) -> Dict[str, Any]:
        """Classify and rewrite a travel query using OpenAI"""
        try:
            # Run the blocking parse() call in a separate thread to keep FastAPI async
            completion = await asyncio.to_thread(
                lambda: self.client.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"<query>{query}</query>"}
                    ],
                    response_format=QueryClassification  # Use your Pydantic model here
                )
            )

            # Access the parsed Pydantic object
            result: QueryClassification = completion.choices[0].message.parsed

            # Validate queries count based on type
            self._validate_response(result)

            return {
                "type": result.type.value,
                "queries": result.queries
            }

        except Exception as e:
            print(traceback.format_exc())
            raise RuntimeError(f"Error classifying query: {e}") from e

    def _validate_response(self, result: QueryClassification) -> None:
        """Validate that the OpenAI response matches expected format"""
        if result.type.value == "generic" and len(result.queries) != 5:
            raise ValueError(f"Generic query should have 5 sub-queries, got {len(result.queries)}")
        if result.type.value == "specific" and len(result.queries) != 1:
            raise ValueError(f"Specific query should have 1 query, got {len(result.queries)}")
        if result.type.value == "ignore" and len(result.queries) != 0:
            raise ValueError(f"Ignore type should have 0 queries, got {len(result.queries)}")
