from __future__ import annotations

from pydantic import Field

from no_llm.providers.env_var import EnvVar
from no_llm.providers.openai import OpenAIProvider


class PerplexityProvider(OpenAIProvider):
    """Perplexity provider configuration"""

    type: str = "perplexity"
    name: str = "Perplexity AI"
    api_key: EnvVar[str] = Field(
        default_factory=lambda: EnvVar[str]("$PERPLEXITY_API_KEY"),
        description="Name of environment variable containing API key",
    )
    base_url: str | None = Field(default="https://api.perplexity.ai/", description="Base URL for Perplexity API")
