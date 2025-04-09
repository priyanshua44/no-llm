import os
from enum import Enum

from pydantic import BaseModel, Field


class ValidationMode(Enum):
    """Validation mode for model configurations"""

    ERROR = "error"
    WARN = "warn"
    CLAMP = "clamp"


class Settings(BaseModel):
    """Global settings for the no_llm library"""

    validation_mode: ValidationMode = Field(
        default=ValidationMode(os.getenv("NO_LLM_VALIDATION_MODE", ValidationMode.WARN.value)),
        description="Validation mode for model configurations",
    )


settings = Settings()
