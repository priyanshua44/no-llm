from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from pydantic import Field

from no_llm.config import (
    ConfigurableModelParameters,
    IntegrationAliases,
    ModelCapability,
    ModelConfiguration,
    ModelConstraints,
    ModelIdentity,
    ModelMetadata,
    ModelMode,
    ModelPricing,
    ModelProperties,
    ParameterValue,
    ParameterVariant,
    QualityProperties,
    RangeValidation,
    SpeedProperties,
    TokenPrices,
)
from no_llm.config.parameters import NOT_GIVEN, NotGiven
from no_llm.providers import OpenRouterProvider, Provider, VertexProvider


class Gemini25ProConfiguration(ModelConfiguration):
    """Configuration for Gemini 2.5 Pro model"""

    identity: ModelIdentity = ModelIdentity(
        id="gemini-2.5-pro",
        name="Gemini 2.5 Pro",
        version="2024.03",
        description="Latest version of Google's flash Gemini model, with higher quality at all levels",
        creator="Google",
    )

    providers: Sequence[Provider] = [VertexProvider(), OpenRouterProvider()]

    mode: ModelMode = ModelMode.CHAT

    capabilities: set[ModelCapability] = {
        ModelCapability.STREAMING,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.TOOLS,
        ModelCapability.JSON_MODE,
        ModelCapability.SYSTEM_PROMPT,
        ModelCapability.VISION,
        ModelCapability.REASONING,
    }

    constraints: ModelConstraints = ModelConstraints(
        context_window=1000000, max_input_tokens=1000000, max_output_tokens=4096
    )

    properties: ModelProperties | None = ModelProperties(
        speed=SpeedProperties(score=248.6, label="High", description="Average (1-3 seconds)"),
        quality=QualityProperties(score=95.0, label="Very High", description="Very High Quality"),
    )

    metadata: ModelMetadata = ModelMetadata(
        privacy_level=[],
        pricing=ModelPricing(token_prices=TokenPrices(input_price_per_1k=0.0004, output_price_per_1k=0.0008)),
        release_date=datetime(2025, 3, 31),
        data_cutoff_date=datetime(2025, 1, 1),
    )

    integration_aliases: IntegrationAliases | None = IntegrationAliases(
        pydantic_ai="gemini-2.5-pro-exp-03-25",
        litellm="vertex_ai/gemini-2.5-pro-exp-03-25",
        langfuse="gemini-2.5-pro",
        lmarena="gemini-2.5-pro-exp-03-25",
        openrouter="google/gemini-2.5-pro-preview-03-25:free",
    )

    class Parameters(ConfigurableModelParameters):
        model_config = ConfigurableModelParameters.model_config
        temperature: ParameterValue[float | NotGiven] = Field(
            default_factory=lambda: ParameterValue[float | NotGiven](
                variant=ParameterVariant.VARIABLE,
                value=0.0,
                validation_rule=RangeValidation(min_value=0.0, max_value=2.0),
            )
        )
        top_p: ParameterValue[float | NotGiven] = Field(
            default_factory=lambda: ParameterValue[float | NotGiven](
                variant=ParameterVariant.VARIABLE,
                value=0.8,
                validation_rule=RangeValidation(min_value=0.0, max_value=1.0),
            )
        )
        top_k: ParameterValue[int | NotGiven] = Field(
            default_factory=lambda: ParameterValue[int | NotGiven](
                variant=ParameterVariant.VARIABLE, value=20, validation_rule=RangeValidation(min_value=1, max_value=100)
            )
        )
        frequency_penalty: ParameterValue[float | NotGiven] = Field(
            default_factory=lambda: ParameterValue[float | NotGiven](
                variant=ParameterVariant.VARIABLE,
                value=0.3,
                validation_rule=RangeValidation(min_value=0.0, max_value=1.0),
            )
        )
        presence_penalty: ParameterValue[float | NotGiven] = Field(
            default_factory=lambda: ParameterValue[float | NotGiven](
                variant=ParameterVariant.VARIABLE,
                value=0.6,
                validation_rule=RangeValidation(min_value=0.0, max_value=1.0),
            )
        )
        max_tokens: ParameterValue[int | NotGiven] = Field(
            default_factory=lambda: ParameterValue[int | NotGiven](
                variant=ParameterVariant.VARIABLE,
                value=4096,
                validation_rule=RangeValidation(min_value=1, max_value=100000),
            )
        )
        stop: ParameterValue[list[str] | NotGiven] = Field(
            default_factory=lambda: ParameterValue[list[str] | NotGiven](
                variant=ParameterVariant.VARIABLE, value=NOT_GIVEN
            )
        )
        seed: ParameterValue[int | NotGiven] = Field(
            default_factory=lambda: ParameterValue[int | NotGiven](
                variant=ParameterVariant.UNSUPPORTED, value=NOT_GIVEN
            )
        )

    parameters: ConfigurableModelParameters = Field(default_factory=Parameters)
