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
    PrivacyLevel,
    QualityProperties,
    RangeValidation,
    SpeedProperties,
    TokenPrices,
)
from no_llm.config.parameters import NOT_GIVEN, NotGiven
from no_llm.providers import AnthropicProvider, BedrockProvider, OpenRouterProvider, Provider, VertexProvider


class Claude35SonnetV2Configuration(ModelConfiguration):
    """Configuration for Claude 3.5 Sonnet v2 model"""

    identity: ModelIdentity = ModelIdentity(
        id="claude-3.5-sonnet-v2",
        name="Claude 3.5 Sonnet",
        version="2024.02",
        description="Latest version of Claude 3.5 Sonnet optimized for enterprise workloads. Features improved vision processing and function calling while maintaining an optimal balance of speed and quality.",
        creator="Anthropic",
    )

    mode: ModelMode = ModelMode.CHAT
    providers: Sequence[Provider] = [VertexProvider(), BedrockProvider(), AnthropicProvider(), OpenRouterProvider()]
    capabilities: set[ModelCapability] = {
        ModelCapability.STREAMING,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.TOOLS,
        ModelCapability.JSON_MODE,
        ModelCapability.SYSTEM_PROMPT,
        ModelCapability.VISION,
    }

    constraints: ModelConstraints = ModelConstraints(
        context_window=200000, max_input_tokens=200000, max_output_tokens=8192
    )

    properties: ModelProperties | None = ModelProperties(
        speed=SpeedProperties(score=81.5, label="Average", description="Average (1-3 seconds)"),
        quality=QualityProperties(score=78.0, label="High", description="High Quality"),
    )

    metadata: ModelMetadata = ModelMetadata(
        privacy_level=[PrivacyLevel.BASIC],
        pricing=ModelPricing(token_prices=TokenPrices(input_price_per_1k=0.05, output_price_per_1k=0.15)),
        release_date=datetime(2024, 2, 1),
        data_cutoff_date=datetime(2024, 8, 1),
    )

    integration_aliases: IntegrationAliases | None = IntegrationAliases(
        # pydantic_ai="claude-3-5-sonnet-v2",
        # litellm="vertex_ai/claude-3-5-sonnet-v2@20241022",
        # langfuse="claude-3-5-sonnet-v2"
        pydantic_ai="claude-3.5-sonnet-v2",
        litellm="vertex_ai/claude-3-5-sonnet-v2@20240229",
        langfuse="claude-3.5-sonnet-v2",
        lmarena="claude-3-5-sonnet-v2-20240229",
        openrouter="anthropic/claude-3.5-sonnet-v2:free",
    )

    class Parameters(ConfigurableModelParameters):
        model_config = ConfigurableModelParameters.model_config
        temperature: ParameterValue[float | NotGiven] = Field(
            default_factory=lambda: ParameterValue[float | NotGiven](
                variant=ParameterVariant.VARIABLE,
                value=0.0,
                validation_rule=RangeValidation(min_value=0.0, max_value=1.0),
            )
        )
        top_p: ParameterValue[float | NotGiven] = Field(
            default_factory=lambda: ParameterValue[float | NotGiven](variant=ParameterVariant.FIXED, value=1.0)
        )
        top_k: ParameterValue[int | NotGiven] = Field(
            default_factory=lambda: ParameterValue[int | NotGiven](variant=ParameterVariant.FIXED, value=40)
        )
        frequency_penalty: ParameterValue[float | NotGiven] = Field(
            default_factory=lambda: ParameterValue[float | NotGiven](
                variant=ParameterVariant.UNSUPPORTED, value=NOT_GIVEN
            )
        )
        presence_penalty: ParameterValue[float | NotGiven] = Field(
            default_factory=lambda: ParameterValue[float | NotGiven](
                variant=ParameterVariant.UNSUPPORTED, value=NOT_GIVEN
            )
        )
        max_tokens: ParameterValue[int | NotGiven] = Field(
            default_factory=lambda: ParameterValue[int | NotGiven](variant=ParameterVariant.FIXED, value=8192)
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
