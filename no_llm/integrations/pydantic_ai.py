from __future__ import annotations as _annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

try:
    from anthropic import AsyncAnthropicVertex
    from mistralai_gcp import MistralGoogleCloud
    from pydantic_ai.models import (
        Model,
        ModelRequestParameters,
        StreamedResponse,
    )
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.fallback import FallbackModel
    from pydantic_ai.models.gemini import GeminiModel
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.anthropic import AnthropicProvider as PydanticAnthropicProvider
    from pydantic_ai.providers.azure import AzureProvider as PydanticAzureProvider
    from pydantic_ai.providers.google_vertex import GoogleVertexProvider as PydanticVertexProvider
    from pydantic_ai.providers.google_vertex import VertexAiRegion
    from pydantic_ai.providers.groq import GroqProvider as PydanticGroqProvider
    from pydantic_ai.providers.mistral import MistralProvider as PydanticMistralProvider
    from pydantic_ai.providers.openai import OpenAIProvider as PydanticOpenAIProvider
    from pydantic_ai.settings import ModelSettings as PydanticModelSettings
except ImportError as _import_error:
    msg = (
        "Please install pydantic-ai to use the Pydantic AI integration, "
        'you can use the `pydantic-ai` optional group — `pip install "no_llm[pydantic-ai]"`'
    )
    raise ImportError(msg) from _import_error

from loguru import logger

from no_llm.config.enums import ModelMode
from no_llm.config.model import ModelConfiguration
from no_llm.integrations._utils import pydantic_mistral_gcp_patch
from no_llm.providers import (
    AnthropicProvider,
    AzureProvider,
    DeepseekProvider,
    FireworksProvider,
    GrokProvider,
    GroqProvider,
    MistralProvider,
    OpenAIProvider,
    OpenRouterProvider,
    PerplexityProvider,
    TogetherProvider,
    VertexProvider,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai import usage
    from pydantic_ai.messages import (
        ModelMessage,
        ModelResponse,
    )

ToolName = str


NoLLMModelName = str | ModelConfiguration


@dataclass
class NoLLMModel(Model):
    """A model that uses no_llm under the hood.

    This allows using any no_llm model through the pydantic-ai interface.
    """

    _pydantic_model: Model | None = field(default=None, repr=False)

    def __init__(
        self,
        model_name: NoLLMModelName,
    ):
        self.model = model_name
        self._pydantic_model = None

    @property
    def model_name(self) -> str:
        """The model name."""
        if isinstance(self.model, str):
            return self.model
        return self.model.identity.id

    @property
    def model_config(self) -> ModelConfiguration:
        if isinstance(self.model, str):
            msg = "Model name is not a model configuration"
            raise TypeError(msg)
        return self.model

    @property
    def system(self) -> str | None:  # type: ignore
        """The system / model provider, ex: openai."""
        return "no_llm"

    def _get_pydantic_model(self, model: ModelConfiguration) -> Model:
        """Get the appropriate pydantic-ai model based on no_llm configuration."""
        if self._pydantic_model is not None:
            return self._pydantic_model

        if model.integration_aliases is None:
            msg = "Model must have integration aliases. It is required for pydantic-ai integration."
            raise TypeError(msg)
        if model.integration_aliases.pydantic_ai is None:
            msg = "Model must have a pydantic-ai integration alias. It is required for pydantic-ai integration."
            raise TypeError(msg)

        models: list[Model] = []

        # Try to build a model for each provider variant
        for provider in model.iter():
            try:
                if isinstance(provider, VertexProvider):
                    if "mistral" in model.identity.id:
                        pydantic_mistral_gcp_patch()
                        models.append(
                            MistralModel(
                                model_name=model.integration_aliases.pydantic_ai or model.identity.id,
                                provider=PydanticMistralProvider(
                                    mistral_client=MistralGoogleCloud(  # type: ignore
                                        project_id=provider.project_id, region=provider.current
                                    ),
                                ),
                            )
                        )
                    elif "claude" in model.identity.id:
                        models.append(
                            AnthropicModel(
                                model_name=model.integration_aliases.pydantic_ai or model.identity.id,
                                provider=PydanticAnthropicProvider(
                                    anthropic_client=AsyncAnthropicVertex(  # type: ignore
                                        project_id=provider.project_id, region=provider.current
                                    ),
                                ),
                            )
                        )
                    elif "gemini" in model.identity.id:
                        models.append(
                            GeminiModel(
                                model_name=model.integration_aliases.pydantic_ai or model.identity.id,
                                provider=PydanticVertexProvider(
                                    region=cast(VertexAiRegion, provider.current),
                                    project_id=provider.project_id,
                                ),
                            )
                        )
                elif isinstance(provider, AnthropicProvider):
                    models.append(
                        AnthropicModel(
                            model_name=model.integration_aliases.pydantic_ai or model.identity.id,
                            provider=PydanticAnthropicProvider(
                                api_key=provider.api_key,
                            ),
                        )
                    )
                elif isinstance(provider, MistralProvider):
                    models.append(
                        MistralModel(
                            model_name=model.integration_aliases.pydantic_ai or model.identity.id,
                            provider=PydanticMistralProvider(api_key=provider.api_key),
                        )
                    )
                elif isinstance(provider, GroqProvider):
                    models.append(
                        GroqModel(
                            model_name=model.integration_aliases.pydantic_ai or model.identity.id,
                            provider=PydanticGroqProvider(api_key=provider.api_key),
                        )
                    )
                elif isinstance(
                    provider,
                    OpenAIProvider
                    | DeepseekProvider
                    | PerplexityProvider
                    | FireworksProvider
                    | TogetherProvider
                    | GrokProvider,
                ):
                    models.append(
                        OpenAIModel(
                            model_name=model.integration_aliases.pydantic_ai or model.identity.id,
                            provider=PydanticOpenAIProvider(api_key=provider.api_key, base_url=provider.base_url),
                        )
                    )
                elif isinstance(provider, OpenRouterProvider):
                    models.append(
                        OpenAIModel(
                            model_name=model.integration_aliases.openrouter or model.identity.id,
                            provider=PydanticOpenAIProvider(api_key=provider.api_key, base_url=provider.base_url),
                        )
                    )
                elif isinstance(provider, AzureProvider):
                    models.append(
                        OpenAIModel(
                            model_name=model.integration_aliases.pydantic_ai or model.identity.id,
                            provider=PydanticAzureProvider(api_key=provider.api_key, azure_endpoint=provider.base_url),
                        )
                    )
            except Exception as e:  # noqa: BLE001
                logger.opt(exception=e).warning(f"Failed to create model for provider {type(provider).__name__}")
                continue

        if not models:
            msg = "No models found for pydantic-ai integration"
            raise RuntimeError(msg)

        # Use FallbackModel if we have multiple models, otherwise use the single model
        self._pydantic_model = FallbackModel(*models)
        return self._pydantic_model

    def _get_model_settings(self, model: ModelConfiguration) -> PydanticModelSettings:
        """Get the appropriate pydantic-ai model settings based on no_llm configuration."""
        return PydanticModelSettings(**model.parameters.get_model_parameters().get_parameters()) # type: ignore

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: PydanticModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, usage.Usage]:
        model = self.model
        if not isinstance(model, ModelConfiguration):
            msg = "Model must be a model configuration"
            raise TypeError(msg)

        if model.mode != ModelMode.CHAT:
            msg = "Model does not support chat mode"
            raise RuntimeError(msg)

        model_settings = self._get_model_settings(model)
        pydantic_model = self._get_pydantic_model(model)
        return await pydantic_model.request(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: PydanticModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        model = self.model
        if not isinstance(model, ModelConfiguration):
            msg = "Model must be a model configuration"
            raise TypeError(msg)

        if model.mode != ModelMode.CHAT:
            msg = "Model does not support chat mode"
            raise RuntimeError(msg)

        model_settings = self._get_model_settings(model)
        pydantic_model = self._get_pydantic_model(model)
        async with pydantic_model.request_stream(messages, model_settings, model_request_parameters) as response:
            yield response
