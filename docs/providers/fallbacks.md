# Provider Fallbacks

no_llm supports multiple levels of fallback to ensure reliable model access:

## Data Center Fallback

Cloud providers can have multiple data centers configured for automatic failover:

```python
from no_llm.providers import VertexProvider

provider = VertexProvider(
    project_id="my-project",
    locations=["us-central1", "europe-west1", "asia-east1"]
)

# Automatically tries each location
for variant in provider.iter():
    try:
        ...
        break  # Success, stop trying other locations
    except Exception:
        continue  # Try next location
```

## Provider Fallback

Models can be configured with multiple providers for service redundancy:

```python
from no_llm.config import ModelConfiguration
from no_llm.providers import OpenAIProvider, AnthropicProvider, AzureProvider

model = ModelConfiguration(
    providers=[
        OpenAIProvider(api_key="$OPENAI_API_KEY"),
        AzureProvider(api_key="$AZURE_API_KEY"),
        AnthropicProvider(api_key="$ANTHROPIC_API_KEY")
    ]
)

# Try each provider in sequence
for provider in model.iter():
    try:
        ...
        break  # Success, stop trying other providers
    except Exception:
        continue  # Try next provider
```

## Combined Fallback Strategy

The iteration system combines both levels of fallback:

```python
model = ModelConfiguration(
    providers=[
        VertexProvider(
            project_id="my-project",
            locations=["us-central1", "europe-west1"]
        ),
        OpenAIProvider(api_key="$OPENAI_API_KEY"),
        AnthropicProvider(api_key="$ANTHROPIC_API_KEY")
    ]
)

# Tries each provider and their variants
for provider in model.iter():
    try:
        ...
        break
    except Exception:
        continue
```

!!! tip "Fallback Order"
    - Providers are tried in the order they are configured
    - Each provider's data centers are tried in the order specified
    - Use `reset_provider_iteration()` to start over from the first provider

!!! note "Environment Validation"
    Providers with missing environment variables (e.g., unset API keys) are automatically skipped during iteration.
