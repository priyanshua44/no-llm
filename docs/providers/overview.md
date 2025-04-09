# Providers

no_llm supports multiple LLM providers through a flexible provider system. Each provider can be configured with its own authentication and endpoint settings.

## Supported Providers

| Provider | Type | Required Configuration |
|----------|------|----------------------|
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` |
| Azure OpenAI | `azure` | `AZURE_API_KEY`, `AZURE_BASE_URL` |
| Google Vertex AI | `vertex` | `VERTEX_PROJECT_ID` |
| Mistral AI | `mistral` | `MISTRAL_API_KEY` |
| Groq | `groq` | `GROQ_API_KEY` |
| Perplexity | `perplexity` | `PERPLEXITY_API_KEY` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` |
| Together AI | `together` | `TOGETHER_API_KEY` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` |
| Grok | `grok` | `GROK_API_KEY` |
| Fireworks | `fireworks` | `FIREWORKS_API_KEY` |
| AWS Bedrock | `bedrock` | `BEDROCK_REGION` |

## Environment Variables

no_llm uses environment variables for provider configuration. The `EnvVar` class provides a secure way to handle API keys and other sensitive information:

```python
from no_llm.providers import OpenAIProvider

provider = OpenAIProvider(
    api_key="$OPENAI_API_KEY",  # Will load from environment
    base_url=None  # Optional API endpoint override
)
```

!!! note "Environment Variable Format"
    Environment variables must be prefixed with `$` in the configuration. The actual environment variable name will not include the `$`.

## Provider Configuration

Providers can be configured in Python or YAML:

```python
from no_llm.config import ModelConfiguration
from no_llm.providers import OpenAIProvider, AnthropicProvider

# Multiple providers for fallback
model = ModelConfiguration(
    providers=[
        OpenAIProvider(api_key="$OPENAI_API_KEY"),
        AnthropicProvider(api_key="$ANTHROPIC_API_KEY")
    ]
)


```yaml
providers:
  - type: openai
    api_key: ${OPENAI_API_KEY}
  - type: anthropic
    api_key: ${ANTHROPIC_API_KEY}
```

## Provider Features

Each provider implementation includes:
- Environment variable handling
- API endpoint configuration
- Parameter mapping
- Error handling

!!! tip "Provider Selection"
    Models can have multiple providers configured for:
    - Fallback handling
    - Load balancing
    - Cost optimization
    - Geographic distribution

See the specific provider documentation for detailed configuration options and features.

## Provider Iteration

Providers support iteration over their variants (e.g., different locations for cloud providers):

```python
for provider in model.iter():
    try:
        ...
        break  # Success, stop trying other providers
    except Exception:
        continue  # Try next provider/variant
```

Each provider implements the `iter()` method to define its iteration behavior:
- Base providers yield just themselves
- Cloud providers yield variants for each location
- Custom providers can implement their own iteration logic
