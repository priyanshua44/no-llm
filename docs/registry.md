# Model Registry

The registry system provides a central interface for managing model configurations and loading them from files.

## Basic Usage

```python
from no_llm.registry import ModelRegistry
from no_llm.config.enums import ModelMode, ModelCapability

# Initialize registry (optionally with config directory)
registry = ModelRegistry("configs/")

# Get a specific model
model = registry.get_model("gpt-4")

# List models with filters
chat_models = registry.list_models(mode=ModelMode.CHAT)
streaming_models = registry.list_models(capabilities={ModelCapability.STREAMING})
```

## Configuration Inheritance

The registry supports merging custom configurations with built-in ones:

```yaml
# configs/models/gpt-4.yml
# Inherits from built-in GPT-4 configuration
identity:
  id: gpt-4  # Must match built-in model ID
  description: "Custom GPT-4 configuration"  # Overrides built-in description

providers:
  - type: azure  # Override provider
    api_key: $AZURE_API_KEY
    deployment: gpt4

constraints:
  max_input_tokens: 6000  # Override specific constraint
```

```python
# Load built-in models first
registry = ModelRegistry()

# Then load custom configurations
# Custom configs will merge with built-in ones
registry.register_models_from_directory("configs/models")
```

!!! note "Configuration Merging"
    - Custom configurations are merged with built-in ones based on model ID
    - Custom values override built-in values
    - Unspecified fields keep their built-in values
    - This allows partial configuration overrides

## Model Filtering

The registry supports flexible model filtering:

```python
from no_llm.registry import SetFilter

# Match ANY of the capabilities
models = registry.list_models(
    capabilities={ModelCapability.STREAMING, ModelCapability.VISION}
)

# Match ALL capabilities
models = registry.list_models(
    capabilities=SetFilter(
        values={ModelCapability.STREAMING, ModelCapability.FUNCTION_CALLING},
        mode="all"
    )
)

# Filter by multiple criteria
models = registry.list_models(
    mode=ModelMode.CHAT,
    capabilities={ModelCapability.STREAMING},
    privacy_levels={PrivacyLevel.BASIC}
)
```

## Registry Management

```python
# Load from directory
registry = ModelRegistry("configs/")

# Reload all configurations
registry.reload_configurations()

# Register single model
registry.register_model(model_config)

# Remove model
registry.remove_model("gpt-4")
```

See the [Model Configuration](configs/overview.md) documentation for details about configuration formats.