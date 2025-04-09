# Model Parameters

no_llm provides two parameter classes: `ModelParameters` for runtime use and `ConfigurableModelParameters` for model configuration.

!!! info "Parameters vs Configurable Parameters"
    - `ModelParameters`: Used at runtime for parameter passing, with simple value validation
    - `ConfigurableModelParameters`: Used in model configurations, with full validation rules, variants, and capability checks


## Supported Parameters

### Sampling Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|----------|
| `temperature` | float | Controls randomness in generation | 1.0 |
| `top_p` | float | Nucleus sampling threshold | 1.0 |
| `top_k` | int \| None | Top-k sampling threshold | None |

### Penalty Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|----------|
| `frequency_penalty` | float \| None | Penalty for token frequency | 0.0 |
| `presence_penalty` | float \| None | Penalty for token presence | 0.0 |
| `logit_bias` | dict[str, float] \| None | Token biasing dictionary | None |

### Output Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|----------|
| `max_tokens` | int \| None | Maximum tokens to generate | None |
| `stop` | list[str] \| None | Stop sequences | None |
| `logprobs` | int \| None | Number of logprobs to return | None |
| `top_logprobs` | int \| None | Most likely tokens to return | None |
| `seed` | int \| None | Random seed for reproducibility | None |

### Request Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|----------|
| `timeout` | float \| None | Request timeout in seconds | None |

### Reasoning Parameters

| Parameter | Type | Description | Default | Required Capability |
|-----------|------|-------------|----------|-------------------|
| `include_reasoning` | bool \| None | Include reasoning steps | False | `REASONING` |
| `reasoning_effort` | "low"\|"medium"\|"high" \| None | Reasoning level | None | `REASONING` |

## Usage Example

```python
# Runtime parameter passing
from no_llm.config.parameters import ModelParameters

params = ModelParameters(
    temperature=0.7,
    max_tokens=100,
    include_reasoning=True
)

# Model configuration
from no_llm.config.parameters import ConfigurableModelParameters

config_params = ConfigurableModelParameters(
    temperature=ParameterValue(
        variant=ParameterVariant.VARIABLE,
        value=0.7,
        validation_rule=RangeValidation(min_value=0.0, max_value=2.0)
    )
)
```

!!! note "NOT_GIVEN vs None"
    In `ModelParameters`, unset parameters use the special value `'NOT_GIVEN'` instead of `None`. This allows distinguishing between explicitly setting a parameter to `None` and not setting it at all.

## Parameter Conversion Flow

ConfigurableModelParameters can be converted to ModelParameters through validation:

```python
# Model configuration with validation rules
config_params = ConfigurableModelParameters(
    temperature=ParameterValue(
        variant=ParameterVariant.VARIABLE,
        value=0.7,
        validation_rule=RangeValidation(min_value=0.0, max_value=2.0)
    )
)

# Get validated runtime parameters
validated_params = config_params.validate_parameters(
    capabilities={ModelCapability.STREAMING},
    temperature=0.8  # Will be validated against rules
)

# Create ModelParameters from validated values
model_params = ModelParameters(**validated_params)
```

!!! warning "Extensibility"
    The parameter system can be extended through inheritance to support additional parameters
    Check out the [Parameter Customization](../configs/inheritance.md) section for more information.