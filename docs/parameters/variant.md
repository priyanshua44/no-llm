# Parameter Variants

Parameters in no_llm can exist in three variants that define their behavior and configurability.

## Available Variants

| Variant | Description |
|---------|-------------|
| `FIXED` | Parameter value cannot be changed |
| `VARIABLE` | Parameter value can be modified within constraints |
| `UNSUPPORTED` | Parameter is not supported by the model |

## Python Usage

```python
from no_llm.config.parameters import ParameterValue, ParameterVariant

# Fixed parameter
temp_fixed = ParameterValue(
    variant=ParameterVariant.FIXED,
    value=0.7
)

# Variable parameter
temp_variable = ParameterValue(
    variant=ParameterVariant.VARIABLE,
    value=0.7
)

# Unsupported parameter
temp_unsupported = ParameterValue(
    variant=ParameterVariant.UNSUPPORTED
)
```

## YAML Configuration

Parameters can be configured in YAML using different formats:

### Simple Format
```yaml
# Fixed value (shorthand)
temperature: 0.7

# Unsupported parameter (shorthand)
frequency_penalty: unsupported
```

### Explicit Format
```yaml
# Fixed parameter
temperature:
  variant: fixed
  value: 0.7

# Variable parameter
top_p:
  variant: variable
  value: 0.9

# Unsupported parameter
logprobs:
  variant: unsupported
```

### Variable with Validation
```yaml
# Variable with range validation
temperature:
  variant: variable
  value: 0.7
  range: [0.0, 2.0]

# Variable with enum validation
reasoning_effort:
  variant: variable
  value: "medium"
  values: ["low", "medium", "high"]
```

!!! tip "Choosing Variants"
    - Use `FIXED` for parameters that should never change
    - Use `VARIABLE` for parameters that users can modify
    - Use `UNSUPPORTED` to explicitly mark unavailable features
    - Default to `VARIABLE` when in doubt
