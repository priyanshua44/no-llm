# Parameter Validation

no_llm provides a robust parameter validation system that ensures model parameters are used correctly and safely.

## Validation Modes

no_llm supports three validation modes for handling invalid parameter values:

| Mode | Description |
|------|-------------|
| `ERROR` | Raises an exception when validation fails |
| `WARN` | Logs a warning and keeps the original parameter value |
| `CLAMP` | Clamps values to the nearest valid value within range |

```python
from no_llm.settings import settings as no_llm_settings
from no_llm.settings import ValidationMode

# Set validation mode
no_llm_settings.validation_mode = ValidationMode.CLAMP

# Now out-of-range values will be clamped instead of raising errors
parameters.validate_parameters(temperature=2.5)  # Will be clamped to 2.0
```

## Types of Validation

### Range Validation

Ensures numeric parameters stay within defined bounds:

```python
# Will raise InvalidRangeError in ERROR mode
# Will clamp to 2.0 in CLAMP mode
parameters.validate_parameters(temperature=2.5)

# Will raise InvalidRangeError in ERROR mode
# Will clamp to 0.0 in CLAMP mode
parameters.validate_parameters(temperature=-0.5)
```

### Enum Validation

Ensures parameters only take predefined values:

```python
# Will raise InvalidEnumError
parameters.validate_parameters(
    reasoning_effort="very_high"  # Only low/medium/high allowed
)
```

### Capability-Based Validation

Some parameters require specific model capabilities:

```python
# Check if parameters are valid for model capabilities
validated_params = parameters.validate_parameters(
    capabilities={ModelCapability.STREAMING},
    include_reasoning=True  # Requires REASONING capability
)
```

## Handling Unsupported Parameters

When validating parameters, you can control how unsupported parameters are handled:

```python
# Silently drop unsupported parameters (default)
validated_params = parameters.validate_parameters(
    capabilities={ModelCapability.STREAMING},
    include_reasoning=True,  # Will be dropped if REASONING not supported
    drop_unsupported=True
)

# Raise error for unsupported parameters
try:
    validated_params = parameters.validate_parameters(
        capabilities={ModelCapability.STREAMING},
        include_reasoning=True,
        drop_unsupported=False  # Will raise UnsupportedParameterError
    )
except UnsupportedParameterError as e:
    print(f"Parameter {e.param_name} requires capability: {e.required_capability}")
```

!!! tip "Validation Best Practices"
    - Use `ERROR` mode during development to catch issues early
    - Use `CLAMP` mode in production for better user experience
    - Always check model capabilities before setting parameters
    - Handle unsupported parameters explicitly in critical code paths
