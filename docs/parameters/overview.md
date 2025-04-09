# Parameter System Overview

The no_llm parameter system provides a robust way to configure, validate, and manage model parameters across different LLM providers.

## Core Components

### Parameter Types
- **Variable**: Modifiable values with validation rules
- **Fixed**: Immutable values
- **Unsupported**: Parameters not supported by the model

[Learn more about parameter variants](variant.md)
[Learn more about parameter classes](model_parameters.md)

### Validation
- Range constraints
- Enum values
- Capability requirements
- Fixed value protection

[Learn more about validation](validation.md)

## Quick Example

```python
from no_llm.config.model import ModelConfiguration
from no_llm.config.enums import ModelCapability

# Configure model parameters
model = ModelConfiguration(
    parameters=ConfigurableModelParameters(
        temperature=0.7,           # Variable parameter
        top_p={'fixed': 0.9},     # Fixed parameter
        include_reasoning=True     # Capability-dependent parameter
    )
)

# Validate and get runtime parameters
params = model.parameters.validate_parameters(
    capabilities={ModelCapability.REASONING},
    temperature=0.8  # Will be validated
)
```

!!! tip "Best Practices"
    - Use `ConfigurableModelParameters` for model definitions
    - Use `ModelParameters` for runtime parameter passing
    - Always validate parameters against model capabilities

See the specific documentation sections for detailed information about each aspect of the parameter system.
