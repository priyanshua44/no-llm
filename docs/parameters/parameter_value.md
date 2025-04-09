# Parameter Value API

The `ParameterValue` class provides a flexible way to define and validate model parameters.

!!! info "Custom Validation vs Pydantic"
    While no_llm uses Pydantic for model definitions, parameter validation uses a custom system. This allows parameter configurations to be easily serialized to YAML/JSON and used in any environment, not just Python. The validation rules become part of the configuration data itself, making it portable and platform-independent.

## Core Methods

### Creation

```python
from no_llm.config.parameters import ParameterValue, ParameterVariant
from no_llm.config.enums import ModelCapability

# Standard creation
param = ParameterValue(
    variant=ParameterVariant.VARIABLE,
    value=0.7,
    validation_rule=RangeValidation(min_value=0.0, max_value=2.0)
)

# Helper for variable parameters
param = ParameterValue.create_variable(
    value=True,
    required_capability=ModelCapability.REASONING
)
```

### Value Access and Validation

```python
# Get value
value = param.get()  # Returns value or 'UNSUPPORTED'

# Validate new value
param.validate_new_value(0.8, "temperature")

# Check capabilities
param.check_capability({ModelCapability.STREAMING})

# Check variant type
param.is_variable()    # True if variable
param.is_fixed()       # True if fixed
param.is_unsupported() # True if unsupported
```

## Validation Rules

```python
# Range validation
RangeValidation(min_value=0.0, max_value=2.0)

# Enum validation
EnumValidation(allowed_values=["low", "medium", "high"])
```

!!! note "Parameter Value Behavior"
    - Fixed parameters cannot be modified after creation
    - Unsupported parameters always return 'UNSUPPORTED'
    - Variable parameters require validation rule checks
