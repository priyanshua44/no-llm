<div align="center">
  <h1>no_llm</h1>
  <em>Standard Interface for Large Language Models</em>
</div>

<div align="center">
  <a href="https://github.com/pedro/no_llm/actions/workflows/ci.yml"><img src="https://github.com/pedro/no_llm/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.python.org/pypi/no_llm"><img src="https://img.shields.io/pypi/v/no_llm.svg" alt="PyPI"></a>
  <a href="https://github.com/pedro/no_llm"><img src="https://img.shields.io/pypi/pyversions/no_llm.svg" alt="versions"></a>
  <a href="https://github.com/pedro/no_llm/blob/main/LICENSE"><img src="https://img.shields.io/github/license/pedro/no_llm.svg" alt="license"></a>
</div>

---

`no/llm` is a Python library that provides a unified interface for working with LLMs, with built-in support for model configuration, parameter validation, and provider management.

## Quick Install

```bash
uv pip install "no_llm[pydantic-ai]"
```

## Quick Example with Pydantic AI

```python
from no_llm.integrations.pydantic_ai import no_llmModel
from no_llm.registry import ModelRegistry

# Get model from registry
registry = ModelRegistry()
model = registry.get_model("gpt-4o")
no_llm_model = no_llmModel(model)

# Use with Pydantic AI
agent = Agent(no_llm_model)
result = await agent.run("What is the capital of France?")
print(result.data)
```

## Why no_llm?

* __Provider Agnostic__: Support for OpenAI, Anthropic, Google, Mistral, Groq, and more through a single interface

* __Built-in Validation__: Type-safe parameter validation and capability checking

* __Provider Fallbacks__: Automatic fallback between providers and data centers

* __Configuration System__: YAML-based model configurations with inheritance support

* __Model Registry__: Central management of models with capability-based filtering

* __Integration Ready__: Works with Pydantic AI, and more frameworks coming soon

!!! tip "Free Testing"
    Get a free API key from [OpenRouter](https://openrouter.ai/keys) to test various models without individual provider accounts.

## Next Steps

- [Configuration Guide](configs/overview.md)
- [Parameter System](parameters/overview.md)
- [Provider Documentation](providers/overview.md)
- [Registry System](registry.md)
