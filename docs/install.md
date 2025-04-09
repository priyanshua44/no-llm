# Installation

## Basic Installation

Install no_llm using uv:

```bash
uv pip install no_llm
```

## Optional Dependencies

no_llm provides optional integrations that can be installed with extras:

```bash
# Install with Pydantic AI support
uv pip install "no_llm[pydantic-ai]"
```

## Development Installation

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/pedro/no_llm
cd no_llm
uv pip install -e ".[pydantic-ai]"
```
