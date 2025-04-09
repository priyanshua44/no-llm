from __future__ import annotations

from no_llm.providers import EnvVar
from no_llm.providers.base import Provider, ParameterMapping
from no_llm.providers.vertex import VertexProvider


class TestProvider(Provider):
    """Test provider for unit tests"""
    type: str = "test"
    name: str = "Test Provider"
    api_key: EnvVar[str] = EnvVar[str]("$TEST_API_KEY")


def test_provider_reset_iterator():
    provider = TestProvider()
    provider._iterator_index = 5
    provider.reset_iterator()
    assert provider._iterator_index == 0


def test_provider_map_parameters():
    provider = TestProvider()
    
    # Add some parameter mappings
    provider.parameter_mappings = {
        "temperature": ParameterMapping(name="temp", supported=True),
        "max_tokens": ParameterMapping(name="max_output_tokens", supported=True),
        "unsupported_param": ParameterMapping(name="unused", supported=False),
    }
    
    params = {
        "temperature": 0.7,
        "max_tokens": 100,
        "unsupported_param": "test",
        "unmapped_param": "keep",
        "direct_param": "direct"
    }
    
    mapped = provider.map_parameters(params)
    
    assert mapped == {
        "temp": 0.7,
        "max_output_tokens": 100,
        "unmapped_param": "keep",
        "direct_param": "direct"
    }