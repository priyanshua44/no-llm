import os
from typing import Any, Generic, TypeVar

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

T = TypeVar("T")


class EnvVar(Generic[T]):
    def __init__(self, var_name: str) -> None:
        if not var_name.startswith("$"):
            msg = "Environment variable name must start with $"
            raise ValueError(msg)
        self.var_name = var_name

    def __get__(self, instance: Any, owner: Any) -> str:
        env_name = self.var_name[1:]  # Remove '$'
        env_value = os.getenv(env_name)
        return env_value if env_value is not None else self.var_name

    def __repr__(self) -> str:
        return self.__get__(None, None)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(cls),
                    core_schema.str_schema(),
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: x.var_name if isinstance(x, cls) else x
            ),
        )

    def is_valid(self) -> bool:
        """Check if environment variable has a real value set"""
        return self.__get__(None, None) != self.var_name
