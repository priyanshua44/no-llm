from __future__ import annotations

import pkgutil
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

import yaml
from loguru import logger

from no_llm.config import ModelCapability, ModelConfiguration, ModelMode, PrivacyLevel
from no_llm.errors import (
    ConfigurationLoadError,
    ModelNotFoundError,
)
from no_llm.models import __all__ as model_configs

if TYPE_CHECKING:
    from collections.abc import Iterator

T = TypeVar("T")


@dataclass
class SetFilter(Generic[T]):
    values: set[T]
    mode: Literal["all", "any"] = "any"


class ModelRegistry:
    """Registry that manages both model configurations and providers"""

    def __init__(self, config_dir: str | Path | None = None):
        self._models: dict[str, ModelConfiguration] = {}
        self._config_dir = Path(config_dir) if config_dir else None

        logger.debug("Initializing ModelRegistry")

        # Load built-in models first
        self._register_builtin_models()

        # Then load any custom configurations if directory provided
        if config_dir:
            logger.debug(f"Using config directory: {config_dir}")
            self._load_configurations()

    def _register_builtin_models(self) -> None:
        """Register built-in model configurations from Python classes"""
        logger.debug("Loading built-in model configurations")

        # Import all model configuration classes
        for config_class_name in model_configs:
            # Dynamically discover and import from all submodules
            from no_llm import models

            for module_info in pkgutil.iter_modules(models.__path__):
                try:
                    module = import_module(f".models.{module_info.name}", package="no_llm")
                    if hasattr(module, config_class_name):
                        config_class: type[ModelConfiguration] = getattr(module, config_class_name)
                        model_config = config_class()  # type: ignore
                        self.register_model(model_config)
                        logger.debug(f"Registered model configuration: {config_class_name}")
                        break
                except ImportError as e:
                    logger.debug(f"Could not import module {module_info.name}: {e}")
                    continue

    def _find_yaml_file(self, base_path: Path, name: str) -> Path:
        """Try both .yml and .yaml extensions"""
        for ext in [".yml", ".yaml"]:
            path = base_path / f"{name}{ext}"
            if path.exists():
                return path
        return base_path / f"{name}.yml"  # Default to .yml if neither exists

    def _merge_configs(self, base: dict, override: dict) -> dict:
        """Deep merge two configuration dictionaries"""
        merged = base.copy()
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                merged[key] = self._merge_configs(base[key], value)
            else:
                merged[key] = value
        return merged

    def _load_model_config(self, model_id: str) -> ModelConfiguration:
        """Load a model configuration from YAML, using builtin as base if it exists"""
        if not self._config_dir:
            msg = "No config directory set"
            raise NotADirectoryError(msg)

        model_file = self._find_yaml_file(self._config_dir / "models", model_id)
        logger.debug(f"Loading model config from: {model_file}")

        try:
            with open(model_file) as f:
                config = yaml.safe_load(f)
            logger.debug(f"Loaded YAML config: {config}")

            # If this is overriding a builtin model, use the builtin as base
            if model_id in self._models:
                logger.debug(f"Found existing model {model_id}, merging configs")
                base_model = self._models[model_id]
                base_config = base_model.dict()
                merged_config = self._merge_configs(base_config, config)
                logger.debug(f'Merged config description: {merged_config["identity"]["description"]}')
                return ModelConfiguration(**merged_config)

            return ModelConfiguration(**config)
        except Exception as e:
            logger.opt(exception=e).error(f"Error loading config from {model_file}: {e}")
            raise ConfigurationLoadError(str(model_file), e) from e

    def register_models_from_directory(self, models_dir: Path | str) -> None:
        """Load and register all model configurations from a directory"""
        models_dir = Path(models_dir)
        if not models_dir.exists():
            logger.warning(f"Models directory not found: {models_dir}")
            return

        logger.debug(f"Loading models from {models_dir}")
        logger.debug(f"Models directory contents: {list(models_dir.iterdir())}")

        # Change the glob pattern to match both .yml and .yaml files
        yaml_files = []
        for ext in ["*.yml", "*.yaml"]:
            yaml_files.extend(list(models_dir.glob(ext)))

        logger.debug(f"Found {len(yaml_files)} YAML files: {[f.name for f in yaml_files]}")

        for model_file in yaml_files:
            model_id = model_file.stem
            try:
                logger.debug(f"Loading model config from file: {model_file}")
                with open(model_file) as f:
                    config = yaml.safe_load(f)
                logger.debug(f"Loaded YAML config: {config}")

                # If this is overriding a builtin model, use the builtin as base
                if model_id in self._models:
                    logger.debug(f"Found existing model {model_id}, merging configs")
                    base_model = self._models[model_id]
                    base_config = base_model.model_dump()
                    merged_config = self._merge_configs(base_config, config)
                    model = ModelConfiguration(**merged_config)
                else:
                    model = ModelConfiguration(**config)

                self.register_model(model)
                logger.debug(f"Registered model: {model_id} with description: {model.identity.description}")
            except Exception as e:  # noqa: BLE001
                logger.opt(exception=e).error(f"Error loading model {model_id}")

    def _load_configurations(self) -> None:
        """Load all configurations from the config directory"""
        if not self._config_dir:
            logger.warning("No config directory set")
            return

        # Load models
        models_dir = self._config_dir / "models"
        logger.debug(f"Models directory path: {models_dir}")
        logger.debug(f"Models directory exists: {models_dir.exists()}")
        if models_dir.exists():
            logger.debug(f"Models directory contents: {list(models_dir.iterdir())}")
        self.register_models_from_directory(models_dir)

    def register_model(self, model: ModelConfiguration) -> None:
        """Register a new model, overriding if it already exists"""
        if model.identity.id in self._models:
            logger.info(f"Overriding existing model configuration: {model.identity.id}")

        self._models[model.identity.id] = model
        logger.debug(f"Registered model: {model.identity.id}")

    def get_model(self, model_id: str) -> ModelConfiguration:
        """Get a specific model by ID"""
        if model_id not in self._models:
            logger.error(f"Model {model_id} not found")
            raise ModelNotFoundError(model_id)
        return self._models[model_id]

    def list_models(
        self,
        *,
        provider: str | None = None,
        capabilities: set[ModelCapability] | SetFilter[ModelCapability] | None = None,
        privacy_levels: set[PrivacyLevel] | SetFilter[PrivacyLevel] | None = None,
        mode: ModelMode | None = None,
    ) -> Iterator[ModelConfiguration]:
        """List models matching the given criteria"""
        # Convert simple sets to SetFilters with default "any" mode
        if isinstance(capabilities, set):
            capabilities = SetFilter(capabilities)
        if isinstance(privacy_levels, set):
            privacy_levels = SetFilter(privacy_levels)

        logger.debug(
            f"Listing models with filters: provider={provider}, capabilities={capabilities}, "
            f"mode={mode}, privacy_levels={privacy_levels}"
        )

        for model in self._models.values():
            if provider and not any(p.type == provider for p in model.providers):
                continue

            if capabilities:
                model_caps = set(model.capabilities)
                if capabilities.mode == "any":
                    if not (model_caps & capabilities.values):
                        continue
                elif not (capabilities.values <= model_caps):
                    continue

            if privacy_levels:
                model_privacy = set(model.metadata.privacy_level)
                if privacy_levels.mode == "any":
                    if not (model_privacy & privacy_levels.values):
                        continue
                elif not (privacy_levels.values <= model_privacy):
                    continue

            if mode and model.mode != mode:
                continue

            yield model

    def remove_model(self, model_id: str) -> None:
        """Remove a model from the registry"""
        if model_id not in self._models:
            logger.error(f"Cannot remove: model {model_id} not found")
            raise ModelNotFoundError(model_id)
        del self._models[model_id]
        logger.debug(f"Removed model: {model_id}")

    def reload_configurations(self) -> None:
        """Reload all configurations from disk"""
        logger.debug("Reloading all configurations")
        self._models.clear()
        self._register_builtin_models()
        self._load_configurations()
