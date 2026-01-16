"""Model registry for managing embedding providers."""

import re
from pathlib import Path
from typing import Literal

import yaml
from dotenv import find_dotenv, load_dotenv

from color_perception_bench.providers.base import (
    BatchConfig,
    EndpointConfig,
    ProviderConfig,
)
from color_perception_bench.providers.local import LocalProvider
from color_perception_bench.providers.openai_compatible import OpenAICompatibleProvider

# Load .env file - search up the directory tree to find it
load_dotenv(find_dotenv(usecwd=True))

REGISTRY_FILE = Path("models.yaml")
VALID_BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

ProviderType = Literal["local", "openai_compatible"]


def sanitize_model_name(name: str) -> str:
    """Sanitize model name for use in filenames."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name.lower())


def load_registry() -> dict:
    """Load the model registry from YAML file."""
    if not REGISTRY_FILE.exists():
        return {"models": {}}

    with open(REGISTRY_FILE) as f:
        data = yaml.safe_load(f) or {}

    if "models" not in data:
        data["models"] = {}

    return data


def save_registry(registry: dict) -> None:
    """Save the model registry to YAML file."""
    with open(REGISTRY_FILE, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)


def list_models() -> list[str]:
    """List all registered model names."""
    registry = load_registry()
    return list(registry.get("models", {}).keys())


def get_model_config(name: str) -> dict | None:
    """Get the configuration for a specific model."""
    registry = load_registry()
    return registry.get("models", {}).get(name)


def add_model(
    name: str,
    provider_type: ProviderType,
    base_url: str,
    text_endpoint: str,
    image_endpoint: str,
    api_key_env_var: str | None = None,
    text_input_field: str = "input",
    text_output_field: str = "embedding",
    image_input_field: str = "input",
    image_output_field: str = "embedding",
    batch_size: int | None = None,
) -> None:
    """
    Add a new model to the registry.

    Args:
        name: Unique name for the model
        provider_type: Type of provider ('local' or 'openai_compatible')
        base_url: Base URL for the API (e.g., 'http://localhost:8080')
        text_endpoint: Path for text embeddings (e.g., '/txt/embed')
        image_endpoint: Path for image embeddings (e.g., '/img/embed')
        api_key_env_var: Environment variable name containing API key
        text_input_field: JSON field name for text input
        text_output_field: JSON field name for text embedding output
        image_input_field: JSON field name for image input
        image_output_field: JSON field name for image embedding output
        batch_size: Override batch size (None = auto-discover)
    """
    registry = load_registry()

    if name in registry.get("models", {}):
        raise ValueError(f"Model '{name}' already exists. Remove it first to update.")

    if batch_size is not None and batch_size not in VALID_BATCH_SIZES:
        raise ValueError(
            f"Invalid batch size {batch_size}. Must be one of {VALID_BATCH_SIZES}"
        )

    model_config = {
        "provider_type": provider_type,
        "base_url": base_url,
        "text_endpoint": {
            "path": text_endpoint,
            "method": "POST",
            "input_field": text_input_field,
            "output_field": text_output_field,
        },
        "image_endpoint": {
            "path": image_endpoint,
            "method": "POST",
            "input_field": image_input_field,
            "output_field": image_output_field,
        },
    }

    if api_key_env_var:
        model_config["api_key_env_var"] = api_key_env_var

    if batch_size is not None:
        model_config["user_batch_size"] = batch_size

    registry["models"][name] = model_config
    save_registry(registry)


def remove_model(name: str) -> bool:
    """
    Remove a model from the registry.

    Returns:
        True if model was removed, False if it didn't exist.
    """
    registry = load_registry()

    if name not in registry.get("models", {}):
        return False

    del registry["models"][name]
    save_registry(registry)
    return True


def update_model_batch_size(name: str, batch_size: int | None) -> None:
    """Update the batch size for a model."""
    registry = load_registry()

    if name not in registry.get("models", {}):
        raise ValueError(f"Model '{name}' not found in registry.")

    if batch_size is not None and batch_size not in VALID_BATCH_SIZES:
        raise ValueError(
            f"Invalid batch size {batch_size}. Must be one of {VALID_BATCH_SIZES}"
        )

    if batch_size is None:
        registry["models"][name].pop("user_batch_size", None)
    else:
        registry["models"][name]["user_batch_size"] = batch_size

    save_registry(registry)


def _config_to_provider_config(name: str, config: dict) -> ProviderConfig:
    """Convert a registry config dict to a ProviderConfig object."""
    text_ep = config["text_endpoint"]
    image_ep = config["image_endpoint"]

    return ProviderConfig(
        name=name,
        base_url=config["base_url"],
        text_endpoint=EndpointConfig(
            path=text_ep["path"],
            method=text_ep.get("method", "POST"),
            input_field=text_ep.get("input_field", "input"),
            output_field=text_ep.get("output_field", "embedding"),
            model=text_ep.get("model"),
            wrap_input=text_ep.get("wrap_input", False),
            input_wrapper_key=text_ep.get("input_wrapper_key"),
        ),
        image_endpoint=EndpointConfig(
            path=image_ep["path"],
            method=image_ep.get("method", "POST"),
            input_field=image_ep.get("input_field", "input"),
            output_field=image_ep.get("output_field", "embedding"),
            model=image_ep.get("model"),
            wrap_input=image_ep.get("wrap_input", False),
            input_wrapper_key=image_ep.get("input_wrapper_key"),
        ),
        api_key_env_var=config.get("api_key_env_var") or config.get("api_key_env"),
        batch_config=BatchConfig(),  # Will be discovered
        user_batch_size=config.get("user_batch_size"),
        task=config.get("task"),
        rate_limit_delay=config.get("rate_limit_delay"),
    )


def get_provider(name: str) -> LocalProvider | OpenAICompatibleProvider:
    """
    Get a provider instance for a registered model.

    Args:
        name: Name of the registered model.

    Returns:
        Provider instance ready for use.

    Raises:
        ValueError: If model not found or provider type unknown.
    """
    config = get_model_config(name)
    if config is None:
        raise ValueError(f"Model '{name}' not found in registry.")

    provider_type = config.get("provider_type", "local")
    provider_config = _config_to_provider_config(name, config)

    if provider_type == "local":
        return LocalProvider(provider_config)
    elif provider_type == "openai_compatible":
        return OpenAICompatibleProvider(provider_config)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


def get_all_providers() -> dict[str, LocalProvider | OpenAICompatibleProvider]:
    """Get provider instances for all registered models."""
    models = list_models()
    return {name: get_provider(name) for name in models}


def create_default_local_model() -> None:
    """Create the default local model configuration if it doesn't exist."""
    registry = load_registry()

    if "nomic-embed-v1.5" not in registry.get("models", {}):
        add_model(
            name="nomic-embed-v1.5",
            provider_type="local",
            base_url="http://localhost:8080",
            text_endpoint="/txt/embed",
            image_endpoint="/img/embed",
            text_input_field="input",
            text_output_field="embedding",
            image_input_field="input",
            image_output_field="embedding",
        )
