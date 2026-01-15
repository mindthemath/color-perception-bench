"""Base classes and protocols for embedding providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image


class ProviderValidationError(Exception):
    """Raised when provider validation fails against OpenAPI schema."""

    pass


@dataclass
class EndpointConfig:
    """Configuration for an embedding endpoint."""

    path: str
    method: str = "POST"
    input_field: str = "input"
    output_field: str = "embedding"


@dataclass
class BatchConfig:
    """Configuration for batching support."""

    supported: bool = False
    max_size: int = 1
    discovered_from_schema: bool = False


@dataclass
class ProviderConfig:
    """Full configuration for an embedding provider."""

    name: str
    base_url: str
    text_endpoint: EndpointConfig
    image_endpoint: EndpointConfig
    api_key_env_var: str | None = None
    batch_config: BatchConfig = field(default_factory=BatchConfig)
    user_batch_size: int | None = None  # User override for batch size

    @property
    def effective_batch_size(self) -> int:
        """Return the effective batch size (user override or discovered max)."""
        if self.user_batch_size is not None:
            return min(self.user_batch_size, self.batch_config.max_size)
        return self.batch_config.max_size if self.batch_config.supported else 1


@runtime_checkable
class AsyncEmbeddingProvider(Protocol):
    """Protocol for async embedding providers."""

    config: ProviderConfig

    async def validate_endpoints(self) -> None:
        """
        Validate endpoints against OpenAPI schema.

        Fetches /openapi.json and verifies that the configured endpoints exist
        and match expected signatures.

        Raises:
            ProviderValidationError: If endpoints don't match schema.
        """
        ...

    async def discover_batch_support(self) -> BatchConfig:
        """
        Discover batch support from OpenAPI schema.

        Parses the schema to determine if the API supports batching
        and what the maximum batch size is.

        Returns:
            BatchConfig with discovered settings.
        """
        ...

    async def get_text_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """
        Get embeddings for a list of text inputs.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors as numpy arrays.
        """
        ...

    async def get_image_embeddings(self, images: list[Image.Image]) -> list[np.ndarray]:
        """
        Get embeddings for a list of images.

        Args:
            images: List of PIL Images to embed.

        Returns:
            List of embedding vectors as numpy arrays.
        """
        ...

    async def close(self) -> None:
        """Close any open connections."""
        ...


class BaseAsyncProvider(ABC):
    """Base implementation for async embedding providers."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._session = None

    @abstractmethod
    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        ...

    @abstractmethod
    async def validate_endpoints(self) -> None:
        """Validate endpoints against OpenAPI schema."""
        ...

    @abstractmethod
    async def discover_batch_support(self) -> BatchConfig:
        """Discover batch support from OpenAPI schema."""
        ...

    @abstractmethod
    async def get_text_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """Get embeddings for a list of text inputs."""
        ...

    @abstractmethod
    async def get_image_embeddings(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Get embeddings for a list of images."""
        ...

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session is not None and hasattr(self._session, 'close'):
            await self._session.close()
            self._session = None

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
