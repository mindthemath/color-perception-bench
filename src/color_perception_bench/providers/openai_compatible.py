"""OpenAI-compatible embedding provider for standard API endpoints."""

import asyncio
import base64
import io
import os

import aiohttp
import numpy as np
from PIL import Image

from color_perception_bench.providers.base import (
    BaseAsyncProvider,
    BatchConfig,
    ProviderConfig,
    ProviderValidationError,
)


class OpenAICompatibleProvider(BaseAsyncProvider):
    """
    Provider for OpenAI-compatible embedding APIs.

    Supports standard OpenAI-style endpoints:
    - POST /v1/embeddings with {"input": "text", "model": "..."}
    - POST /v1/embeddings with {"input": ["base64..."], "model": "..."}

    Also works with other providers that follow similar patterns
    (Together, Anyscale, Fireworks, etc.)
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._openapi_schema: dict | None = None
        self._api_key: str | None = None

    def _get_api_key(self) -> str | None:
        """Get API key from environment variable."""
        if self._api_key is not None:
            return self._api_key

        if self.config.api_key_env_var:
            self._api_key = os.environ.get(self.config.api_key_env_var)
            if not self._api_key:
                raise ProviderValidationError(
                    f"API key environment variable '{self.config.api_key_env_var}' is not set. "
                    f"Please add it to your .env file."
                )
        return self._api_key

    def _get_headers(self) -> dict:
        """Get headers for API requests."""
        headers = {"Content-Type": "application/json"}
        api_key = self._get_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def _ensure_session(self):
        """Create aiohttp session if not exists."""
        if self._session is None:
            # Don't set headers at session creation - add them per-request instead
            self._session = aiohttp.ClientSession()

    async def _fetch_openapi_schema(self) -> dict:
        """Fetch and cache the OpenAPI schema."""
        if self._openapi_schema is not None:
            return self._openapi_schema

        await self._ensure_session()
        assert self._session is not None
        url = f"{self.config.base_url}/openapi.json"

        try:
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    # OpenAI and some providers don't expose /openapi.json
                    # Return empty schema to skip validation
                    self._openapi_schema = {"paths": {}}
                    return self._openapi_schema
                self._openapi_schema = await resp.json()
                return self._openapi_schema
        except aiohttp.ClientError:
            # If connection fails, skip validation
            self._openapi_schema = {"paths": {}}
            return self._openapi_schema

    async def validate_endpoints(self) -> None:
        """Validate that configured endpoints exist in OpenAPI schema."""
        schema = await self._fetch_openapi_schema()
        paths = schema.get("paths", {})

        # If no paths (schema not available), skip validation
        if not paths:
            return

        # Validate text endpoint
        text_path = self.config.text_endpoint.path
        if text_path not in paths:
            available = list(paths.keys())
            raise ProviderValidationError(
                f"Text endpoint '{text_path}' not found in OpenAPI schema. "
                f"Available paths: {available}"
            )

        # Validate image endpoint (may be same as text for multimodal APIs)
        img_path = self.config.image_endpoint.path
        if img_path not in paths:
            available = list(paths.keys())
            raise ProviderValidationError(
                f"Image endpoint '{img_path}' not found in OpenAPI schema. "
                f"Available paths: {available}"
            )

    async def discover_batch_support(self) -> BatchConfig:
        """
        Discover batch support from OpenAPI schema.

        OpenAI-style APIs typically support batching via array inputs.
        """
        schema = await self._fetch_openapi_schema()
        paths = schema.get("paths", {})

        text_path = self.config.text_endpoint.path
        text_method = self.config.text_endpoint.method.lower()

        if text_path in paths and text_method in paths[text_path]:
            endpoint_spec = paths[text_path][text_method]
            batch_config = self._check_batch_support_in_spec(endpoint_spec, schema)
            if batch_config.supported:
                self.config.batch_config = batch_config
                return batch_config

        # OpenAI typically supports batching even if not explicit in schema
        # Default to reasonable batch size
        self.config.batch_config = BatchConfig(
            supported=True, max_size=2048, discovered_from_schema=False
        )
        return self.config.batch_config

    def _check_batch_support_in_spec(
        self, endpoint_spec: dict, full_schema: dict
    ) -> BatchConfig:
        """Check if an endpoint spec indicates batch support."""
        request_body = endpoint_spec.get("requestBody", {})
        content = request_body.get("content", {})

        for content_type, content_spec in content.items():
            if "json" in content_type:
                schema = content_spec.get("schema", {})
                schema = self._resolve_ref(schema, full_schema)

                properties = schema.get("properties", {})
                input_field = self.config.text_endpoint.input_field

                if input_field in properties:
                    input_schema = self._resolve_ref(
                        properties[input_field], full_schema
                    )

                    if input_schema.get("type") == "array":
                        max_items = input_schema.get("maxItems", 2048)
                        return BatchConfig(
                            supported=True,
                            max_size=max_items,
                            discovered_from_schema=True,
                        )

                    for key in ("anyOf", "oneOf"):
                        if key in input_schema:
                            for option in input_schema[key]:
                                option = self._resolve_ref(option, full_schema)
                                if option.get("type") == "array":
                                    max_items = option.get("maxItems", 2048)
                                    return BatchConfig(
                                        supported=True,
                                        max_size=max_items,
                                        discovered_from_schema=True,
                                    )

        return BatchConfig(supported=False, max_size=1)

    def _resolve_ref(self, schema: dict, full_schema: dict) -> dict:
        """Resolve a $ref in the schema."""
        if "$ref" not in schema:
            return schema

        ref_path = schema["$ref"]
        if ref_path.startswith("#/"):
            parts = ref_path[2:].split("/")
            resolved = full_schema
            for part in parts:
                resolved = resolved.get(part, {})
            return resolved

        return schema

    async def get_text_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """Get embeddings for text inputs using OpenAI-style API."""
        await self._ensure_session()
        assert self._session is not None

        url = f"{self.config.base_url}{self.config.text_endpoint.path}"
        input_field = self.config.text_endpoint.input_field

        embeddings = []
        batch_size = self.config.effective_batch_size

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Wrap items if needed (e.g., Jina style: {"text": "..."} or {"image": "..."})
            if self.config.text_endpoint.wrap_input:
                wrapper_key = self.config.text_endpoint.input_wrapper_key or "text"
                batch_input = [{wrapper_key: text} for text in batch]
            else:
                # OpenAI style: input can be string or array
                batch_input = batch if len(batch) > 1 else batch[0]

            payload = {input_field: batch_input}

            # Add model if specified
            if self.config.text_endpoint.model:
                payload["model"] = self.config.text_endpoint.model

            # Add task if specified (e.g., for jina-embeddings-v4)
            if self.config.task:
                payload["task"] = self.config.task

            async with self._session.post(
                url, json=payload, headers=self._get_headers()
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Text embedding failed: {resp.status} - {text}")

                data = await resp.json()
                batch_embeddings = self._parse_embedding_response(data)
                embeddings.extend(batch_embeddings)

            # Rate limit delay if configured
            if self.config.rate_limit_delay and i + batch_size < len(texts):
                await asyncio.sleep(self.config.rate_limit_delay)

        return embeddings

    async def get_image_embeddings(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Get embeddings for image inputs using OpenAI-style API."""
        await self._ensure_session()
        assert self._session is not None

        url = f"{self.config.base_url}{self.config.image_endpoint.path}"
        input_field = self.config.image_endpoint.input_field

        # Convert images to base64
        def img_to_base64(img: Image.Image) -> str:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

        base64_images = [img_to_base64(img) for img in images]
        embeddings = []
        batch_size = self.config.effective_batch_size

        # Process in batches
        for i in range(0, len(base64_images), batch_size):
            batch = base64_images[i : i + batch_size]

            # Wrap items if needed (e.g., Jina style: {"text": "..."} or {"image": "..."})
            if self.config.image_endpoint.wrap_input:
                wrapper_key = self.config.image_endpoint.input_wrapper_key or "image"
                batch_input = [{wrapper_key: img} for img in batch]
            else:
                batch_input = batch if len(batch) > 1 else batch[0]

            payload = {input_field: batch_input}

            # Add model if specified
            if self.config.image_endpoint.model:
                payload["model"] = self.config.image_endpoint.model

            # Add task if specified (e.g., for jina-embeddings-v4)
            if self.config.task:
                payload["task"] = self.config.task

            async with self._session.post(
                url, json=payload, headers=self._get_headers()
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(
                        f"Image embedding failed: {resp.status} - {text}"
                    )

                data = await resp.json()
                batch_embeddings = self._parse_embedding_response(data)
                embeddings.extend(batch_embeddings)

            # Rate limit delay if configured
            if self.config.rate_limit_delay and i + batch_size < len(base64_images):
                await asyncio.sleep(self.config.rate_limit_delay)

        return embeddings

    def _parse_embedding_response(self, data: dict) -> list[np.ndarray]:
        """Parse embedding response in various formats."""
        # OpenAI style: {"data": [{"embedding": [...], "index": 0}, ...]}
        if "data" in data:
            items = data["data"]
            if isinstance(items, list):
                # Sort by index if present
                if items and "index" in items[0]:
                    items = sorted(items, key=lambda x: x.get("index", 0))
                return [
                    np.array(item.get("embedding", item), dtype=np.float32)
                    for item in items
                ]

        # Direct embedding field
        output_field = self.config.text_endpoint.output_field
        if output_field in data:
            result = data[output_field]
            if isinstance(result, list):
                if result and isinstance(result[0], list):
                    # List of embeddings
                    return [np.array(emb, dtype=np.float32) for emb in result]
                else:
                    # Single embedding
                    return [np.array(result, dtype=np.float32)]

        # Fallback: try common field names
        for field in ("embeddings", "embedding", "vectors", "vector"):
            if field in data:
                result = data[field]
                if isinstance(result, list):
                    if result and isinstance(result[0], list):
                        return [np.array(emb, dtype=np.float32) for emb in result]
                    else:
                        return [np.array(result, dtype=np.float32)]

        raise RuntimeError(f"Could not parse embedding response: {list(data.keys())}")
