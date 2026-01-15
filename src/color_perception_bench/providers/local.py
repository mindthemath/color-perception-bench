"""Local embedding provider for localhost API."""

import base64
import io

import aiohttp
import numpy as np
from PIL import Image

from color_perception_bench.providers.base import (
    BaseAsyncProvider,
    BatchConfig,
    ProviderConfig,
    ProviderValidationError,
)


class LocalProvider(BaseAsyncProvider):
    """
    Provider for local embedding API (localhost:8080 style).

    Expects endpoints like:
    - POST /txt/embed with {"input": "text"} -> {"embedding": [...]}
    - POST /img/embed with {"input": "base64..."} -> {"embedding": [...]}
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._openapi_schema: dict | None = None

    async def _ensure_session(self):
        """Create aiohttp session if not exists."""
        if self._session is None:
            self._session = aiohttp.ClientSession()

    async def _fetch_openapi_schema(self) -> dict:
        """Fetch and cache the OpenAPI schema."""
        if self._openapi_schema is not None:
            return self._openapi_schema

        await self._ensure_session()
        url = f"{self.config.base_url}/openapi.json"

        try:
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    raise ProviderValidationError(
                        f"Failed to fetch OpenAPI schema from {url}: HTTP {resp.status}"
                    )
                self._openapi_schema = await resp.json()
                return self._openapi_schema
        except aiohttp.ClientError as e:
            raise ProviderValidationError(
                f"Failed to connect to {url}: {e}"
            ) from e

    async def validate_endpoints(self) -> None:
        """Validate that configured endpoints exist in OpenAPI schema."""
        schema = await self._fetch_openapi_schema()
        paths = schema.get("paths", {})

        # Validate text endpoint
        text_path = self.config.text_endpoint.path
        if text_path not in paths:
            available = list(paths.keys())
            raise ProviderValidationError(
                f"Text endpoint '{text_path}' not found in OpenAPI schema. "
                f"Available paths: {available}"
            )

        text_methods = paths[text_path]
        expected_method = self.config.text_endpoint.method.lower()
        if expected_method not in text_methods:
            raise ProviderValidationError(
                f"Text endpoint '{text_path}' does not support {expected_method.upper()}. "
                f"Available methods: {list(text_methods.keys())}"
            )

        # Validate image endpoint
        img_path = self.config.image_endpoint.path
        if img_path not in paths:
            available = list(paths.keys())
            raise ProviderValidationError(
                f"Image endpoint '{img_path}' not found in OpenAPI schema. "
                f"Available paths: {available}"
            )

        img_methods = paths[img_path]
        expected_method = self.config.image_endpoint.method.lower()
        if expected_method not in img_methods:
            raise ProviderValidationError(
                f"Image endpoint '{img_path}' does not support {expected_method.upper()}. "
                f"Available methods: {list(img_methods.keys())}"
            )

    async def discover_batch_support(self) -> BatchConfig:
        """
        Discover batch support from OpenAPI schema.

        Looks for array types in request body schemas to detect batching.
        """
        schema = await self._fetch_openapi_schema()
        paths = schema.get("paths", {})

        # Check text endpoint for batch support
        text_path = self.config.text_endpoint.path
        text_method = self.config.text_endpoint.method.lower()

        if text_path in paths and text_method in paths[text_path]:
            endpoint_spec = paths[text_path][text_method]
            batch_config = self._check_batch_support_in_spec(endpoint_spec, schema)
            if batch_config.supported:
                self.config.batch_config = batch_config
                return batch_config

        # Default: no batch support
        self.config.batch_config = BatchConfig(supported=False, max_size=1)
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

                # Look for input field
                properties = schema.get("properties", {})
                input_field = self.config.text_endpoint.input_field

                if input_field in properties:
                    input_schema = self._resolve_ref(
                        properties[input_field], full_schema
                    )

                    # Check if input accepts array
                    if input_schema.get("type") == "array":
                        max_items = input_schema.get("maxItems", 1024)
                        return BatchConfig(
                            supported=True,
                            max_size=max_items,
                            discovered_from_schema=True,
                        )

                    # Check for anyOf/oneOf with array option
                    for key in ("anyOf", "oneOf"):
                        if key in input_schema:
                            for option in input_schema[key]:
                                option = self._resolve_ref(option, full_schema)
                                if option.get("type") == "array":
                                    max_items = option.get("maxItems", 1024)
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
        """Get embeddings for text inputs."""
        await self._ensure_session()

        url = f"{self.config.base_url}{self.config.text_endpoint.path}"
        input_field = self.config.text_endpoint.input_field
        output_field = self.config.text_endpoint.output_field

        embeddings = []

        if self.config.batch_config.supported and len(texts) > 1:
            # Batch request
            payload = {input_field: texts}
            async with self._session.post(url, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Text embedding failed: {resp.status} - {text}")
                data = await resp.json()

                # Handle batch response (could be list of embeddings or nested)
                result = data.get(output_field, data.get("embeddings", data.get("data", [])))
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict):
                        # OpenAI style: [{"embedding": [...]}, ...]
                        embeddings = [
                            np.array(item.get("embedding", item), dtype=np.float32)
                            for item in result
                        ]
                    else:
                        # Direct list of embeddings: [[...], [...], ...]
                        embeddings = [np.array(emb, dtype=np.float32) for emb in result]
                else:
                    embeddings = [np.array(result, dtype=np.float32)]
        else:
            # Individual requests
            for text in texts:
                payload = {input_field: text}
                async with self._session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        text_resp = await resp.text()
                        raise RuntimeError(
                            f"Text embedding failed for '{text[:50]}...': "
                            f"{resp.status} - {text_resp}"
                        )
                    data = await resp.json()
                    emb = data.get(output_field, data)
                    embeddings.append(np.array(emb, dtype=np.float32))

        return embeddings

    async def get_image_embeddings(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Get embeddings for image inputs."""
        await self._ensure_session()

        url = f"{self.config.base_url}{self.config.image_endpoint.path}"
        input_field = self.config.image_endpoint.input_field
        output_field = self.config.image_endpoint.output_field

        embeddings = []

        # Convert images to base64
        def img_to_base64(img: Image.Image) -> str:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            # Add data URL prefix for local API compatibility
            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{img_b64}"

        base64_images = [img_to_base64(img) for img in images]

        if self.config.batch_config.supported and len(images) > 1:
            # Batch request (uses plural field name for batches)
            batch_field = "contents" if input_field == "content" else input_field + "s"
            payload = {batch_field: base64_images}
            async with self._session.post(url, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Image embedding failed: {resp.status} - {text}")
                data = await resp.json()

                result = data.get(output_field, data.get("embeddings", data.get("data", [])))
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict):
                        embeddings = [
                            np.array(item.get("embedding", item), dtype=np.float32)
                            for item in result
                        ]
                    else:
                        embeddings = [np.array(emb, dtype=np.float32) for emb in result]
                else:
                    embeddings = [np.array(result, dtype=np.float32)]
        else:
            # Individual requests
            for b64_img in base64_images:
                payload = {input_field: b64_img}
                async with self._session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"Image embedding failed: {resp.status} - {text}")
                    data = await resp.json()
                    emb = data.get(output_field, data)
                    embeddings.append(np.array(emb, dtype=np.float32))

        return embeddings
