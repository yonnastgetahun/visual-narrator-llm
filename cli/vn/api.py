from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

import httpx


DEFAULT_API_URL = "https://visual-narrator-demo.vercel.app"


class VNApiError(RuntimeError):
    """Raised when the Visual Narrator API returns an unusable response."""


class VNClient:
    def __init__(self, api_url: str = DEFAULT_API_URL, api_key: str | None = None, timeout: float = 120.0):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def describe_frame(self, frame_path: Path) -> dict[str, Any]:
        frame_base64 = encode_file_base64(frame_path)
        payload: dict[str, Any] = {"frame_base64": frame_base64}
        if self.api_key:
            payload["api_key"] = self.api_key

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        started = time.perf_counter()
        data = self._post_json("/api/v1/describe-frame", payload, headers=headers)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        if not isinstance(data, dict):
            raise VNApiError("describe-frame returned a non-object response")

        data.setdefault("latency_ms", elapsed_ms)
        return data

    def benchmark_frame(self, image_path: Path) -> dict[str, Any]:
        payload = {"frame_data": encode_file_base64(image_path)}
        data = self._post_json("/api/benchmark-competitors", payload)
        if not isinstance(data, dict):
            raise VNApiError("benchmark-competitors returned a non-object response")
        return data

    def create_key(self, email: str) -> dict[str, Any]:
        data = self._post_json("/api/keys/create", {"email": email})
        if not isinstance(data, dict):
            raise VNApiError("keys/create returned a non-object response")
        return data

    def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> Any:
        url = f"{self.api_url}{path}"
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                response = client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise VNApiError(f"request to {url} failed: {exc}") from exc

        if response.status_code >= 400:
            message = _extract_error(response)
            raise VNApiError(f"{response.status_code} response from {url}: {message}")

        try:
            return response.json()
        except ValueError as exc:
            raise VNApiError(f"invalid JSON response from {url}: {response.text[:300]}") from exc


def encode_file_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _extract_error(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:300]
    if isinstance(data, dict):
        return str(data.get("error") or data.get("message") or data.get("detail") or data)
    return str(data)
