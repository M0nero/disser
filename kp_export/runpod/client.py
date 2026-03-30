from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, Optional


class RunpodClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://rest.runpod.io/v1",
        timeout_sec: float = 60.0,
        opener: Optional[Callable[..., Any]] = None,
    ) -> None:
        if not str(api_key or "").strip():
            raise ValueError("api_key is required")
        self.api_key = str(api_key)
        self.base_url = str(base_url).rstrip("/")
        self.timeout_sec = float(timeout_sec)
        self._opener = opener or urllib.request.urlopen

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        data = None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
        try:
            with self._opener(request, timeout=self.timeout_sec) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Runpod API {method} {url} failed: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Runpod API {method} {url} failed: {exc.reason}") from exc
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def list_pods(self) -> Any:
        return self._request("GET", "/pods")

    def get_pod(self, pod_id: str) -> Any:
        return self._request("GET", f"/pods/{pod_id}")

    def create_pod(self, payload: Dict[str, Any]) -> Any:
        return self._request("POST", "/pods", payload)

    def delete_pod(self, pod_id: str) -> Any:
        return self._request("DELETE", f"/pods/{pod_id}")

    def stop_pod(self, pod_id: str) -> Any:
        return self._request("POST", f"/pods/{pod_id}/stop")

    def start_pod(self, pod_id: str) -> Any:
        return self._request("POST", f"/pods/{pod_id}/resume")

    def list_network_volumes(self) -> Any:
        return self._request("GET", "/networkvolumes")

    def get_network_volume(self, volume_id: str) -> Any:
        return self._request("GET", f"/networkvolumes/{volume_id}")

    def create_network_volume(self, *, name: str, size_gb: int, data_center_id: str) -> Any:
        return self._request(
            "POST",
            "/networkvolumes",
            {
                "name": str(name),
                "size": int(size_gb),
                "dataCenterId": str(data_center_id),
            },
        )
