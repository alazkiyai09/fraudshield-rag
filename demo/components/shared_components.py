from __future__ import annotations

import os
import time
from typing import Any

import requests
import streamlit as st


def get_setting(name: str, default: str = "") -> str:
    """Read Streamlit secret first, then environment fallback."""
    try:
        secret_value = st.secrets.get(name)
        if secret_value is not None:
            return str(secret_value)
    except Exception:
        pass

    return os.getenv(name, default)


def build_headers(api_key: str | None = None) -> dict[str, str]:
    headers: dict[str, str] = {}
    token = (api_key or "").strip()
    if token:
        headers["X-API-Key"] = token
    return headers


def _decode_json_or_text(resp: requests.Response) -> dict[str, Any]:
    try:
        parsed = resp.json()
        if isinstance(parsed, dict):
            return parsed
        return {"data": parsed}
    except ValueError:
        return {"raw_text": resp.text}


def api_call_with_retry(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    json_payload: dict[str, Any] | None = None,
    files: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    max_retries: int = 2,
    timeout: int = 60,
) -> dict[str, Any]:
    """HTTP call helper with retry for cold starts and timeouts."""
    for attempt in range(max_retries + 1):
        try:
            if method.upper() == "GET":
                resp = requests.get(url, headers=headers, timeout=timeout)
            elif method.upper() == "POST":
                if files is not None:
                    resp = requests.post(url, headers=headers, files=files, data=data, timeout=timeout)
                else:
                    resp = requests.post(url, headers=headers, json=json_payload, timeout=timeout)
            else:
                return {"error": f"Unsupported HTTP method: {method}"}

            payload = _decode_json_or_text(resp)

            if resp.status_code == 200:
                payload["_status_code"] = resp.status_code
                return payload

            if resp.status_code in {503, 504} and attempt < max_retries:
                time.sleep(3)
                continue

            detail = payload.get("detail") or payload.get("raw_text") or "Unknown API error"
            return {
                "error": f"API returned {resp.status_code}",
                "status_code": resp.status_code,
                "detail": detail,
                "response": payload,
            }
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                time.sleep(1)
                continue
            return {"error": "API timeout — service may still be warming up. Please retry."}
        except requests.RequestException as exc:
            if attempt < max_retries:
                time.sleep(1)
                continue
            return {"error": f"Request failed: {exc}"}

    return {"error": "API request failed after retries."}


def show_api_status(api_url: str, headers: dict[str, str] | None = None) -> tuple[bool, dict[str, Any]]:
    """Render API health status in the sidebar and return health payload."""
    health_url = f"{api_url.rstrip('/')}/health"
    result = api_call_with_retry(health_url, method="GET", headers=headers, timeout=15, max_retries=1)

    if "error" in result:
        if result.get("status_code") in {503, 504}:
            st.sidebar.warning("⏳ API warming up (cold start)")
        else:
            st.sidebar.error(f"❌ API Error: {result.get('detail', result['error'])}")
        return False, {}

    st.sidebar.success("✅ API Connected")
    return True, result


def show_footer() -> None:
    st.markdown("---")
    st.markdown(
        "Built by [Ahmad Whafa Azka](https://github.com/alazkiyai09) · "
        "[Portfolio](https://github.com/alazkiyai09/fraud-ai-portfolio) · "
        "2 peer-reviewed publications (Springer, IEEE)"
    )
