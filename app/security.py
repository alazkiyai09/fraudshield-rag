import hmac

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    expected_api_key = settings.api_key.strip()
    if not expected_api_key:
        return

    provided_api_key = (x_api_key or "").strip()
    if provided_api_key and hmac.compare_digest(provided_api_key, expected_api_key):
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key.",
    )
