from collections import defaultdict, deque
from collections.abc import Callable
from threading import Lock
from time import monotonic

from fastapi import Depends, HTTPException, Request, status

from app.config import Settings, get_settings


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def enforce(self, bucket: str, key: str, limit: int, window_seconds: int) -> None:
        if limit <= 0:
            return

        now = monotonic()
        event_key = f"{bucket}:{key}"
        cutoff = now - window_seconds

        with self._lock:
            window = self._events[event_key]
            while window and window[0] <= cutoff:
                window.popleft()

            if len(window) >= limit:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Retry later.",
                )

            window.append(now)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


rate_limiter = InMemoryRateLimiter()


def _client_identifier(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        client_ip = forwarded_for.split(",", maxsplit=1)[0].strip()
        if client_ip:
            return client_ip

    if request.client and request.client.host:
        return request.client.host

    return "unknown"


def build_rate_limit_dependency(
    bucket: str,
    limit_getter: Callable[[Settings], int],
):
    def dependency(
        request: Request,
        settings: Settings = Depends(get_settings),
    ) -> None:
        rate_limiter.enforce(
            bucket=bucket,
            key=_client_identifier(request),
            limit=limit_getter(settings),
            window_seconds=60,
        )

    return dependency


limit_ingest_requests = build_rate_limit_dependency(
    bucket="ingest",
    limit_getter=lambda settings: settings.rate_limit_ingest_per_minute,
)

limit_query_requests = build_rate_limit_dependency(
    bucket="query",
    limit_getter=lambda settings: settings.rate_limit_query_per_minute,
)
