from __future__ import annotations

from urllib.parse import urlsplit


def ensure_local_path(value: str, *, kind: str) -> None:
    scheme = urlsplit(value).scheme
    if scheme and len(scheme) > 1:
        raise ValueError(f"{kind} must be a local filesystem path, not a URI-style location.")
