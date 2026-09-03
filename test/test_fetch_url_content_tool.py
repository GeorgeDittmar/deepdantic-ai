from __future__ import annotations

from types import SimpleNamespace

import pytest

from pydantask.tools import default_tools


class _FakeStreamResponse:
    def __init__(
        self, *, url: str, status_code: int, headers: dict[str, str], body: bytes
    ):
        self.status_code = status_code
        self.url = url
        self.headers = headers
        self._body = body
        self.encoding = "utf-8"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise Exception("status error")

    async def aiter_bytes(self):
        # yield in 2 chunks to exercise streaming
        mid = max(1, len(self._body) // 2)
        yield self._body[:mid]
        yield self._body[mid:]


class _FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        self._response: _FakeStreamResponse | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, method: str, url: str, headers: dict[str, str]):
        assert method == "GET"
        assert "User-Agent" in headers
        assert url == "https://example.com/"
        self._response = _FakeStreamResponse(
            url=url,
            status_code=200,
            headers={"content-type": "text/plain; charset=utf-8"},
            body=b"hello world",
        )
        return self._response


@pytest.mark.asyncio
async def test_fetch_url_content_happy_path(monkeypatch: pytest.MonkeyPatch):
    # Avoid DNS lookups and treat example.com as safe.
    monkeypatch.setattr(
        default_tools, "_host_looks_local_or_private", lambda host: (False, "")
    )

    # Patch httpx.AsyncClient used inside the tool.
    monkeypatch.setattr(default_tools.httpx, "AsyncClient", _FakeAsyncClient)

    text = await default_tools.fetch_url_content(
        "https://example.com/", max_chars=5_000
    )
    assert "Fetched:" in text
    assert "Status: 200" in text
    assert "hello world" in text


@pytest.mark.asyncio
async def test_fetch_url_content_blocks_localhost(monkeypatch: pytest.MonkeyPatch):
    # Ensure we don't actually try to fetch.
    monkeypatch.setattr(default_tools.httpx, "AsyncClient", _FakeAsyncClient)

    text = await default_tools.fetch_url_content("http://localhost:8000/")
    assert "blocked" in text.lower()
