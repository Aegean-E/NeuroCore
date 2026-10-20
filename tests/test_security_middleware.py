"""Regression tests for M6: auth (opt-in API token) and CSRF (cross-origin write) enforcement."""

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from core import security


def _guarded_client() -> TestClient:
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"ok": True}

    @app.post("/do")
    async def do():
        return {"done": True}

    security.install_security_middleware(app)
    return TestClient(app)


def test_csrf_blocks_cross_origin_write(monkeypatch):
    monkeypatch.delenv("NEUROCORE_TRUSTED_ORIGINS", raising=False)
    monkeypatch.delenv("NEUROCORE_API_TOKEN", raising=False)
    client = _guarded_client()
    response = client.post("/do", headers={"origin": "http://evil.example.com"})
    assert response.status_code == 403


def test_csrf_allows_same_origin_write(monkeypatch):
    monkeypatch.delenv("NEUROCORE_TRUSTED_ORIGINS", raising=False)
    monkeypatch.delenv("NEUROCORE_API_TOKEN", raising=False)
    client = _guarded_client()
    assert client.post("/do", headers={"origin": "http://testserver"}).status_code == 200
    assert client.post("/do").status_code == 200


def test_csrf_allows_trusted_origin(monkeypatch):
    monkeypatch.setenv("NEUROCORE_TRUSTED_ORIGINS", "http://app.example.com")
    monkeypatch.delenv("NEUROCORE_API_TOKEN", raising=False)
    client = _guarded_client()
    assert client.post("/do", headers={"origin": "http://app.example.com"}).status_code == 200


def test_get_is_not_csrf_checked(monkeypatch):
    monkeypatch.delenv("NEUROCORE_API_TOKEN", raising=False)
    client = _guarded_client()
    assert client.get("/", headers={"origin": "http://evil.example.com"}).status_code == 200


def test_api_token_required_for_non_loopback_when_set(monkeypatch):
    monkeypatch.setenv("NEUROCORE_API_TOKEN", "s3cret")
    client = _guarded_client()
    assert client.get("/").status_code == 403
    assert client.get("/", headers={"x-api-token": "s3cret"}).status_code == 200
    assert client.get("/", headers={"x-api-token": "wrong"}).status_code == 403
    assert client.get("/", headers={"authorization": "Bearer s3cret"}).status_code == 200


def test_api_token_not_required_when_unset(monkeypatch):
    monkeypatch.delenv("NEUROCORE_API_TOKEN", raising=False)
    client = _guarded_client()
    assert client.get("/").status_code == 200


def test_loopback_client_is_exempt_from_token():
    loopback = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
    remote = SimpleNamespace(client=SimpleNamespace(host="10.0.0.5"))
    assert security._client_is_loopback(loopback) is True
    assert security._client_is_loopback(remote) is False