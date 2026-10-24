"""Download/cache behavior without real network requests or credentials."""

import json

import pytest
import requests

from data.gov_download import download_resale_csv


class Response:
    def __init__(self, payload=None, content=b"month,town\n2026-09,BISHAN\n", fail=False):
        self.payload, self.content, self.fail = payload, content, fail

    def raise_for_status(self):
        if self.fail:
            raise requests.HTTPError("signed-url?secret=must-not-appear")

    def json(self):
        return self.payload

    def iter_content(self, chunk_size):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def fake_network(monkeypatch, *, failed_download=False):
    calls = []
    responses = iter(
        [
            Response({"code": 0, "data": {"message": "initiated"}}),
            Response(
                {"code": 0, "data": {"url": "https://storage.example/resale.csv?signed=true"}}
            ),
            Response(fail=failed_download),
        ]
    )

    class Session:
        def mount(self, *args):
            pass

        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return next(responses)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("data.gov_download.requests.Session", Session)
    return calls


@pytest.mark.parametrize("key", ["", "test-only-key"])
def test_optional_auth_and_download_does_not_leak_key(tmp_path, monkeypatch, key):
    calls = fake_network(monkeypatch)
    path, manifest = download_resale_csv(tmp_path, api_key=key)
    expected = {"x-api-key": key} if key else {}
    assert calls[0][1]["headers"] == expected and calls[1][1]["headers"] == expected
    assert "headers" not in calls[2][1]
    assert path.read_bytes() == b"month,town\n2026-09,BISHAN\n"
    assert manifest["bytes"] == path.stat().st_size
    assert "signed=true" not in json.dumps(manifest)
    assert "test-only-key" not in json.dumps(manifest)
    # Rebuilding from a verified cache neither calls the API nor changes retrieval time.
    same_path, cached = download_resale_csv(tmp_path)
    assert same_path == path and cached == manifest and len(calls) == 3


def test_modified_cache_is_not_silently_trusted(tmp_path, monkeypatch):
    calls = fake_network(monkeypatch)
    path, _ = download_resale_csv(tmp_path)
    path.write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        download_resale_csv(tmp_path)
    assert len(calls) == 3


def test_failed_refresh_keeps_previous_cache_and_hides_signed_url(tmp_path, monkeypatch):
    fake_network(monkeypatch)
    path, original = download_resale_csv(tmp_path)
    original_bytes = path.read_bytes()
    fake_network(monkeypatch, failed_download=True)
    with pytest.raises(RuntimeError) as caught:
        download_resale_csv(tmp_path, refresh=True)
    assert "must-not-appear" not in str(caught.value)
    assert path.read_bytes() == original_bytes
    assert json.loads((tmp_path / "resale_2017_onwards.manifest.json").read_text()) == original
    assert not list(tmp_path.glob("*.part"))
