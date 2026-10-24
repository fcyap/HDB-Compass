"""Download the official resale CSV, optionally authenticated, into an atomic cache."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from data.market_database import RESALE_DATASET_ID, RESALE_SOURCE_URL

API_BASE = "https://api-open.data.gov.sg/v1/public/api/datasets"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_resale_csv(
    cache_dir: Path, api_key: str = "", refresh: bool = False
) -> tuple[Path, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "resale_2017_onwards.csv"
    manifest = cache_dir / "resale_2017_onwards.manifest.json"
    if not refresh and target.exists() and manifest.exists():
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if metadata.get("sha256") == sha256_file(target):
            return target, metadata
        raise ValueError("Cached resale checksum differs. Use --refresh to download it again.")

    headers = {"x-api-key": api_key} if api_key else {}
    retry = Retry(
        total=4,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    temporary = None
    try:
        with requests.Session() as session:
            session.mount("https://", HTTPAdapter(max_retries=retry))

            def request_metadata(action: str) -> dict:
                response = session.get(
                    f"{API_BASE}/{RESALE_DATASET_ID}/{action}", headers=headers, timeout=(10, 60)
                )
                response.raise_for_status()
                payload = response.json()
                # Code 0 = success; pending/rate-limit codes are handled by polling.
                if payload.get("code") not in (0, 24):
                    raise ValueError(f"data.gov.sg {action} failed (code {payload.get('code')}).")
                return payload

            request_metadata("initiate-download")
            download_url = None
            for attempt in range(10):
                payload = request_metadata("poll-download")
                download_url = (payload.get("data") or {}).get("url")
                if download_url:
                    break
                time.sleep(min(2**attempt, 30))
            if not download_url:
                raise TimeoutError(
                    "data.gov.sg did not prepare the resale download. Try again later."
                )

            # Do not forward the API key to the signed object-storage URL.
            with session.get(download_url, stream=True, timeout=(10, 120)) as response:
                response.raise_for_status()
                with NamedTemporaryFile(dir=cache_dir, suffix=".csv.part", delete=False) as stream:
                    temporary = Path(stream.name)
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        stream.write(chunk)
            if temporary.stat().st_size == 0:
                raise ValueError("Downloaded resale CSV is empty.")
            metadata = {
                "dataset_id": RESALE_DATASET_ID,
                "source_url": RESALE_SOURCE_URL,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "sha256": sha256_file(temporary),
                "bytes": temporary.stat().st_size,
            }
            temporary.replace(target)
            manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            return target, metadata
    except requests.RequestException as exc:
        # Request exceptions can contain signed URLs. Never print them or credentials.
        raise RuntimeError(
            f"Public resale download failed ({type(exc).__name__}); check network access or retry later."
        ) from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
