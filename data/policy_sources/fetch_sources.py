"""
Capture the official pages listed in data/policy_sources/sources.yaml (Member 3: HDB policy corpus).

  auto pages    fetched here: robots.txt is checked, one request per second, the project named in the User-Agent
  manual pages  saved by a person in a browser (HDB and MAS refuse scripts; their bot protection is respected, not
                worked around) as raw/<id>.html or raw/<id>.pdf; --record-manual hashes them into the manifest

Every capture is kept untouched in raw/ and recorded in raw/manifest.json (url, method, captured_at, sha256, status),
so each chunk the RAG cites can be traced to a dated, hashed copy of the page.

    .venv/bin/python -m data.policy_sources.fetch_sources                 # fetch auto pages not captured yet
    .venv/bin/python -m data.policy_sources.fetch_sources --refresh       # re-fetch all auto pages, report changes
    .venv/bin/python -m data.policy_sources.fetch_sources --record-manual # record browser-saved HDB/MAS/IRAS pages
    .venv/bin/python -m data.policy_sources.fetch_sources --status        # what is captured, what is missing
    .venv/bin/python -m data.policy_sources.fetch_sources --discover-cpf  # CPF housing pages not in sources.yaml
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from data.gov_download import sha256_file

HERE = Path(__file__).resolve().parent
SOURCES_YAML = HERE / "sources.yaml"
RAW_DIR = HERE / "raw"
MANIFEST = RAW_DIR / "manifest.json"
USER_AGENT = "HDB-Compass-research/0.1 (SMU IS469 student project; policy corpus capture)"
REQUEST_GAP_S = 1.0
RAW_SUFFIXES = (".html", ".htm", ".pdf")


def load_sources() -> List[Dict[str, Any]]:
    data = yaml.safe_load(SOURCES_YAML.read_text(encoding="utf-8")) or {}
    sources = data.get("sources") or []
    ids = [s["id"] for s in sources]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate source ids in sources.yaml: {sorted(duplicates)}")
    _FILE_NAMES.clear()
    _FILE_NAMES.update({s["id"]: s["file"] for s in sources if s.get("file")})
    return sources


_FILE_NAMES: Dict[str, str] = {}   # id -> saved file name, for captures not named <id>.html (sources.yaml `file`)


def load_manifest() -> Dict[str, Any]:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: Dict[str, Any]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def raw_file(source_id: str) -> Optional[Path]:
    """The capture for a source, whichever format it was saved in."""
    if source_id in _FILE_NAMES:
        path = RAW_DIR / _FILE_NAMES[source_id]
        return path if path.exists() else None
    for suffix in RAW_SUFFIXES:
        path = RAW_DIR / f"{source_id}{suffix}"
        if path.exists():
            return path
    return None


def _session() -> requests.Session:
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"], respect_retry_after_header=True)
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = USER_AGENT
    return session


_ROBOTS: Dict[str, RobotFileParser] = {}


def allowed_by_robots(session: requests.Session, url: str) -> bool:
    parts = urlsplit(url)
    root = f"{parts.scheme}://{parts.netloc}"
    if root not in _ROBOTS:
        parser = RobotFileParser()
        try:
            response = session.get(f"{root}/robots.txt", timeout=(10, 30))
            # a missing robots.txt (404 page) allows everything; a refused one (401/403) allows nothing
            if response.status_code in (401, 403):
                parser.disallow_all = True
            elif response.ok and "text/plain" in response.headers.get("content-type", ""):
                parser.parse(response.text.splitlines())
            else:
                parser.parse([])
        except requests.RequestException:
            parser.parse([])
        _ROBOTS[root] = parser
    return _ROBOTS[root].can_fetch(USER_AGENT, url)


def looks_blocked(html: str) -> bool:
    """A 200 response that is really an error or bot-check page (MAS answers "service is currently unavailable")."""
    head = html[:20000].lower()
    return any(marker in head for marker in (
        "service is currently unavailable", "<title>page not found", "access denied", "are you a robot",
        "captcha", "request blocked"))


def fetch_auto(sources: List[Dict[str, Any]], refresh: bool) -> int:
    manifest = load_manifest()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    with _session() as session:
        for source in sources:
            if source.get("capture") != "auto":
                continue
            sid, url = source["id"], source["url"]
            if not refresh and raw_file(sid) and sid in manifest:
                print(f"  kept     {sid} (captured {manifest[sid]['captured_at'][:10]})")
                continue
            if not allowed_by_robots(session, url):
                print(f"  SKIPPED  {sid}: robots.txt disallows {url}")
                failures += 1
                continue
            time.sleep(REQUEST_GAP_S)
            try:
                response = session.get(url, timeout=(10, 60))
            except requests.RequestException as exc:
                print(f"  FAILED   {sid}: {type(exc).__name__}")
                failures += 1
                continue
            if response.status_code != 200 or looks_blocked(response.text):
                print(f"  FAILED   {sid}: HTTP {response.status_code}"
                      + (" (error or bot-check page)" if response.status_code == 200 else "")
                      + " - capture it by hand (capture: manual)")
                failures += 1
                continue
            target = RAW_DIR / f"{sid}.html"
            previous = manifest.get(sid, {}).get("sha256")
            target.write_bytes(response.content)
            digest = sha256_file(target)
            manifest[sid] = {"url": url, "final_url": response.url, "method": "auto", "status": response.status_code,
                             "captured_at": datetime.now(timezone.utc).isoformat(), "sha256": digest,
                             "file": target.name}
            change = "" if previous is None else (" (unchanged)" if previous == digest else " (CHANGED since last capture)")
            print(f"  fetched  {sid}{change}")
    save_manifest(manifest)
    return failures


def record_manual(sources: List[Dict[str, Any]]) -> int:
    manifest = load_manifest()
    missing = 0
    for source in sources:
        if source.get("capture") != "manual":
            continue
        sid = source["id"]
        path = raw_file(sid)
        if path is None:
            print(f"  MISSING  {sid}: save {source['url']} as raw/{sid}.html (or .pdf)"
                  + (" [optional]" if source.get("optional") else ""))
            missing += 0 if source.get("optional") else 1
            continue
        digest = sha256_file(path)
        if manifest.get(sid, {}).get("sha256") == digest:
            print(f"  kept     {sid}")
            continue
        if path.suffix != ".pdf" and looks_blocked(path.read_text(encoding="utf-8", errors="ignore")):
            print(f"  REJECTED {sid}: the saved file is an error or bot-check page; save the page again")
            missing += 1
            continue
        manifest[sid] = {"url": source["url"], "method": "manual", "status": None,
                         "captured_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                         "recorded_at": datetime.now(timezone.utc).isoformat(), "sha256": digest, "file": path.name,
                         "url_verified": bool(source.get("verified"))}
        print(f"  recorded {sid} ({path.name})" + ("" if source.get("verified") else "  [set verified: true in sources.yaml]"))
    save_manifest(manifest)
    return missing


def status(sources: List[Dict[str, Any]]) -> int:
    manifest = load_manifest()
    missing = 0
    for source in sources:
        sid = source["id"]
        entry = manifest.get(sid)
        flag = " [excluded]" if source.get("exclude") else ""
        if entry and raw_file(sid):
            print(f"  ok       {sid:<62} {entry['method']:<6} {entry['captured_at'][:10]}{flag}")
        else:
            missing += 0 if source.get("optional") else 1
            print(f"  missing  {sid:<62} {source.get('capture'):<6} {source['url']}"
                  + (" [optional]" if source.get("optional") else ""))
    captured = sum(1 for src in sources if src["id"] in manifest and raw_file(src["id"]))
    optional_missing = len(sources) - captured - missing
    print(f"{captured}/{len(sources)} sources captured" + (f" ({optional_missing} optional still missing)" if optional_missing else ""))
    return missing


def discover_cpf(sources: List[Dict[str, Any]]) -> None:
    """CPF housing pages in CPF's sitemap that sources.yaml does not list yet (for a person to review and add)."""
    known = {s["url"].rstrip("/") for s in sources}
    with _session() as session:
        xml = session.get("https://www.cpf.gov.sg/sitemap.xml", timeout=(10, 60)).text
    locs = re.findall(r"<loc>([^<]+)</loc>", xml)
    pattern = re.compile(r"/member/(home-ownership|infohub/educational-resources/.*(hous|grant|hdb|flat|home))", re.I)
    for url in sorted(u for u in locs if pattern.search(u) and u.rstrip("/") not in known):
        print(f"  candidate  {url}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Capture the official pages of the HDB policy corpus.")
    parser.add_argument("--refresh", action="store_true", help="re-fetch every auto page and report changed pages")
    parser.add_argument("--record-manual", action="store_true", help="hash browser-saved pages into the manifest")
    parser.add_argument("--status", action="store_true", help="list captured and missing sources")
    parser.add_argument("--discover-cpf", action="store_true", help="list CPF housing pages not in sources.yaml")
    args = parser.parse_args(argv)
    sources = load_sources()
    if args.status:
        return 0 if status(sources) == 0 else 1
    if args.discover_cpf:
        discover_cpf(sources)
        return 0
    if args.record_manual:
        return 0 if record_manual(sources) == 0 else 1
    return 0 if fetch_auto(sources, refresh=args.refresh) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
