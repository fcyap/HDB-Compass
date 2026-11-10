"""
Turn the captured official pages (data/policy_sources/raw/) into the Markdown documents the RAG ingests (Member 3).

The page text is kept VERBATIM: navigation, footers and scripts are dropped (trafilatura for HTML, pymupdf4llm for
PDF), nothing is paraphrased or summarised. One document per page, so every chunk cites the exact page it came from.
Each document starts with the header data/ingest_hdb_policy_rag.extract_frontmatter reads:

    # <page title>
    **Source URL:** ...   **Publisher:** ...   **Policy Category:** ...   **Retrieved:** <capture date>
    **Page Date:** <date the page states, else "not stated on the page">   **Content SHA256:** <raw capture hash>

    .venv/bin/python -m data.policy_sources.extract_sources            # write data/policy_sources/extracted/*.md
    .venv/bin/python -m data.policy_sources.extract_sources --install  # also install them as the RAG corpus
                                                                        # 
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data.policy_sources.fetch_sources import load_manifest, load_sources, raw_file

HERE = Path(__file__).resolve().parent
EXTRACTED_DIR = HERE / "extracted"
DATA_DIR = HERE.parent
CORPUS_DIR = DATA_DIR / "policy_documents"
MIN_WORDS = 80   # less than this after extraction means a failed capture (a bot page, an empty JS shell)

# Page furniture trafilatura sometimes keeps: lines that are UI, not policy text.
BOILERPLATE = re.compile(
    r"^(was this (page|article|information) (helpful|useful)\??|share( this)?( page| article)?:?|print( this page)?|"
    r"back to top|last updated:?$|yes\s*no|thank you for your feedback.*|rate this (page|article).*|"
    r"\*?source: cpf board\*?|skip to (main )?content)$",
    re.I,
)
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE_PATTERNS = (
    re.compile(rf"(?:last updated|updated on|published on)[:\s]*(\d{{1,2}} (?:{MONTHS}) \d{{4}})", re.I),
    re.compile(rf"^\*?(\d{{1,2}} (?:{MONTHS}) \d{{4}})\*?$", re.M),
)


def page_date(text: str, metadata_date: Optional[str]) -> Optional[str]:
    """The date the page itself states as its date (ISO): a "last updated / published on" label anywhere, or a date
    on its own line at the top (CPF articles: "*9 May 2025*"). None otherwise. A date elsewhere in the text is a
    policy's history ("from 3 March 2006"), and trafilatura's metadata date is a guess, so neither is used."""
    candidates = [(DATE_PATTERNS[0], text), (DATE_PATTERNS[1], text[:400])]
    for pattern, scope in candidates:
        m = pattern.search(scope)
        if m:
            try:
                return datetime.strptime(m.group(1), "%d %B %Y").date().isoformat()
            except ValueError:
                pass
    return None


_NEXT_DATA = re.compile(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S)
_RICH_HTML = re.compile(r"<(p|table|h[1-6]|li|ul|ol|div|strong)\b", re.I)
_SKIP_KEYS = {"dictionary", "sidebarData", "headLinks", "site", "sidebarError", "header", "footer", "navigation"}


def _fragment_markdown(fragment: str) -> str:
    """One rich-text HTML fragment as Markdown, wording unchanged (tables kept)."""
    import trafilatura
    wrapped = f"<html><body><article>{fragment}</article></body></html>"
    return (trafilatura.extract(wrapped, output_format="markdown", include_tables=True, include_links=False,
                                include_images=False, favor_recall=True) or "").strip()


def next_data_page(html: str) -> Optional[Dict[str, Any]]:
    """HDB's site (Next.js on Sitecore) renders its text with JavaScript, so a browser's "HTML only" save holds the
    article in the __NEXT_DATA__ JSON, not in the page body. Returns {"body", "url", "title"} from that JSON (the
    main rich-text fields, then each accordion as a "### title" section, in page order), or None for other pages."""
    m = _NEXT_DATA.search(html)
    if not m:
        return None
    try:
        props = json.loads(m.group(1))["props"]["pageProps"]
    except (ValueError, KeyError, TypeError):
        return None
    parts: List[str] = []
    seen: set = set()

    def emit(text: str) -> None:
        text = text.strip()
        if text and text not in seen:
            seen.add(text)
            parts.append(text)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "bodyContentVal" in node or "titleVal" in node:          # an accordion section
                if node.get("titleVal"):
                    emit(f"### {node['titleVal'].strip()}")
                if node.get("bodyContentVal"):
                    emit(_fragment_markdown(node["bodyContentVal"]))
                elif node.get("descVal"):
                    emit(node["descVal"])
                return
            for key, value in node.items():
                if key not in _SKIP_KEYS:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str) and len(node) > 40 and _RICH_HTML.search(node):
            emit(_fragment_markdown(node))

    walk(props.get("layoutData", {}))
    walk(props.get("componentProps", {}))
    context = ((props.get("layoutData") or {}).get("sitecore") or {}).get("context") or {}
    route = ((props.get("layoutData") or {}).get("sitecore") or {}).get("route") or {}
    path = context.get("itemPath") or ""
    title = None
    for key in ("pageTitle", "title", "Title", "heading"):
        value = ((route.get("fields") or {}).get(key) or {})
        if isinstance(value, dict) and isinstance(value.get("value"), str) and value["value"].strip():
            title = re.sub(r"<[^>]+>", "", value["value"]).strip()
            break
    return {"body": "\n\n".join(p for p in parts if p),
            "url": f"https://www.hdb.gov.sg{path}" if path.startswith("/") else None,
            "title": title}


def page_url(html: str) -> Optional[str]:
    """The page's own address: <link rel=canonical>, og:url, Next.js route, or Chrome's "saved from url" note."""
    for pattern in (r'<link[^>]+rel="canonical"[^>]+href="([^"]+)"', r'<meta[^>]+property="og:url"[^>]+content="([^"]+)"',
                    r"saved from url=\(\d+\)(\S+)"):
        m = re.search(pattern, html)
        if m and m.group(1).startswith("http"):
            import html as _html
            return _html.unescape(m.group(1).strip())
    page = next_data_page(html)
    return page["url"] if page else None


def extract_html(path: Path) -> Tuple[str, Optional[str], Optional[str]]:
    """(markdown body, page title, metadata date) of a saved HTML page."""
    import trafilatura
    html = path.read_text(encoding="utf-8", errors="ignore")
    body = trafilatura.extract(html, output_format="markdown", include_tables=True, include_links=False,
                               include_images=False, include_comments=False, favor_recall=True) or ""
    page = next_data_page(html)
    # whichever holds the article: the rendered HTML (CPF, a "complete" save) or the Next.js data (HDB "HTML only")
    if page and len(page["body"].split()) > len(body.split()):
        body = page["body"]
    meta = trafilatura.extract_metadata(html)
    title = clean_title(meta.title if meta else None)
    if not title:
        # a bare site name ("CPFB") as the <title>: use the page's first heading instead
        heading = re.search(r"^#{1,3} (.+)$", body, re.M)
        title = clean_title(heading.group(1)) if heading else None
    return body, title, (meta.date if meta else None)


SITE_NAMES = r"(?:CPFB|CPF Board|HDB|Housing & Development Board|Housing and Development Board|MAS|IRAS|MND)"


def clean_title(title: Optional[str]) -> Optional[str]:
    """The page's own title without the site name before or after it ("CPFB | X", "X | HDB")."""
    if not title:
        return None
    title = re.sub(rf"^\s*{SITE_NAMES}\s*[|\-–:]\s*", "", title)
    title = re.sub(rf"\s*[|\-–]\s*{SITE_NAMES}\b.*$", "", title).strip()
    return None if not title or re.fullmatch(SITE_NAMES, title) else title


def extract_pdf(path: Path) -> Tuple[str, Optional[str], Optional[str]]:
    """(markdown body, title, None) of a page saved as PDF."""
    import pymupdf4llm
    body = pymupdf4llm.to_markdown(str(path))
    first = next((ln.strip("# ").strip() for ln in body.splitlines() if ln.strip()), None)
    return body, first, None


_FOOTNOTE_REF = re.compile(r"<sup>[^<]*</sup>")
_ESCAPED = re.compile(r"\\([\[\]()])")


def _cells(row: str) -> List[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def merge_banner_headers(text: str) -> str:
    """HDB tables often have a banner header ("| | Classification of flats | | |") above the real column names
    ("| | Standard | Plus | Prime |", the first body row). Read as written, every cell is labelled with the banner
    and the column names are lost ("Classification of flats: No; : Yes"). Make the column-name row the header, with
    the banner kept in its empty first cell. Cell wording is unchanged."""
    lines = text.splitlines()
    out: List[str] = []
    i = 0
    while i < len(lines):
        if (i + 2 < len(lines) and lines[i].lstrip().startswith("|") and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1])
                and lines[i + 2].lstrip().startswith("|")):
            head, names = _cells(lines[i]), _cells(lines[i + 2])
            filled = [c for c in head if c]
            if len(head) == len(names) >= 3 and len(filled) == 1 and not names[0] and sum(1 for c in names[1:] if c) >= 2:
                names[0] = filled[0]
                out.extend(["| " + " | ".join(names) + " |", lines[i + 1]])
                i += 3
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def lift_footnote_rows(text: str) -> str:
    """HDB puts a table's footnotes in a row of their own ("| [1] Prime flats include ... | | |"). Read as a row they
    become a fact about nothing; they are moved below the table as a "Notes:" paragraph, wording unchanged."""
    out: List[str] = []
    notes: List[str] = []
    for line in text.splitlines() + [""]:
        is_row = line.lstrip().startswith("|")
        if is_row:
            cells = _cells(line)
            filled = [c for c in cells if c]
            if len(filled) == 1 and re.match(r"^\[\d+\]", filled[0]):
                notes.append(filled[0])
                continue
        elif notes:
            out.extend(["", "Notes: " + " ".join(notes), ""])
            notes = []
        out.append(line)
    return "\n".join(out).rstrip("\n")


def clean_body(body: str) -> str:
    """Drop UI lines and demote headings so the document title stays the only '# ' line (the ingest's hierarchy
    starts a new document at '# '). The wording of every kept line is unchanged."""
    lines: List[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if BOILERPLATE.match(stripped.strip("*_ ")):
            continue
        if re.match(r"^# ", line):
            line = "#" + line           # '# X' -> '## X'
        heading = re.match(r"^(#{1,6} )(.*)$", line)
        if heading:
            # HDB bolds some headings ("## **Eligibility conditions**"); a heading is already emphasis, and the markup
            # would end up inside every chunk's section breadcrumb
            text = re.sub(r"[*_]{2,}", "", heading.group(2)).strip()
            if not text:
                continue
            line = heading.group(1) + text
        lines.append(line.rstrip())
    text = "\n".join(lines)
    # footnote markers (<sup>[1]</sup>) are references, not words; escaped brackets read as plain brackets
    text = _ESCAPED.sub(r"\1", _FOOTNOTE_REF.sub("", text))
    text = re.sub(r"(\S)[ \t]+\*\*(?=\s*\||\s*$)", r"\1**", text, flags=re.M)   # "**term **" left by a removed marker
    text = lift_footnote_rows(merge_banner_headers(text))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def raw_body(source: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[str]]:
    path = raw_file(source["id"])
    if path is None:
        raise FileNotFoundError(f"{source['id']}: no capture in raw/ (run fetch_sources.py)")
    body, title, meta_date = extract_pdf(path) if path.suffix == ".pdf" else extract_html(path)
    return clean_body(body), title, meta_date


SITE_BANNER_MIN_PAGES = 3
SITE_BANNER_MIN_CHARS = 40


def site_banners(bodies: Dict[str, Tuple[str, str]]) -> set:
    """Long lines that repeat on SITE_BANNER_MIN_PAGES or more pages of one publisher: site notices and furniture
    ("Our CPF hotlines are temporarily unavailable ..."), not policy text. bodies: id -> (publisher, body)."""
    from collections import Counter
    seen: Dict[str, Counter] = {}
    for publisher, body in bodies.values():
        lines = {ln.strip() for ln in body.splitlines() if len(ln.strip()) >= SITE_BANNER_MIN_CHARS}
        seen.setdefault(publisher, Counter()).update(lines)
    return {line for counts in seen.values() for line, n in counts.items() if n >= SITE_BANNER_MIN_PAGES}


def build_document(source: Dict[str, Any], manifest_entry: Dict[str, Any], body: str, title: Optional[str],
                   meta_date: Optional[str], banners: set) -> Tuple[str, Dict[str, Any]]:
    body = "\n".join(ln for ln in body.splitlines() if ln.strip() not in banners)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    words = len(body.split())
    if words < MIN_WORDS:
        raise ValueError(f"{source['id']}: only {words} words extracted from {path.name}; re-capture the page")
    retrieved = (manifest_entry.get("captured_at") or "")[:10]
    effective = page_date(body, meta_date)
    title = title or source["id"].replace("_", " ").title()
    header = [
        f"# {title}",
        "",
        f"**Source URL:** {source['url']}",
        f"**Publisher:** {source.get('publisher', '')}",
        f"**Policy Category:** {source['policy_category']}",
        f"**Retrieved:** {retrieved}",
        f"**Page Date:** {effective or 'not stated on the page'}",
        f"**Capture Method:** {manifest_entry.get('method', '')}",
        f"**Content SHA256:** {manifest_entry.get('sha256', '')}",
        "",
        "---",
        "",
    ]
    # the ingest reads header lines until the first '## '; a page that starts with prose gets its title as a section
    if not body.lstrip().startswith("## "):
        body = f"## {title}\n\n{body}"
    stats = {"id": source["id"], "words": words, "title": title, "effective": effective, "retrieved": retrieved}
    return "\n".join(header) + body + "\n", stats


def extract_all(sources: List[Dict[str, Any]]) -> Tuple[List[Tuple[str, str]], List[str]]:
    manifest = load_manifest()
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    written: List[Tuple[str, str]] = []
    problems: List[str] = []
    sources = [s for s in sources if not s.get("exclude")]
    extracted: Dict[str, Tuple[str, Optional[str], Optional[str]]] = {}
    for source in sources:
        sid = source["id"]
        if sid not in manifest or raw_file(sid) is None:
            if not source.get("optional"):
                problems.append(f"not captured: {sid}")
            continue
        extracted[sid] = raw_body(source)
    banners = site_banners({sid: (s.get("publisher", ""), extracted[s["id"]][0])
                            for s in sources if (sid := s["id"]) in extracted})
    for index, source in enumerate(sources, start=1):
        sid = source["id"]
        if sid not in extracted:
            continue
        try:
            text, stats = build_document(source, manifest[sid], *extracted[sid], banners)
        except (FileNotFoundError, ValueError) as exc:
            problems.append(str(exc))
            continue
        name = f"{index:02d}_{sid}.md"
        (EXTRACTED_DIR / name).write_text(text, encoding="utf-8")
        written.append((name, sid))
        print(f"  wrote {name:<42} {stats['words']:>5} words  effective {stats['effective']}  ({stats['title'][:50]})")
    for stale in EXTRACTED_DIR.glob("*.md"):
        if stale.name not in {n for n, _ in written}:
            stale.unlink()
    return written, problems


def install(written: List[Tuple[str, str]]) -> None:
    """Make the extracted documents the RAG corpus, replacing the previous install."""
    if not written:
        raise SystemExit("Nothing extracted; not installing.")
    if CORPUS_DIR.exists():
        shutil.rmtree(CORPUS_DIR)
    CORPUS_DIR.mkdir(parents=True)
    for name, _ in written:
        shutil.copy2(EXTRACTED_DIR / name, CORPUS_DIR / name)
    print(f"  installed {len(written)} documents into {CORPUS_DIR.relative_to(DATA_DIR.parent)}; "
          f"now run: .venv/bin/python data/ingest_hdb_policy_rag.py")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract captured official pages into verbatim Markdown documents.")
    parser.add_argument("--install", action="store_true",
                        help="install the extracted documents as data/policy_documents (old ones -> *_legacy)")
    parser.add_argument("--allow-missing", action="store_true",
                        help="install even when some sources are not captured yet")
    args = parser.parse_args(argv)
    written, problems = extract_all(load_sources())
    for problem in problems:
        print(f"  ! {problem}")
    print(f"{len(written)} documents extracted, {len(problems)} sources missing or failed")
    if args.install:
        if problems and not args.allow_missing:
            print("Not installing: some sources are missing (capture them, or pass --allow-missing).")
            return 1
        install(written)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
