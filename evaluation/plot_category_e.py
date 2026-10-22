"""
Accuracy-vs-latency figure for the preliminary Category E run (evaluation/plot_category_e.py). Member 4.
Emphasis form: Variant C (the proposed architecture) in the accent blue, A and B in greys, a letter in
every dot so identity never rests on colour. Two panels, each with its own single axis (no dual axis):
  left  - mean Tool F1, constraint satisfaction (CSR) and map precision per variant, 0..1
  right - seconds per query, one dot per query, median tick labelled
Static SVG (no dependencies); hover titles on every mark; the markdown tables are the table view.

    .venv/bin/python evaluation/plot_category_e.py evaluation/results/prelim_vllm_Qwen_Qwen3-30B-A3B-GPTQ-Int4.jsonl
"""
from __future__ import annotations

import json
import math
import sys
from html import escape
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR = {"C": "#2a78d6", "B": "#52514e", "A": "#898781"}       # accent on the proposed graph, greys for baselines
NAME = {"A": "A · monolith (11 tools)", "B": "B · 2-agent split", "C": "C · 5-agent graph"}
FONT = "font-family='system-ui, -apple-system, Segoe UI, Helvetica, Arial, sans-serif'"


def load(path: Path) -> List[Dict[str, Any]]:
    latest: Dict[tuple, Dict[str, Any]] = {}
    for l in path.read_text().splitlines():
        if l.strip():
            r = json.loads(l); latest[(r["variant"], r["id"])] = r
    return [r for r in latest.values() if r.get("status") == "ok"]


def mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def nice_step(v: float) -> int:
    """Round tick step so the axis carries 4 to 6 ticks: 5, 10, 20 or 50 seconds."""
    for step in (5, 10, 20, 50):
        if v / step <= 5:
            return step
    return 100


def nice_max(v: float) -> float:
    step = nice_step(v)
    return max(step, math.ceil(v / step) * step)


def build_svg(rows: List[Dict[str, Any]], model: str) -> str:
    vs = [v for v in ("A", "B", "C") if any(r["variant"] == v for r in rows)]
    metric = {v: {
        "Task success": mean((1.0 if r["scores"].get("task_success") else 0.0) for r in rows if r["variant"] == v),
        "Tool selection F1": mean(r["scores"]["tool_selection"]["f1"] for r in rows if r["variant"] == v),
        "Constraint satisfaction": mean(r["scores"]["csr"]["rate"] for r in rows if r["variant"] == v),
        "Map shows only matches": mean((r["scores"].get("map") or {}).get("rate") for r in rows if r["variant"] == v),
    } for v in vs}
    lat = {v: [r for r in rows if r["variant"] == v] for v in vs}
    med = {v: median([r["latency_s"] for r in lat[v]]) for v in vs}

    wins = {v: sum(1 for r in rows if r["variant"] == v and r["scores"].get("task_success")) for v in vs}
    n_each = {v: sum(1 for r in rows if r["variant"] == v) for v in vs}
    others = [v for v in vs if v != "C"]
    title = (f"The 5-agent graph completes {wins['C']} of {n_each['C']} Category E tasks; the baselines complete at most "
             f"{max(wins[v] for v in others)}" if "C" in vs and others else "Accuracy and latency by architecture")
    n_q = len({r["id"] for r in rows})
    sub = (" · ".join(f"{v}: median {med[v]:.0f} s per query" for v in ("C", "B", "A") if v in vs)
           + f" · {n_q} Category E queries × {len(vs)} variants")

    W, H = 960, 440
    L0, L1 = 210, 470            # accuracy panel x-range (0..1)
    R0, R1 = 600, 920            # latency panel x-range (0..max s)
    acc_y = [146, 196, 246, 296]
    lat_y = [156, 221, 286]
    tmax = nice_max(max(r["latency_s"] for r in rows) * 1.05)
    xa = lambda v: L0 + v * (L1 - L0)
    xr = lambda s: R0 + s / tmax * (R1 - R0)
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' height='{H}' role='img' "
           f"aria-label='{escape(title)}' {FONT}>",
           f"<rect width='{W}' height='{H}' fill='{SURFACE}'/>",
           f"<text x='24' y='34' font-size='17' font-weight='600' fill='{INK}'>{escape(title)}</text>",
           f"<text x='24' y='56' font-size='13' fill='{INK2}'>{escape(sub)}</text>"]
    # legend (always present for >= 2 series)
    lx = 24
    for v in ("C", "B", "A"):
        if v in vs:
            out.append(f"<circle cx='{lx + 8}' cy='84' r='8' fill='{COLOR[v]}'/><text x='{lx + 8}' y='88' font-size='10.5' "
                       f"font-weight='700' fill='#ffffff' text-anchor='middle'>{v}</text>"
                       f"<text x='{lx + 22}' y='88' font-size='12.5' fill='{INK}'>{escape(NAME[v])}</text>")
            lx += 205
    # panel headings
    out.append(f"<text x='{L0}' y='116' font-size='12.5' font-weight='600' fill='{INK2}'>Accuracy (mean over queries, 1 = best)</text>")
    out.append(f"<text x='{R0}' y='116' font-size='12.5' font-weight='600' fill='{INK2}'>Seconds per query (tick = median)</text>")
    # gridlines + axes
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f"<line x1='{xa(t):.1f}' x2='{xa(t):.1f}' y1='126' y2='318' stroke='{GRID}' stroke-width='1'/>"
                   f"<text x='{xa(t):.1f}' y='336' font-size='11' fill='{MUTED}' text-anchor='middle'>{t:g}</text>")
    step = nice_step(tmax)
    for i in range(int(tmax // step) + 1):
        s = i * step
        out.append(f"<line x1='{xr(s):.1f}' x2='{xr(s):.1f}' y1='126' y2='318' stroke='{GRID}' stroke-width='1'/>"
                   f"<text x='{xr(s):.1f}' y='336' font-size='11' fill='{MUTED}' text-anchor='middle'>{s:g}</text>")
    out.append(f"<line x1='{L0}' x2='{L1}' y1='318' y2='318' stroke='{AXIS}' stroke-width='1'/>")
    out.append(f"<line x1='{R0}' x2='{R1}' y1='318' y2='318' stroke='{AXIS}' stroke-width='1'/>")
    out.append(f"<text x='{(R0 + R1) / 2}' y='358' font-size='11.5' fill='{INK2}' text-anchor='middle'>seconds per query, end to end</text>")
    # accuracy rows: one row per metric, dots dodged vertically so equal scores never hide each other
    dodge = {"C": -11, "B": 0, "A": 11}
    for y, name in zip(acc_y, ["Task success", "Tool selection F1", "Constraint satisfaction", "Map shows only matches"]):
        weight = "600" if name == "Task success" else "400"
        out.append(f"<text x='{L0 - 12}' y='{y + 4}' font-size='12.5' font-weight='{weight}' fill='{INK}' text-anchor='end'>{escape(name)}</text>")
        for v in ("A", "B", "C"):
            if v not in vs:
                continue
            val = metric[v][name]
            yy = y + dodge[v]
            if val is None:
                out.append(f"<text x='{L0 + 4}' y='{yy + 4}' font-size='10.5' fill='{MUTED}'>{v}: no block recommended</text>")
                continue
            out.append(f"<g><title>{escape(f'{NAME[v]}: {name} {val:.2f}')}</title>"
                       f"<circle cx='{xa(val):.1f}' cy='{yy}' r='7' fill='{COLOR[v]}' stroke='{SURFACE}' stroke-width='2'/>"
                       f"<text x='{xa(val):.1f}' y='{yy + 3.5}' font-size='9.5' font-weight='700' fill='#ffffff' text-anchor='middle'>{v}</text></g>")
            if v == "C":
                out.append(f"<text x='{xa(val) + 12:.1f}' y='{yy + 4}' font-size='11' font-weight='600' fill='{INK}'>{val:.2f}</text>")
    # latency rows: one row per variant
    for y, v in zip(lat_y, ("C", "B", "A")):
        if v not in vs:
            continue
        out.append(f"<text x='{R0 - 12}' y='{y + 4}' font-size='12.5' fill='{INK}' text-anchor='end'>{v}</text>")
        for i, r in enumerate(sorted(lat[v], key=lambda r: r["id"])):
            jy = y + ((i % 5) - 2) * 4
            rid, rlat = r["id"], r["latency_s"]
            out.append(f"<g><title>{escape(f'{v} {rid}: {rlat:.1f} s')}</title>"
                       f"<circle cx='{xr(r['latency_s']):.1f}' cy='{jy}' r='4.5' fill='{COLOR[v]}' fill-opacity='0.65' stroke='{SURFACE}' stroke-width='1'/></g>")
        m = med[v]
        out.append(f"<line x1='{xr(m):.1f}' x2='{xr(m):.1f}' y1='{y - 16}' y2='{y + 16}' stroke='{COLOR[v]}' stroke-width='2.5'/>"
                   f"<text x='{xr(m):.1f}' y='{y - 21}' font-size='11' font-weight='600' fill='{INK}' text-anchor='middle'>{m:.0f} s</text>")
    out.append(f"<text x='24' y='{H - 30}' font-size='11' fill='{MUTED}'>{escape(f'Source: preliminary run, {model} served by vLLM on the SMU Origami cluster, temperature 0.')}</text>")
    out.append(f"<text x='24' y='{H - 14}' font-size='11' fill='{MUTED}'>SQL, finance and policy tools are mocks; spatial tools are real; no runtime auditor yet (Member 5).</text>")
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    path = Path(sys.argv[1]).resolve()
    rows = load(path)
    model = next((r.get("model") for r in rows if r.get("model")), "?")
    out_dir = PROJECT_ROOT / "evaluation" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (path.stem + "_accuracy_latency.svg")
    out.write_text(build_svg(rows, model))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
