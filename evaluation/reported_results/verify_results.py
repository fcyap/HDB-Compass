"""Recompute the reported Experiment 1 comparisons from tracked CSVs (standard library only).

Run from any directory: python evaluation/reported_results/verify_results.py
These checks validate the saved artifacts, not the benchmark's answer keys or model quality.
"""
from pathlib import Path
import csv
import itertools
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent
TAGS = ('q38fp8', 'q330b90', 'gptoss90', 'gemma4nvfp90', 'haiku55', 'q38m', 'q3m', 'gptm', 'gemm')
# q38fp8, q330b90, gptoss90, gemma4nvfp90: initial open-weight runs (code 23c9076).
# haiku55: hosted reference row (Anthropic API, code a6e8557).
# q38m (with q38m-noauditor), q3m, gptm, gemm: re-runs of the four open-weight models on code 4ff2767.
RERUNS = (('q38fp8', 'q38m'), ('q330b90', 'q3m'), ('gptoss90', 'gptm'), ('gemma4nvfp90', 'gemm'))
AUDITOR_PAIRS = (('q38fp8', 'q38fp8-noauditor'), ('q38m', 'q38m-noauditor'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(tag):
    with (ROOT / f'{tag}.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(len({(r['variant'], r['id']) for r in rows}) == len(rows), f'{tag}: duplicate rows')
    for r in rows:
        require(r['status'] == 'ok', f'{tag}: non-ok row')
        require(r['task_success'] in ('True', 'False'), f'{tag}: invalid success value')
        require(r['category'] == r['id'][0], f'{tag}: wrong category')
        r['passed'] = r['task_success'] == 'True'
        for key in ('input_tokens', 'output_tokens', 'total_tokens'):
            r[key] = int(r[key])
        require(r['input_tokens'] + r['output_tokens'] == r['total_tokens'], f'{tag}: token sum')
    return rows


def wilson(k, n):
    z = 1.96
    p = k / n
    denominator = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denominator
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return centre-half, centre+half


def paired(first, second, without_f=False):
    a = {r['id']: r['passed'] for r in first if not without_f or r['category'] != 'F'}
    b = {r['id']: r['passed'] for r in second if not without_f or r['category'] != 'F'}
    require(a.keys() == b.keys(), 'Paired question sets differ')
    x = sum(a[i] and not b[i] for i in a)
    y = sum(b[i] and not a[i] for i in a)
    n = x+y
    p = min(1, 2*sum(math.comb(n, k) for k in range(min(x, y)+1))/2**n)
    return f'{x}/{y}, p={p:.6g}'


def main():
    all_rows = {tag: load(tag) for tag in TAGS}
    print('| Run | Variant | Pass /90 | Without F /75 (Wilson 95%) | Mean tokens |')
    print('|---|---|---|---|---|')
    for tag, rows in all_rows.items():
        report = json.loads((ROOT / f'architecture_eval_{tag}.json').read_text())
        for v in 'ABC':
            group = [r for r in rows if r['variant'] == v]
            require(len(group) == 90, f'{tag}/{v}: expected 90 rows')
            require(all(sum(r['category'] == c for r in group) == 15 for c in 'ABCDEF'), f'{tag}/{v}: category counts')
            require({r['id'] for r in group} == {r['id'] for r in all_rows['q38fp8'] if r['variant'] == v}, f'{tag}/{v}: question sets')
            successes = sum(r['passed'] for r in group)
            expected = report['experiment1'][v]
            require(successes == expected['task_success'] and len(group) == expected['n'], f'{tag}/{v}: summary disagreement')
            require(math.isclose(statistics.mean(float(r['tool_f1']) for r in group), expected['tool_selection_f1']), f'{tag}/{v}: Tool F1 disagreement')
            for c in 'ABCDEF':
                require(sum(r['passed'] for r in group if r['category'] == c) == expected['by_category'][c]['task_success'], f'{tag}/{v}/{c}: category disagreement')
            non_f = [r for r in group if r['category'] != 'F']
            k = sum(r['passed'] for r in non_f)
            lo, hi = wilson(k, len(non_f))
            mean_tokens = statistics.mean(r['total_tokens'] for r in group)
            print(f'| {tag} | {v} | {successes}/90 | {k}/75 ({lo:.1%}–{hi:.1%}) | {mean_tokens:,.0f} |')
        for first, second in (('C','A'), ('C','B'), ('B','A')):
            a = [r for r in rows if r['variant'] == first]
            b = [r for r in rows if r['variant'] == second]
            print(f'{tag} {first} vs {second}: all {paired(a,b)}; without F {paired(a,b,True)}')
    for first, second in itertools.combinations(TAGS, 2):
        same = [paired([r for r in all_rows[first] if r['variant'] == v], [r for r in all_rows[second] if r['variant'] == v]) for v in 'ABC']
        print(f'Same variant {first} vs {second}: A {same[0]}; B {same[1]}; C {same[2]}')
    for initial, rerun in RERUNS:
        for v in 'ABC':
            a = [r for r in all_rows[initial] if r['variant'] == v]
            b = [r for r in all_rows[rerun] if r['variant'] == v]
            d_lat = statistics.mean(float(r['latency_s']) for r in b) - statistics.mean(float(r['latency_s']) for r in a)
            d_tok = statistics.mean(r['total_tokens'] for r in b) - statistics.mean(r['total_tokens'] for r in a)
            print(f'Before/after {initial} -> {rerun} {v}: {sum(r["passed"] for r in a)} -> {sum(r["passed"] for r in b)} of 90; '
                  f'fail->pass/pass->fail all {paired(b, a)}; without F {paired(b, a, True)}; '
                  f'mean latency {d_lat:+.1f} s (CSV, rounded); mean total tokens {d_tok:+,.0f}')
    off_rows = {}
    for on_tag, off_tag in AUDITOR_PAIRS:
        on = [r for r in all_rows[on_tag] if r['variant'] == 'C']
        off = load(off_tag)
        require(len(off) == 90 and {r['variant'] for r in off} == {'C'}, f'Expected 90 auditor-off C rows in {off_tag}')
        off_rows[off_tag] = off
        print(f'Auditor on vs off ({on_tag}): {paired(on, off)}; without F {paired(on, off, True)}')
        off_by_id = {r['id']: r for r in off}
        deltas = [r['total_tokens']-off_by_id[r['id']]['total_tokens'] for r in on]
        print(f'Auditor extra tokens ({on_tag}): mean {statistics.mean(deltas):.3f}; median {statistics.median(deltas):g}; upper-middle {sorted(deltas)[len(deltas)//2]}')
    a, b = off_rows['q38fp8-noauditor'], off_rows['q38m-noauditor']
    print(f'Before/after q38fp8-noauditor -> q38m-noauditor C: {sum(r["passed"] for r in a)} -> {sum(r["passed"] for r in b)} of 90; '
          f'fail->pass/pass->fail all {paired(b, a)}; without F {paired(b, a, True)}')
    total = sum(len(rows) for rows in all_rows.values()) + sum(len(rows) for rows in off_rows.values())
    print(f'Verified {total:,} tracked rows and agreement with per-model JSON task counts and Tool F1.')


if __name__ == '__main__':
    main()
