import json

from evaluation.merge_streams import main, merge_rows


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_last_row_per_variant_and_id_wins(tmp_path):
    first = tmp_path / "s0.jsonl"
    second = tmp_path / "s1.jsonl"
    _write(first, [{"variant": "A", "id": "A1-01", "status": "model_unreachable"},
                   {"variant": "B", "id": "A1-01", "status": "ok"}])
    _write(second, [{"variant": "A", "id": "A1-01", "status": "ok"},
                    {"variant": "A", "id": "A1-02", "status": "ok"}])
    rows = merge_rows([first, second])
    by_key = {(r["variant"], r["id"]): r["status"] for r in rows}
    assert by_key == {("A", "A1-01"): "ok", ("B", "A1-01"): "ok", ("A", "A1-02"): "ok"}


def test_cli_writes_merged_file(tmp_path, capsys):
    src = tmp_path / "s.jsonl"
    out = tmp_path / "out" / "merged.jsonl"
    _write(src, [{"variant": "C", "id": "F1-01", "status": "ok"}])
    assert main([str(src), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8").splitlines()[0])["id"] == "F1-01"
    assert "1 rows (1 ok)" in capsys.readouterr().out
