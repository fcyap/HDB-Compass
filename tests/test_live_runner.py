"""
The live runner fails fast (tests/test_live_runner.py). Owned by Member 4.
On 31 Oct the laptop's VPN dropped during a live run: the tunnel's local port stayed open, every model call in a row
waited out its timeout and retries, and two lost rows took about 4 hours. evaluation/run_category_e_prelim.py now asks
the model server one quick question (GET <base_url>/models) before each row of a live run and stops at once when it
fails to answer twice in a row. These tests use a stand-in model server on 127.0.0.1 (a random free port, never the
network) that answers a few times and then stops; the rows themselves run with no model (LLM_PROFILE=none).

    LLM_PROFILE=none .venv/bin/python -m pytest tests/test_live_runner.py -q
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

os.environ["LLM_PROFILE"] = "none"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import run_category_e_prelim as rp  # noqa: E402


class FakeModelServer:
    """Answers GET /v1/models as the script says, one step per request, then repeats the last step:
    "ok" (200), "401" (refused), "drop" (closes without an answer) or "hang" (never answers, like a dead tunnel)."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []          # (time, Authorization header) per request
        self.first_unanswered = None
        self.release = threading.Event()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):   # keep the test output clean
                pass

            def do_GET(self):
                n = len(owner.requests)
                owner.requests.append((time.monotonic(), self.headers.get("Authorization")))
                step = owner.script[min(n, len(owner.script) - 1)]
                if step != "ok" and owner.first_unanswered is None:
                    owner.first_unanswered = time.monotonic()
                if step == "hang":
                    owner.release.wait(30)
                    return
                if step == "drop":
                    self.close_connection = True
                    return
                code = 200 if step == "ok" else int(step)
                body = json.dumps({"object": "list", "data": [{"id": "fake-model"}]}).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def answered(self) -> int:
        return sum(1 for i in range(len(self.requests)) if self.script[min(i, len(self.script) - 1)] == "ok")

    def close(self):
        self.release.set()
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def fake_server():
    made = []

    def make(script):
        made.append(FakeModelServer(script))
        return made[-1]
    yield make
    for s in made:
        s.close()


@pytest.fixture
def fast_probe(monkeypatch, tmp_path):
    """Short probe waits, and results in the test's scratch folder (never evaluation/results)."""
    monkeypatch.setattr(rp, "PROBE_TIMEOUT_S", 0.5)
    monkeypatch.setattr(rp, "PROBE_PAUSE_S", 0.1)
    monkeypatch.setattr(rp, "RESULTS_DIR", tmp_path)
    return tmp_path


def live(url: str) -> dict:
    return {"name": "vllm", "model": "fake-model", "base_url": url, "api_key": "test-key", "system_suffix": ""}


def run_main(monkeypatch, profile: dict, *args: str):
    monkeypatch.setattr(rp, "active_profile", lambda: profile)
    monkeypatch.setattr(sys, "argv", ["run_category_e_prelim.py", *args])
    rp.main()


def rows(folder: Path) -> list:
    files = sorted(folder.glob("prelim_*.jsonl"))
    return [json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]


# --- the quick question ---------------------------------------------------------------------------------------------
def test_an_answering_server_passes_and_gets_the_api_key(fake_server, fast_probe):
    srv = fake_server(["ok"])
    assert rp.server_answers(live(srv.url)) == (True, "HTTP 200")
    assert srv.requests[0][1] == "Bearer test-key"


def test_a_refusing_server_does_not_pass(fake_server, fast_probe):
    srv = fake_server(["401"])
    assert rp.server_answers(live(srv.url)) == (False, "HTTP 401")


def test_a_closed_port_fails_at_once():
    with socket.socket() as s:   # a free port with nothing listening on it
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    t0 = time.monotonic()
    ok, why = rp.server_answers(live(f"http://127.0.0.1:{port}/v1"), timeout=2)
    assert not ok and "Connection" in why and time.monotonic() - t0 < 2


def test_a_hanging_server_fails_within_the_probe_timeout(fake_server, fast_probe):
    srv = fake_server(["hang"])
    t0 = time.monotonic()
    ok, why = rp.server_answers(live(srv.url))
    assert not ok and time.monotonic() - t0 < 2, why


def test_one_missed_answer_is_not_a_lost_server(fake_server, fast_probe):
    srv = fake_server(["drop", "ok"])
    assert rp.server_down(live(srv.url)) is None and len(srv.requests) == 2


def test_two_missed_answers_in_a_row_are(fake_server, fast_probe):
    srv = fake_server(["drop", "hang"])
    assert rp.server_down(live(srv.url)) and len(srv.requests) == 2


# --- the runner -----------------------------------------------------------------------------------------------------
IDS = "E1-01,E1-02,E1-03,E1-04"


def test_the_runner_stops_within_seconds_and_writes_no_row_it_did_not_run(fake_server, fast_probe, monkeypatch, capsys):
    srv = fake_server(["ok", "ok", "hang"])      # answers before two rows, then the tunnel goes dead
    with pytest.raises(SystemExit) as stop:
        run_main(monkeypatch, live(srv.url), "--variants", "C", "--ids", IDS, "--tag", "probe")
    stopped_at = time.monotonic()
    out = capsys.readouterr().out
    assert stop.value.code == 3                                      # run_live_eval.sh stops on this
    assert "MODEL_UNREACHABLE" in out and "did not answer twice" in out and "stopped after 2/4" in out
    assert stopped_at - srv.first_unanswered < 3                     # two 0.5 s tries, not hours
    assert srv.answered() == 2 and len(srv.requests) == 4            # one question per row, two at the end
    written = rows(fast_probe)
    assert [r["id"] for r in written] == ["E1-01", "E1-02"]          # only the rows that ran
    assert all(r["status"] == "ok" for r in written)


def test_the_resume_runs_the_rest_once_the_server_answers_again(fake_server, fast_probe, monkeypatch, capsys):
    gone = fake_server(["ok", "drop"])
    with pytest.raises(SystemExit):
        run_main(monkeypatch, live(gone.url), "--variants", "C", "--ids", IDS, "--tag", "probe")
    assert len(rows(fast_probe)) == 1
    back = fake_server(["ok"])
    run_main(monkeypatch, live(back.url), "--variants", "C", "--ids", IDS, "--tag", "probe")
    out = capsys.readouterr().out
    assert "1 rows already done, 3 to run" in out and "MODEL_UNREACHABLE" not in out.split("1 rows already done")[1]
    assert [r["id"] for r in rows(fast_probe)] == ["E1-01", "E1-02", "E1-03", "E1-04"]
    assert len(back.requests) == 3


def test_a_server_gone_before_the_first_row_runs_nothing(fake_server, fast_probe, monkeypatch, capsys):
    srv = fake_server(["drop"])
    with pytest.raises(SystemExit) as stop:
        run_main(monkeypatch, live(srv.url), "--variants", "C", "--ids", IDS, "--tag", "probe")
    assert stop.value.code == 3 and "stopped after 0/4" in capsys.readouterr().out
    assert rows(fast_probe) == []


def test_the_no_model_run_never_asks_a_server(fast_probe, monkeypatch):
    def must_not_run(*_a, **_k):
        raise AssertionError("the no-model run asked a model server")
    monkeypatch.setattr(rp, "server_answers", must_not_run)
    none = {"name": "none", "model": None, "base_url": None, "api_key": None}
    run_main(monkeypatch, none, "--variants", "C", "--ids", "E1-01", "--tag", "probe")   # no SystemExit
    assert [r["status"] for r in rows(fast_probe)] == ["ok"]


def test_rows_whose_model_calls_failed_still_stop_the_run(fake_server, fast_probe, monkeypatch, capsys):
    # the server answers the quick question but the rows lose the model: the existing two-row stop still applies
    srv = fake_server(["ok"])

    def lost(mod, item, variant, profile, run_id):
        return {"run": run_id, "variant": variant, "id": item["id"], "sub_use_case": item.get("sub_use_case"),
                "status": "model_unreachable", "latency_s": 0.0}
    monkeypatch.setattr(rp, "run_one", lost)
    with pytest.raises(SystemExit) as stop:
        run_main(monkeypatch, live(srv.url), "--variants", "C", "--ids", IDS, "--tag", "probe")
    assert stop.value.code == 3 and "stopped after 2/4" in capsys.readouterr().out
    assert [r["status"] for r in rows(fast_probe)] == ["model_unreachable"] * 2   # kept, and rerun on resume
