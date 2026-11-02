"""
Member 4's last loose ends before the code freeze (tests/test_final_loose_ends.py). Owned by Member 4.

  * "4 room" straight after a question about one block narrows that block's answer; a fuller message is a new search.
  * The evaluation runs only on our own open model (none, ollama, vllm), never a hosted or "auto" profile.
  * cluster/run_live_eval.sh: KEEP_JOB=0 cancels the GPU job (on 1 Nov it kept job 347095 running).
  * cluster/demo_start.sh: frontend packages are installed again when package.json changed after the last install.

    LLM_PROFILE=none .venv/bin/python -m pytest tests/test_final_loose_ends.py -q
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

os.environ["LLM_PROFILE"] = "none"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import arch_3_specialist_5agent as C  # noqa: E402
from agents import supervisor as sv  # noqa: E402


def app(query, previous=None):
    return C.graph.invoke(sv.initial_state(query=query, variant="C", map_state={}, previous=previous, fast_path=True))


@pytest.fixture(scope="module")
def block_question():
    os.environ["HDB_ADDONS"] = "app"
    try:
        first = app("Resale price for Tampines Blk 884?")
        yield first
    finally:
        os.environ.pop("HDB_ADDONS", None)


# --- 1. "4 room" after a block question ----------------------------------------------------------------------------
@pytest.mark.parametrize("reply,flat", [("4 room", "4 ROOM"), ("3-room?", "3 ROOM"), ("what about 4 room", "4 ROOM")])
def test_a_flat_type_after_a_block_question_narrows_that_block(block_question, reply, flat, monkeypatch):
    monkeypatch.setenv("HDB_ADDONS", "app")
    out = app(reply, block_question)
    c = out["user_constraints"]
    assert c["wants_block_history"] and c["addresses"] and c["flat_types"] == [flat] and c["towns"] == []
    tools = [t["tool"] for t in out["tool_call_log"]]
    assert "get_block_transaction_history" in tools and "query_resale_blocks_by_criteria" not in tools
    assert "Blk 884 TAMPINES ST 83" in out["draft_response"]
    other = "3 ROOM" if flat == "4 ROOM" else "4 ROOM"
    assert f"{other}:" not in out["draft_response"]


@pytest.mark.parametrize("reply", ["4 room in Bedok under 600k", "4-room in Tampines under $600k"])
def test_a_fuller_message_is_a_new_search(block_question, reply, monkeypatch):
    monkeypatch.setenv("HDB_ADDONS", "app")
    out = app(reply, block_question)
    assert not out["user_constraints"].get("wants_block_history")
    assert "query_resale_blocks_by_criteria" in [t["tool"] for t in out["tool_call_log"]]


def test_without_an_earlier_block_question_nothing_is_narrowed():
    c = {"flat_types": ["4 ROOM"], "towns": ["TAMPINES"]}
    sv.narrow_block_question(c, {"towns": ["TAMPINES"], "wants_market_snapshot": True}, "4 room")
    assert c == {"flat_types": ["4 ROOM"], "towns": ["TAMPINES"]}


# --- 2. only our own open model ------------------------------------------------------------------------------------
@pytest.mark.parametrize("profile", ["groq", "openai", "gemini", "compatible", "auto", "GROQ"])
def test_hosted_and_automatic_profiles_are_refused(profile, monkeypatch):
    from evaluation.run_category_e_prelim import require_open_model_profile
    monkeypatch.setenv("LLM_PROFILE", profile)
    with pytest.raises(SystemExit, match="hosted or automatic profile"):
        require_open_model_profile()


@pytest.mark.parametrize("profile", ["none", "ollama", "vllm"])
def test_our_own_profiles_run(profile, monkeypatch):
    from evaluation.run_category_e_prelim import require_open_model_profile
    monkeypatch.setenv("LLM_PROFILE", profile)
    assert require_open_model_profile() == profile


def test_the_refusal_comes_before_any_hosted_service_is_contacted():
    out = subprocess.run([sys.executable, "evaluation/run_category_e_prelim.py", "--limit", "1", "--tag", "refusal"],
                         cwd=ROOT, env={**{k: v for k, v in os.environ.items() if k != "HDB_ADDONS"}, "LLM_PROFILE": "groq"},
                         capture_output=True, text=True, timeout=120)
    assert out.returncode != 0 and "hosted or automatic profile" in out.stderr
    assert "GROQ_API_KEY" not in out.stderr


# --- 3. KEEP_JOB=0 cancels the job -----------------------------------------------------------------------------------
def _shell_function(script: str, name: str) -> str:
    text = (ROOT / "cluster" / script).read_text()
    m = re.search(rf"^{name}\(\) \{{.*?^}}$|^{name}\(\) \{{[^\n]*\}}$", text, re.M | re.S)
    assert m, f"{name} not found in {script}"
    return m.group(0)


@pytest.mark.skipif(sys.platform == "win32", reason="Bash shell scripts run on POSIX / Linux cluster")
@pytest.mark.parametrize("value,keeps", [("", False), ("0", False), ("no", False), ("false", False),
                                         ("1", True), ("true", True), ("YES", True)])
def test_keep_job_keeps_only_for_a_yes(value, keeps):
    fn = _shell_function("run_live_eval.sh", "keep_job")
    out = subprocess.run(["bash", "-c", fn + "\nkeep_job"], env={**os.environ, "KEEP_JOB": value})
    assert (out.returncode == 0) == keeps
    assert '[[ -n "${KEEP_JOB:-}" ]]' not in (ROOT / "cluster" / "run_live_eval.sh").read_text()


# --- 4. frontend packages after package.json changed ----------------------------------------------------------------
def _stale(tmp_path) -> bool:
    fn = _shell_function("demo_start.sh", "frontend_packages_stale")
    return subprocess.run(["bash", "-c", fn + "\nfrontend_packages_stale"], cwd=tmp_path).returncode == 0


@pytest.mark.skipif(sys.platform == "win32", reason="Bash shell scripts run on POSIX / Linux cluster")
def test_packages_are_installed_when_missing_or_older_than_package_json(tmp_path):
    front = tmp_path / "frontend"
    front.mkdir()
    (front / "package.json").write_text("{}")
    assert _stale(tmp_path)                                    # no node_modules yet
    (front / "node_modules").mkdir()
    assert _stale(tmp_path)                                    # never installed by npm
    lock = front / "node_modules" / ".package-lock.json"
    lock.write_text("{}")
    past = time.time() - 60
    os.utime(front / "package.json", (past, past))
    assert not _stale(tmp_path)                                # installed after the last package.json change
    os.utime(front / "package.json", None)
    os.utime(lock, (past, past))
    assert _stale(tmp_path)                                    # package.json changed since: install again


def test_the_install_puts_the_lock_file_back():
    text = (ROOT / "cluster" / "demo_start.sh").read_text()
    assert "git checkout -- frontend/package-lock.json" in text
