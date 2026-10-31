"""
Demo and cluster scripts (tests/test_demo_scripts.py). Owned by Member 4.
Runs the scripts in a scratch copy of the repo with stand-ins for ssh, lsof, npm and caffeinate on the PATH, so no
test ever reaches the cluster, starts a server or keeps the laptop awake. They prove what demo day relies on: a
script never falls back to a password prompt, --dry-run stops before any ssh, and the early warnings fire.
Also checks the llm config names no paid model and the guardrail/auditor contract in docs/INTERFACES.md §3.

    LLM_PROFILE=none .venv/bin/pytest tests/test_demo_scripts.py -q
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

os.environ["LLM_PROFILE"] = "none"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents import supervisor as sv  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="the cluster scripts need bash")

SCRIPTS = ["demo_start.sh", "run_live_eval.sh", "tunnel.sh"]
LOGIN = "student@cluster.invalid"


def _stub(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(0o755)


def sandbox(tmp: Path, vite_engines: str | None = None, rank_bm25: bool = True, fake_node: bool = False) -> dict:
    """A copy of the scripts and frontend/package.json, a stand-in .venv python, and stand-in commands that log
    their arguments. Returns the env to run a script with (HOME is the scratch dir, so no real ssh socket is seen)."""
    (tmp / "cluster").mkdir()
    for name in SCRIPTS:
        shutil.copy(PROJECT_ROOT / "cluster" / name, tmp / "cluster" / name)
    (tmp / "frontend").mkdir()
    shutil.copy(PROJECT_ROOT / "frontend" / "package.json", tmp / "frontend" / "package.json")
    if vite_engines is not None:
        vite = tmp / "frontend" / "node_modules" / "vite"
        vite.mkdir(parents=True)
        (vite / "package.json").write_text(json.dumps({"name": "vite", "engines": {"node": vite_engines}}))
    (tmp / ".venv" / "bin").mkdir(parents=True)
    _stub(tmp / ".venv" / "bin" / "python", "exit 0" if rank_bm25 else "exit 1")
    (tmp / ".env").write_text(f"CLUSTER_LOGIN={LOGIN}\n")
    bin_dir = tmp / "fakebin"
    bin_dir.mkdir()
    for cmd, code in [("ssh", 255), ("npm", 1), ("caffeinate", 0), ("lsof", 1), ("curl", 1)]:
        _stub(bin_dir / cmd, f'echo "$@" >> "{tmp}/{cmd}.calls"\nexit {code}')
    if fake_node:
        _stub(bin_dir / "node", "exit 0")
    env = {k: v for k, v in os.environ.items()
           if k not in ("CLUSTER_LOGIN", "CLUSTER_SOCK", "LLM_TIMEOUT_S", "LLM_MAX_RETRIES", "CLUSTER_BATCH",
                        "CLUSTER_JOB_FILE", "KEEP_JOB")}
    env.update(PATH=f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}", HOME=str(tmp))
    return env


def run(tmp: Path, env: dict, script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(tmp / "cluster" / script), *args], cwd=tmp, env=env,
                          capture_output=True, text=True, timeout=60)


def calls(tmp: Path, cmd: str) -> list[str]:
    log = tmp / f"{cmd}.calls"
    return log.read_text().splitlines() if log.exists() else []


# --- the scripts as written -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", SCRIPTS)
def test_script_parses(name):
    assert subprocess.run(["bash", "-n", str(PROJECT_ROOT / "cluster" / name)]).returncode == 0


@pytest.mark.parametrize("name", SCRIPTS)
def test_every_shared_session_ssh_call_is_batch_mode(name):
    # A call through the socket must never ask for a password: with ControlMaster auto in ~/.ssh/config, a closed
    # socket would otherwise make ssh open a new session and prompt.
    for line in (PROJECT_ROOT / "cluster" / name).read_text().splitlines():
        code = line.split("#")[0] if not line.lstrip().startswith(("echo", "#")) else ""
        if re.search(r"\bssh\b", code) and '-S "$SOCK"' in code:
            assert "BatchMode=yes" in code, line


@pytest.mark.parametrize("name", SCRIPTS)
def test_headers_say_log_in_with_ssh_not_the_long_master_command(name):
    text = (PROJECT_ROOT / "cluster" / name).read_text()
    assert "ssh -M" not in text and "log in with  ssh $" in text.replace("Log in", "log in")


# --- demo_start.sh --------------------------------------------------------------------------------------------------
def test_dry_run_prints_the_plan_and_never_runs_ssh(tmp_path):
    env = sandbox(tmp_path, vite_engines=">=1.0.0")
    out = run(tmp_path, env, "demo_start.sh", "--dry-run", "--preview")
    assert out.returncode == 0, out.stdout + out.stderr
    assert calls(tmp_path, "ssh") == [] and calls(tmp_path, "npm") == [] and calls(tmp_path, "caffeinate") == []
    assert "dry run: nothing started, no ssh" in out.stdout
    assert f"ssh -o BatchMode=yes -S {tmp_path}/.ssh/hdb-cluster.sock -O check {LOGIN}" in out.stdout
    assert "LLM_TIMEOUT_S=25 LLM_MAX_RETRIES=0" in out.stdout                      # the demo default
    assert "vite build" in out.stdout and "vite preview" in out.stdout and "npm run dev" not in out.stdout
    assert "keep the laptop awake" in out.stdout and "cancel the GPU job" in out.stdout
    assert not (tmp_path / "frontend" / "dist").exists()


def test_dry_run_keeps_the_timeout_override_and_keep_job(tmp_path):
    env = sandbox(tmp_path, vite_engines=">=1.0.0")
    env["LLM_TIMEOUT_S"] = "40"
    out = run(tmp_path, env, "demo_start.sh", "--dry-run", "--keep-job")
    assert "LLM_TIMEOUT_S=40" in out.stdout and "leave the GPU job running (--keep-job)" in out.stdout
    assert "npm run dev" in out.stdout and "vite preview" not in out.stdout


def test_dry_run_offline_needs_no_cluster_login(tmp_path):
    env = sandbox(tmp_path, vite_engines=">=1.0.0")
    (tmp_path / ".env").write_text("")
    out = run(tmp_path, env, "demo_start.sh", "--dry-run", "--offline")
    assert out.returncode == 0 and "LLM_PROFILE=none" in out.stdout and "CLUSTER_LOGIN" not in out.stdout


def test_unknown_option_is_refused(tmp_path):
    env = sandbox(tmp_path)
    out = run(tmp_path, env, "demo_start.sh", "--dryrun")
    assert out.returncode == 2 and "unknown option --dryrun" in out.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")
def test_warns_when_node_is_older_than_vite_needs_and_when_rank_bm25_is_missing(tmp_path):
    env = sandbox(tmp_path, vite_engines=">=99.0.0", rank_bm25=False)
    out = run(tmp_path, env, "demo_start.sh", "--dry-run")
    assert "needs Node >=99.0.0" in out.stdout and "install Node 22 LTS" in out.stdout
    assert "rank_bm25 is not installed" in out.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")
def test_no_warning_when_node_and_rank_bm25_are_fine(tmp_path):
    env = sandbox(tmp_path, vite_engines="^1.0.0 || >=2.0.0")
    out = run(tmp_path, env, "demo_start.sh", "--dry-run")
    assert "needs Node" not in out.stdout and "rank_bm25" not in out.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")
def test_node_range_comes_from_package_json_before_npm_install(tmp_path):
    # No node_modules yet: the Vite major in frontend/package.json (Vite 7 and later) gives ^20.19.0 || >=22.12.0
    env = sandbox(tmp_path)
    major = int(json.loads((PROJECT_ROOT / "frontend" / "package.json").read_text())["devDependencies"]["vite"]
                .lstrip("^~>=").split(".")[0])
    have = tuple(int(x) for x in subprocess.run(["node", "--version"], capture_output=True, text=True)
                 .stdout.strip().lstrip("v").split("."))
    fits = (have[0] == 20 and have >= (20, 19, 0)) or have >= (22, 12, 0)
    out = run(tmp_path, env, "demo_start.sh", "--dry-run")
    assert major >= 7
    assert ("needs Node ^20.19.0 || >=22.12.0" in out.stdout) is (not fits), out.stdout


def test_no_shared_session_stops_with_the_login_hint_and_no_password_prompt(tmp_path):
    env = sandbox(tmp_path, fake_node=True)
    (tmp_path / "frontend" / "node_modules").mkdir()
    out = run(tmp_path, env, "demo_start.sh")
    assert out.returncode == 1
    assert f"log in with  ssh {LOGIN}  in your terminal" in out.stdout
    ssh = calls(tmp_path, "ssh")
    assert ssh and all("BatchMode=yes" in c for c in ssh) and "-O check" in ssh[0]
    assert calls(tmp_path, "caffeinate") and "-w" in calls(tmp_path, "caffeinate")[0]   # awake for the run


# --- run_live_eval.sh and tunnel.sh ---------------------------------------------------------------------------------
def test_live_eval_without_a_session_stops_with_the_login_hint(tmp_path):
    env = sandbox(tmp_path)
    out = run(tmp_path, env, "run_live_eval.sh", "tag1", "C")
    assert out.returncode == 1 and f"log in with  ssh {LOGIN}  in your terminal" in out.stdout
    assert all("BatchMode=yes" in c for c in calls(tmp_path, "ssh"))
    assert "scancel" not in out.stdout                                   # no job was left by an earlier run


JOB_FILE = Path("evaluation") / "results" / "cluster_job.txt"   # gitignored, like every results file


def live_sandbox(tmp: Path, squeue: str = "", scancel: int = 0, run_code: int = 0) -> dict:
    """sandbox() with a live shared session: a stand-in ssh that answers the cluster commands run_live_eval.sh sends
    (squeue lists `squeue`, sbatch submits job 555, scancel exits with `scancel`), a stand-in tunnel.sh, a server
    that is ready at once, a smoke test that passes and a runner that exits with `run_code`. Nothing leaves the laptop.
    The three answers come from the returned env (STUB_*), so a second run in the same folder can change them."""
    env = sandbox(tmp)
    env.update(STUB_SQUEUE=squeue, STUB_SCANCEL=str(scancel), STUB_RUN_CODE=str(run_code))
    _stub(tmp / "fakebin" / "ssh", f"""echo "$@" >> "{tmp}/ssh.calls"
case "$*" in
  *"-O check"*) exit 0 ;;
  *squeue*) [ -n "$STUB_SQUEUE" ] && echo "$STUB_SQUEUE"; exit 0 ;;
  *sbatch*) echo 555; exit 0 ;;
  *scancel*) exit "$STUB_SCANCEL" ;;
  *vllm_endpoint.txt*) echo "node1 8100 555"; exit 0 ;;
esac
exit 0""")
    _stub(tmp / "fakebin" / "curl", "exit 0")
    _stub(tmp / "cluster" / "tunnel.sh", f'echo "$@" >> "{tmp}/tunnel.calls"\nexit 0')
    _stub(tmp / ".venv" / "bin" / "python", f"""echo "$* LLM_TIMEOUT_S=$LLM_TIMEOUT_S LLM_MAX_RETRIES=$LLM_MAX_RETRIES" >> "{tmp}/python.calls"
case "$*" in
  *smoke_test*) echo "OK: well-formed"; exit 0 ;;
  *run_category_e_prelim*) [ "$STUB_RUN_CODE" = 3 ] && echo "MODEL_UNREACHABLE: stopped after 2/60"; exit "$STUB_RUN_CODE" ;;
esac
exit 0""")
    return env


def recorded_job(tmp: Path) -> str | None:
    f = tmp / JOB_FILE
    return f.read_text().strip() if f.exists() else None


def test_live_eval_without_a_session_prints_the_scancel_for_a_job_left_running(tmp_path):
    # 31 Oct: the VPN dropped while KEEP_JOB had left job 344978 running; the later steps could not reach the cluster
    env = sandbox(tmp_path)
    (tmp_path / JOB_FILE).parent.mkdir(parents=True)
    (tmp_path / JOB_FILE).write_text("344978\n")
    out = run(tmp_path, env, "run_live_eval.sh", "tag1", "C")
    assert out.returncode == 1 and f"log in with  ssh {LOGIN}  in your terminal" in out.stdout
    assert "after logging in, run  scancel 344978" in out.stdout
    assert recorded_job(tmp_path) == "344978"                            # still known to the next run
    assert [c for c in calls(tmp_path, "ssh") if "-O check" not in c] == []


def test_keep_job_records_the_job_and_the_next_run_reuses_then_cancels_it(tmp_path):
    env = live_sandbox(tmp_path)
    out = run(tmp_path, {**env, "KEEP_JOB": "1"}, "run_live_eval.sh", "t1", "C")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "submitted job 555" in out.stdout and "left job 555 running (KEEP_JOB)" in out.stdout
    assert "scancel 555" in out.stdout and recorded_job(tmp_path) == "555"
    assert not any("scancel" in c for c in calls(tmp_path, "ssh"))
    (tmp_path / "ssh.calls").unlink()
    env["STUB_SQUEUE"] = "555"                                           # the job is still queued
    out = run(tmp_path, env, "run_live_eval.sh", "t2", "C")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "using job 555 left running by an earlier run" in out.stdout and "cancelled job 555" in out.stdout
    ssh = calls(tmp_path, "ssh")
    assert not any("sbatch" in c for c in ssh) and any(c.endswith("scancel 555") for c in ssh)
    assert all("BatchMode=yes" in c for c in ssh) and recorded_job(tmp_path) is None


def test_a_recorded_job_that_has_ended_is_forgotten(tmp_path):
    env = live_sandbox(tmp_path, squeue="901")
    (tmp_path / JOB_FILE).parent.mkdir(parents=True)
    (tmp_path / JOB_FILE).write_text("777\n")
    out = run(tmp_path, env, "run_live_eval.sh", "t1", "C")
    assert "job 777 from an earlier run has ended" in out.stdout and "using running job 901" in out.stdout
    assert "cancelled job 901" in out.stdout and recorded_job(tmp_path) is None


def test_a_cancel_that_cannot_reach_the_cluster_keeps_the_record_and_prints_the_scancel(tmp_path):
    env = live_sandbox(tmp_path, scancel=255)
    out = run(tmp_path, env, "run_live_eval.sh", "t1", "C")
    assert f"Log in with  ssh {LOGIN}  and run: scancel 555" in out.stdout
    assert recorded_job(tmp_path) == "555"


def test_a_lost_model_server_stops_the_script_before_the_next_tag(tmp_path):
    env = live_sandbox(tmp_path, run_code=3)
    out = run(tmp_path, env, "run_live_eval.sh", "t1,t2", "C")
    assert out.returncode == 1 and "STEP_FAILED t1: the model server stopped answering" in out.stdout
    assert "MODEL_UNREACHABLE" in out.stdout and "RUN_DONE" not in out.stdout
    runs = [c for c in calls(tmp_path, "python") if "run_category_e_prelim" in c]
    assert len(runs) == 1 and "--tag t1" in runs[0]                      # t2 never started
    assert "cancelled job 555" in out.stdout                             # the job is still cancelled on the way out


def test_live_eval_model_calls_use_evaluation_timeouts_unless_overridden(tmp_path):
    env = live_sandbox(tmp_path)
    run(tmp_path, env, "run_live_eval.sh", "t1", "C")
    runs = [c for c in calls(tmp_path, "python") if "run_category_e_prelim" in c]
    assert runs and "LLM_TIMEOUT_S=90 LLM_MAX_RETRIES=1" in runs[0]
    (tmp_path / "python.calls").unlink()
    run(tmp_path, {**env, "LLM_TIMEOUT_S": "200", "LLM_MAX_RETRIES": "0"}, "run_live_eval.sh", "t1", "C")
    runs = [c for c in calls(tmp_path, "python") if "run_category_e_prelim" in c]
    assert "LLM_TIMEOUT_S=200 LLM_MAX_RETRIES=0" in runs[0]


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck is not installed")
def test_live_eval_passes_shellcheck():
    out = subprocess.run(["shellcheck", "-S", "warning", str(PROJECT_ROOT / "cluster" / "run_live_eval.sh")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout


def test_tunnel_called_from_a_script_never_falls_back_to_a_password_prompt(tmp_path):
    env = sandbox(tmp_path)
    env["CLUSTER_BATCH"] = "1"
    out = run(tmp_path, env, "tunnel.sh")
    assert out.returncode == 1 and f"log in with  ssh {LOGIN}" in out.stdout and calls(tmp_path, "ssh") == []
    closed = run(tmp_path, env, "tunnel.sh", "--close")                 # demo cleanup after the session dropped
    assert closed.returncode == 0 and "no forward to close" in closed.stdout and calls(tmp_path, "ssh") == []


# --- config and the guardrail/auditor contract ----------------------------------------------------------------------
def test_config_names_no_paid_model():
    text = (PROJECT_ROOT / "config" / "experiment_config.yaml").read_text()
    assert not re.search(r"gpt-|text-embedding-3|openai\.com|claude-|gemini", text)
    cfg = yaml.safe_load(text)
    assert cfg["global"]["temperature"] == 0.0 and cfg["global"]["seed"] == 42      # still read by agents/llm.py
    for profile in cfg["llm"]["profiles"].values():
        assert profile["base_url"].startswith("http://localhost:")


def _interfaces_section_3() -> str:
    doc = (PROJECT_ROOT / "docs" / "INTERFACES.md").read_text()
    return doc.split("## 3.", 1)[1].split("\n## 4.", 1)[0]


def test_guardrail_contract_names_the_hooks_the_supervisor_loads():
    section, source = _interfaces_section_3(), (PROJECT_ROOT / "agents" / "supervisor.py").read_text()
    for name in ("input_guardrail_node", "runtime_auditor_node"):
        assert f"`{name}(state" in section and f'"{name}"' in source
    assert "agents/guardrails.py" in section and 'module: str = "agents.guardrails"' in source   # the module the loader imports


def test_auditor_recipe_in_the_contract_separates_the_model_opening_from_the_code_facts():
    # The recipe in docs/INTERFACES.md §3: the code-written facts are template_response(state), the same text the
    # synthesis appended, so the auditor judges only what comes before them.
    from unittest.mock import patch
    from langchain_core.messages import AIMessage
    from agents import arch_3_specialist_5agent as C
    item = next(i for i in json.loads((PROJECT_ROOT / "evaluation" / "category_e_queries.json").read_text())
                if i["id"] == "E1-03")
    state = C.graph.invoke(sv.initial_state(item["query"], "C"))

    class Writer:
        def invoke(self, _msgs):
            return AIMessage(content="Both Ang Mo Kio blocks fit your search.")

    with patch.object(sv, "get_llm", return_value=Writer()):
        after = {**state, **sv.synthesize_node({**state, "draft_response": ""})}
    assert "sv.template_response(state)" in _interfaces_section_3()
    facts = sv.template_response(after) if sv.writes_facts_in_code(after) else ""
    draft = after["draft_response"]
    opening = draft.split(facts, 1)[0].strip() if facts and facts in draft else draft
    assert opening == "Both Ang Mo Kio blocks fit your search."
    assert sv.template_response(state) == facts                     # the same text on every retry of the loop
