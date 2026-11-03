#!/usr/bin/env bash
# Experiment 1 on the whole 90-question benchmark, on the cluster model, in one GPU job. Member 5.
#
#   bash cluster/run_architecture_eval.sh            # results tagged arch90 and arch90-noauditor
#   bash cluster/run_architecture_eval.sh mytag      # another pass under its own tags
#
# Variants A, B and C on evaluation/benchmark_hdb_90.json with the runtime auditor on, then Variant C again with the
# auditor off (HDB_AUDITOR=off), so the auditor's effect is measured on its own; then the report is written to
# evaluation/results/architecture_eval_<tag>.md. Both passes go through cluster/run_live_eval.sh, so it needs what that
# script needs: the SMU VPN (GlobalProtect), CLUSTER_LOGIN in the gitignored .env and the shared ssh session
# (ssh -M -S ~/.ssh/hdb-cluster.sock ... -fN "$CLUSTER_LOGIN", see the README). The first pass keeps the GPU job for the
# second, which cancels it. A failed pass stops here; run the same command again to resume (finished rows are skipped).
set -uo pipefail
TAG="${1:-arch90}"
cd "$(dirname "$0")/.."
DATASET="evaluation/benchmark_hdb_90.json"

echo "=================== $TAG: A,B,C with the auditor on ==================="
EVAL_DATASET="$DATASET" KEEP_JOB=1 bash cluster/run_live_eval.sh "$TAG" A,B,C
rc=$?
if [[ $rc -ne 0 ]]; then
  echo "!! $TAG failed (rc=$rc). A kept GPU job is in evaluation/results/cluster_job.txt (cancel it with: scancel <id>),"
  echo "   or run this command again to resume."
  exit $rc
fi

echo "=================== ${TAG}-noauditor: C with the auditor off ==================="
EVAL_DATASET="$DATASET" HDB_AUDITOR=off bash cluster/run_live_eval.sh "${TAG}-noauditor" C
rc=$?
if [[ $rc -ne 0 ]]; then
  echo "!! ${TAG}-noauditor failed (rc=$rc); run this command again to resume."
  exit $rc
fi

.venv/bin/python -m evaluation.run_architecture_eval --report --tag "$TAG"
