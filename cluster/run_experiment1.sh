#!/usr/bin/env bash
# Experiment 1, one full pass on the cluster model, in one GPU job. Member 4.
#
#   bash cluster/run_experiment1.sh            # results tagged exp1_*
#   bash cluster/run_experiment1.sh exp1b      # another pass under its own tags
#
# Variants A, B and C on every question set (Category E original and extra, B, A, C and F), C on the 60 varied
# Category E wordings, and two ablations of C on the original Category E questions: the model writes the facts too
# (HDB_CODE_WRITTEN_FACTS=0), and Member 5's auditor without its model judge (HDB_AUDITOR_LLM=off). The runner picks
# the add-on tools per set (all variants for Category A, none for the rest) and refuses any hosted model profile.
# About 2.5 hours. Needs the SMU VPN and your login (ssh $CLUSTER_LOGIN) like cluster/run_live_eval.sh, which each run
# goes through. Every run but the last keeps the GPU job (KEEP_JOB=1); the last cancels it. A failed run stops the
# pass and prints how to cancel the kept job; rerunning the same command resumes, since each run skips rows it has.
set -uo pipefail
PREFIX="${1:-exp1}"
cd "$(dirname "$0")/.."

run() {   # run DATASET TAG VARIANTS KEEP [ENV=VALUE ...]
  local dataset="$1" tag="$2" variants="$3" keep="$4"; shift 4
  echo; echo "=================== $tag (${dataset:-category E}, $variants) ==================="
  local env=()
  [[ "$keep" == keep ]] && env+=(KEEP_JOB=1)
  [[ -n "$dataset" ]] && env+=(EVAL_DATASET="$dataset")
  env ${env[@]+"${env[@]}"} "$@" bash cluster/run_live_eval.sh "$tag" "$variants"
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "!! $tag failed (rc=$rc); the pass stops here. A kept GPU job is in evaluation/results/cluster_job.txt"
    echo "   (cancel it with: scancel <id>), or rerun this command to resume."
    exit $rc
  fi
}

run ""         "${PREFIX}"          A,B,C keep
run extra      "${PREFIX}_extra"    A,B,C keep
run category_b "${PREFIX}_catb"     A,B,C keep
run category_a "${PREFIX}_cata"     A,B,C keep
run category_c "${PREFIX}_catc"     A,B,C keep
run category_f "${PREFIX}_catf"     A,B,C keep
run varied     "${PREFIX}_varied"   C     keep
run ""         "${PREFIX}_nofacts"  C     keep HDB_CODE_WRITTEN_FACTS=0
run ""         "${PREFIX}_nojudge"  C     last HDB_AUDITOR_LLM=off
echo; echo "== Experiment 1 pass ${PREFIX} done; the GPU job was cancelled by the last run"
