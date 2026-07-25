#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
SHARD_COUNT="${SHARDS:-5}"
MPL_CACHE="${MPLCONFIGDIR:-/private/tmp/vqml-symmetry-mpl}"

mkdir -p "${MPL_CACHE}"
cd "${PROJECT_DIR}"

# This path intentionally does not call paper/make_figures.py or make. The
# accepted manuscript and figure PDFs are immutable inputs to this rerun.
shard_pids=()
for ((shard_index = 0; shard_index < SHARD_COUNT; shard_index++)); do
  MPLCONFIGDIR="${MPL_CACHE}" "${PYTHON_BIN}" \
    -m experiments.leak_free_paper_experiments \
    --stage run \
    --shard-index "${shard_index}" \
    --shard-count "${SHARD_COUNT}" &
  shard_pids+=("$!")
done

shard_failure=0
for shard_pid in "${shard_pids[@]}"; do
  if ! wait "${shard_pid}"; then
    shard_failure=1
  fi
done
if ((shard_failure != 0)); then
  echo "At least one experiment shard failed; refusing to finalize." >&2
  exit 1
fi

MPLCONFIGDIR="${MPL_CACHE}" "${PYTHON_BIN}" \
  -m experiments.leak_free_paper_experiments \
  --stage finalize \
  --shard-count "${SHARD_COUNT}"
