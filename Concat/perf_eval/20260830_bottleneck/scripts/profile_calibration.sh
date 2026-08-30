#!/usr/bin/env bash
set -euo pipefail
umask 022

source "$(dirname "$0")/env.sh"
if [[ $(id -u) -ne 0 ]]; then
  echo "profiling requires container root because CANN OPP is root-readable" >&2
  exit 1
fi
stage=${1:?usage: profile_calibration.sh --attempt1|--retries}
raw_root="$TASK_HERE/raw/calibration"
mkdir -p "$raw_root" "$TASK_HERE/metadata"

profile_one() {
  local attempt=$1 physical=$2 logical=$3 destination temporary
  destination="$raw_root/attempt_${attempt}/physical_${physical}"
  if find "$destination" -type f -name 'op_summary*.csv' -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] calibration attempt=$attempt physical=$physical"
    return
  fi
  "$PYTHON_BIN" "$TASK_HERE/scripts/device_gate.py" \
    --output "$TASK_HERE/metadata/device_gate_calibration_a${attempt}_p${physical}.txt"
  mkdir -p "$destination"
  temporary=$(mktemp -d /tmp/concat_20260830_calibration.XXXXXX)
  echo "[profile] calibration attempt=$attempt physical=$physical logical=$logical"
  NPU_DEVICE="$logical" msprof --output="$temporary" --aic-metrics=PipeUtilization \
    --application="$PYTHON_BIN $TASK_HERE/run_suite.py --suite calibration --device $logical"
  mv "$temporary"/* "$destination"/
  rmdir "$temporary"
  chmod -R a+rX "$destination"
}

case "$stage" in
  --attempt1)
    for pair in "5 0" "6 1" "7 2"; do
      read -r physical logical <<< "$pair"
      profile_one 1 "$physical" "$logical"
    done
    ;;
  --retries)
    while read -r physical logical; do
      [[ -z "${physical:-}" ]] && continue
      profile_one 2 "$physical" "$logical"
    done < "$TASK_HERE/metadata/calibration_retry_devices.txt"
    ;;
  *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac
