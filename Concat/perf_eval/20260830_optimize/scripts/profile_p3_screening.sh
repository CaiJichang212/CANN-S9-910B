#!/usr/bin/env bash
set -euo pipefail
umask 022

source "$(dirname "$0")/env.sh"

run_dir=${1:?usage: profile_p3_screening.sh RUN_DIR ROUND}
round=${2:?usage: profile_p3_screening.sh RUN_DIR ROUND}
physical=${P3_PHYSICAL_DEVICE:?set P3_PHYSICAL_DEVICE}
logical=${P3_LOGICAL_DEVICE:-0}
round_name=$(printf 'round_%02d' "$round")
screening_label=${P3_SCREENING_LABEL:-screening}
raw_root="$run_dir/raw/$screening_label"

if (( round % 2 )); then
  labels=(p1 p3_boundary)
else
  labels=(p3_boundary p1)
fi

for label in "${labels[@]}"; do
  if [[ "$label" == p1 ]]; then
    private_version=p3_p1_baseline
  else
    private_version=p3_boundary
  fi
  destination="$raw_root/$round_name/$label/physical_$physical"
  if find "$destination" -type f -name 'op_summary*.csv' -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] P3 screening round=$round_name version=$label physical=$physical"
    continue
  fi
  "$PYTHON_BIN" "$TASK_HERE/scripts/device_gate.py" \
    --output "$run_dir/metadata/device_gate_${screening_label}_${round_name}_${label}_p${physical}.txt" \
    --physical "$physical"
  use_version "$private_version"
  mkdir -p "$destination"
  temporary=$(mktemp -d /tmp/concat_p3_screen.XXXXXX)
  echo "[profile] P3 screening round=$round_name version=$label physical=$physical logical=$logical cases=10"
  NPU_DEVICE="$logical" msprof --output="$temporary" --aic-metrics=PipeUtilization \
    --application="$PYTHON_BIN $TASK_HERE/p3_screen_runner.py --device $logical"
  mv "$temporary"/* "$destination"/
  rmdir "$temporary"
  chmod -R a+rX "$destination"
done

"$PYTHON_BIN" "$TASK_HERE/scripts/device_gate.py" \
  --output "$run_dir/metadata/device_gate_${screening_label}_${round_name}_post_p${physical}.txt" \
  --physical "$physical"
echo "P3_SCREEN_PROFILE_COMPLETE round=$round_name physical=$physical logical=$logical"
