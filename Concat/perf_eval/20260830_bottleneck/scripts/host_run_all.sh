#!/usr/bin/env bash
set -euo pipefail
umask 077

here=$(cd "$(dirname "$0")/.." && pwd)
container=concat_bottleneck_20260830
recorded_post=0

if [[ -n "$(find "$here/raw" -type f -print -quit 2>/dev/null)" ]]; then
  echo "refusing to rebuild and reuse existing raw profiles without an immutable run fingerprint" >&2
  echo "use summarize_all.sh for the archived run, or copy the workflow to a new dated directory for recollection" >&2
  exit 2
fi

cleanup() {
  if sudo -n docker inspect "$container" >/dev/null 2>&1; then
    if [[ $recorded_post -eq 0 ]] && [[ $(sudo -n docker inspect -f '{{.State.Running}}' "$container") == true ]]; then
      bash "$here/scripts/record_host_state.sh" post || true
    fi
    if [[ $(sudo -n docker inspect -f '{{.State.Running}}' "$container") == true ]]; then
      sudo -n docker stop "$container" >/dev/null
    fi
  fi
}
trap cleanup EXIT

if [[ $(sudo -n docker inspect -f '{{.State.Running}}' "$container") != true ]]; then
  sudo -n docker start "$container" >/dev/null
fi
if ! sudo -n docker exec "$container" getent passwd 9002 >/dev/null; then
  sudo -n docker exec "$container" useradd --uid 9002 --gid 100 \
    --home-dir /tmp/concat_20260830_uid9002 --no-create-home --shell /bin/bash concatperf
fi
bash "$here/scripts/record_host_state.sh" pre

sudo -n docker exec "$container" bash "$here/scripts/build_opp_root.sh"
sudo -n docker exec --user 9002:100 "$container" bash "$here/scripts/prepare_private_env.sh"
sudo -n docker exec "$container" bash "$here/scripts/verify_runtime_root.sh"
sudo -n docker exec --user 9002:100 "$container" bash "$here/scripts/collect_runtime_evidence.sh"
sudo -n docker exec "$container" bash "$here/scripts/run_correctness.sh"
sudo -n docker exec --user 9002:100 "$container" bash "$here/scripts/collect_correctness_evidence.sh"
sudo -n docker exec "$container" bash "$here/scripts/profile_calibration.sh" --attempt1
sudo -n docker exec --user 9002:100 "$container" bash "$here/scripts/run_python.sh" \
  "$here/scripts/summarize_calibration.py" --initial
sudo -n docker exec "$container" bash "$here/scripts/profile_calibration.sh" --retries
sudo -n docker exec --user 9002:100 "$container" bash "$here/scripts/run_python.sh" \
  "$here/scripts/summarize_calibration.py" --final
sudo -n docker exec "$container" bash "$here/scripts/profile_latency.sh"
sudo -n docker exec --user 9002:100 "$container" bash "$here/scripts/run_python.sh" \
  "$here/scripts/summarize_latency.py"
sudo -n docker exec "$container" bash "$here/scripts/profile_deep.sh"
sudo -n docker exec --user 9002:100 "$container" bash "$here/scripts/run_python.sh" \
  "$here/scripts/summarize_deep.py"
sudo -n docker exec --user 9002:100 "$container" bash "$here/scripts/summarize_all.sh"

bash "$here/scripts/record_host_state.sh" post
recorded_post=1
echo "workflow complete; the task container will now be stopped"
