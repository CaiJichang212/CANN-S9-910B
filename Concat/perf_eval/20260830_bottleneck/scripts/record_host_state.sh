#!/usr/bin/env bash
set -euo pipefail
umask 077

phase=${1:?usage: record_host_state.sh pre|post}
case "$phase" in
  pre|post) ;;
  *) echo "phase must be pre or post" >&2; exit 2 ;;
esac

here=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$here/metadata"
output="$here/metadata/host_${phase}_state.txt"

{
  date --iso-8601=ns
  echo "container=concat_bottleneck_20260830"
  sudo -n docker inspect -f 'Id={{.Id}} Image={{.Image}} Pid={{.State.Pid}} Status={{.State.Status}} Started={{.State.StartedAt}} Devices={{json .HostConfig.Devices}} Mounts={{json .Mounts}}' concat_bottleneck_20260830
  sudo -n docker top concat_bottleneck_20260830 -eo pid,ppid,user,comm,args
  sudo -n docker exec concat_bottleneck_20260830 readlink /proc/1/ns/mnt
  sudo -n docker exec concat_bottleneck_20260830 readlink /proc/1/ns/pid
  sudo -n docker exec concat_bottleneck_20260830 readlink /proc/1/ns/net
  echo "excluded_container=jinyr_vllm_new"
  sudo -n docker inspect -f 'Id={{.Id}} Pid={{.State.Pid}} Status={{.State.Status}} Devices={{json .HostConfig.Devices}}' jinyr_vllm_new
  npu-smi info
} > "$output" 2>&1

echo "wrote $output"
