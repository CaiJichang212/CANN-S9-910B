#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"
export BUILD_IMAGE="${BUILD_IMAGE:-swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-openeuler24.03-py3.11}"
: "${RELEASE_CANDIDATE_ID:?set RELEASE_CANDIDATE_ID to the accepted candidate ID}"
export RELEASE_CANDIDATE_ID
cd "$TASK_ROOT"
bash build_and_pack.sh
