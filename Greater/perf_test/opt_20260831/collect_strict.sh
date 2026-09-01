#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  collect_strict.sh --run <custom_opp.run> --out <new-output-dir> \
    --device <logical-device-id> <spec> [<spec> ...]

The output directory must not already exist. Specs are prof_matrix.py keys.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

sha_file() {
    sha256sum -- "$1" | awk '{print $1}'
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERF_DIR="$(dirname "$SCRIPT_DIR")"
GREATER_DIR="$(dirname "$PERF_DIR")"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PROF_MATRIX="$PERF_DIR/prof_matrix.py"
PARSER="$SCRIPT_DIR/parse_strict.py"

RUN_ARG=""
OUT_ARG=""
DEVICE=""
SPECS=()

while (($#)); do
    case "$1" in
        --run)
            (($# >= 2)) || die "--run requires a value"
            RUN_ARG="$2"
            shift 2
            ;;
        --out)
            (($# >= 2)) || die "--out requires a value"
            OUT_ARG="$2"
            shift 2
            ;;
        --device)
            (($# >= 2)) || die "--device requires a value"
            DEVICE="$2"
            shift 2
            ;;
        --)
            shift
            while (($#)); do
                SPECS+=("$1")
                shift
            done
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            die "unknown option: $1"
            ;;
        *)
            SPECS+=("$1")
            shift
            ;;
    esac
done

[[ -n "$RUN_ARG" ]] || die "--run is required"
[[ -n "$OUT_ARG" ]] || die "--out is required"
[[ "$DEVICE" =~ ^[0-9]+$ ]] || die "--device must be a non-negative integer"
((${#SPECS[@]} > 0)) || die "at least one spec is required"
[[ -f "$PROF_MATRIX" ]] || die "missing profiler driver: $PROF_MATRIX"
[[ -f "$PARSER" ]] || die "missing strict parser: $PARSER"

declare -A SEEN_SPECS=()
for spec in "${SPECS[@]}"; do
    [[ "$spec" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "unsafe spec name: $spec"
    [[ -z "${SEEN_SPECS[$spec]+x}" ]] || die "duplicate spec: $spec"
    SEEN_SPECS[$spec]=1
done

RUN_PATH="$(realpath -- "$RUN_ARG")"
[[ -f "$RUN_PATH" && -r "$RUN_PATH" ]] || die "run package is not a readable file: $RUN_PATH"
OUT_DIR="$(realpath -m -- "$OUT_ARG")"
if [[ -e "$OUT_DIR" || -L "$OUT_DIR" ]]; then
    die "output already exists: $OUT_DIR"
fi
OUT_REPO_REL=""
if [[ "$OUT_DIR" == "$REPO_ROOT/"* ]]; then
    OUT_REPO_REL="${OUT_DIR#"$REPO_ROOT/"}"
fi

for command_name in git sha256sum awk realpath msprof npu-smi readelf timeout; do
    command -v "$command_name" >/dev/null 2>&1 || die "required command not found: $command_name"
done
PYTHON_BIN="${PYTHON:-python3}"
PYTHON_PATH="$(command -v "$PYTHON_BIN")" || die "Python executable not found: $PYTHON_BIN"

[[ -n "${ASCEND_HOME_PATH:-}" ]] || die "ASCEND_HOME_PATH is not set; source CANN 8.5 set_env.sh first"
[[ -n "${ASCEND_OPP_PATH:-}" ]] || die "ASCEND_OPP_PATH is not set; source CANN 8.5 set_env.sh first"

CANN_VERSION_FILE=""
for candidate in \
    "$ASCEND_HOME_PATH/compiler/version.info" \
    "$ASCEND_HOME_PATH/share/info/asc-devkit/version.info"; do
    if [[ -f "$candidate" ]]; then
        CANN_VERSION_FILE="$(readlink -f -- "$candidate")"
        break
    fi
done
[[ -n "$CANN_VERSION_FILE" ]] || die "cannot find the CANN version file under ASCEND_HOME_PATH"
CANN_VERSION="$(sed -n 's/^Version=//p' "$CANN_VERSION_FILE" | head -n 1)"
[[ -n "$CANN_VERSION" ]] || die "cannot read CANN version from $CANN_VERSION_FILE"

SOURCE_HOST="$REPO_ROOT/Greater/op_project/custom_greater/op_host/greater.cpp"
SOURCE_TILING="$REPO_ROOT/Greater/op_project/custom_greater/op_host/greater_tiling.h"
SOURCE_KERNEL="$REPO_ROOT/Greater/op_project/custom_greater/op_kernel/greater.cpp"
SOURCE_CALLER="$GREATER_DIR/extension/custom_op.cpp"
SOURCE_HELPER="$GREATER_DIR/common/pytorch_npu_helper.hpp"
for source_file in "$SOURCE_HOST" "$SOURCE_TILING" "$SOURCE_KERNEL" \
                   "$SOURCE_CALLER" "$SOURCE_HELPER"; do
    [[ -f "$source_file" ]] || die "missing source file: $source_file"
done
mapfile -t PYBIND_CANDIDATES < <(
    find "$GREATER_DIR" -maxdepth 1 -type f \
        -name 'custom_ops_lib.cpython-39-*.so' -print
)
((${#PYBIND_CANDIDATES[@]} == 1)) \
    || die "expected one Python 3.9 custom_ops_lib, found ${#PYBIND_CANDIDATES[@]}"
PYBIND_PATH="${PYBIND_CANDIDATES[0]}"

compute_dirty_hash() {
    (
        cd "$REPO_ROOT"
        git diff --binary HEAD --
        while IFS= read -r -d '' untracked; do
            if [[ -n "$OUT_REPO_REL" &&
                  ( "$untracked" == "$OUT_REPO_REL" || "$untracked" == "$OUT_REPO_REL/"* ) ]]; then
                continue
            fi
            printf '\0UNTRACKED\0%s\0' "$untracked"
            sha256sum -- "$untracked"
        done < <(git ls-files --others --exclude-standard -z | LC_ALL=C sort -z)
    ) | sha256sum | awk '{print $1}'
}

STARTED_AT="$(date -Iseconds)"
GIT_HEAD="$(git -C "$REPO_ROOT" rev-parse HEAD)"
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=all)" ]]; then
    GIT_DIRTY=true
else
    GIT_DIRTY=false
fi
DIRTY_HASH="$(compute_dirty_hash)"
HOST_HASH="$(sha_file "$SOURCE_HOST")"
TILING_HASH="$(sha_file "$SOURCE_TILING")"
KERNEL_HASH="$(sha_file "$SOURCE_KERNEL")"
CALLER_HASH="$(sha_file "$SOURCE_CALLER")"
HELPER_HASH="$(sha_file "$SOURCE_HELPER")"
PYBIND_HASH="$(sha_file "$PYBIND_PATH")"
RUN_HASH="$(sha_file "$RUN_PATH")"
SPEC_ORDER_HASH="$(printf '%s\n' "${SPECS[@]}" | sha256sum | awk '{print $1}')"

ARTIFACT_SOURCE_MANIFEST="$(dirname "$RUN_PATH")/source_manifest.txt"
[[ -f "$ARTIFACT_SOURCE_MANIFEST" ]] \
    || die "missing artifact source manifest: $ARTIFACT_SOURCE_MANIFEST"
manifest_value() {
    local key="$1"
    awk -F= -v wanted="$key" '$1 == wanted {sub(/^[^=]*=/, ""); print; found = 1} END {if (!found) exit 1}' \
        "$ARTIFACT_SOURCE_MANIFEST"
}
ARTIFACT_ID="$(manifest_value artifact_id)" || die "artifact_id missing from source manifest"
ARTIFACT_HOST_HASH="$(manifest_value source_host_sha256)" \
    || die "source_host_sha256 missing from source manifest"
ARTIFACT_TILING_HASH="$(manifest_value source_tiling_sha256)" \
    || die "source_tiling_sha256 missing from source manifest"
ARTIFACT_KERNEL_HASH="$(manifest_value source_kernel_sha256)" \
    || die "source_kernel_sha256 missing from source manifest"
ARTIFACT_RUN_HASH="$(manifest_value run_sha256)" \
    || die "run_sha256 missing from source manifest"
[[ "$ARTIFACT_RUN_HASH" == "$RUN_HASH" ]] \
    || die "run hash does not match artifact source manifest"
ARTIFACT_SOURCE_MANIFEST_HASH="$(sha_file "$ARTIFACT_SOURCE_MANIFEST")"

mkdir -p -- "$(dirname "$OUT_DIR")"
mkdir -- "$OUT_DIR"
mkdir -- "$OUT_DIR/metadata" "$OUT_DIR/specs"
MANIFEST="$OUT_DIR/run_manifest.txt"
CURRENT_SPEC="none"
RUN_STATUS="failed"

finish() {
    local exit_code=$?
    trap - EXIT
    if [[ -n "${MANIFEST:-}" && -f "$MANIFEST" ]]; then
        {
            printf 'ended_at=%s\n' "$(date -Iseconds)"
            printf 'status=%s\n' "$RUN_STATUS"
            printf 'exit_code=%s\n' "$exit_code"
            printf 'last_spec=%s\n' "$CURRENT_SPEC"
        } >> "$MANIFEST"
    fi
    exit "$exit_code"
}
trap finish EXIT

printf '%s\n' "${SPECS[@]}" > "$OUT_DIR/spec_order.txt"
cp -- /proc/1/cgroup "$OUT_DIR/metadata/container_cgroup.txt"

{
    printf 'manifest_version=1\n'
    printf 'operator=Greater\n'
    printf 'started_at=%s\n' "$STARTED_AT"
    printf 'output_dir=%s\n' "$OUT_DIR"
    printf 'git_head=%s\n' "$GIT_HEAD"
    printf 'git_dirty=%s\n' "$GIT_DIRTY"
    printf 'dirty_diff_sha256=%s\n' "$DIRTY_HASH"
    printf 'source_host_sha256=%s\n' "$HOST_HASH"
    printf 'source_tiling_sha256=%s\n' "$TILING_HASH"
    printf 'source_kernel_sha256=%s\n' "$KERNEL_HASH"
    printf 'source_caller_sha256=%s\n' "$CALLER_HASH"
    printf 'source_helper_sha256=%s\n' "$HELPER_HASH"
    printf 'pybind_path=%s\n' "$PYBIND_PATH"
    printf 'pybind_sha256=%s\n' "$PYBIND_HASH"
    printf 'run_path=%s\n' "$RUN_PATH"
    printf 'run_sha256=%s\n' "$RUN_HASH"
    printf 'artifact_source_manifest=%s\n' "$ARTIFACT_SOURCE_MANIFEST"
    printf 'artifact_source_manifest_sha256=%s\n' "$ARTIFACT_SOURCE_MANIFEST_HASH"
    printf 'artifact_id=%s\n' "$ARTIFACT_ID"
    printf 'artifact_source_host_sha256=%s\n' "$ARTIFACT_HOST_HASH"
    printf 'artifact_source_tiling_sha256=%s\n' "$ARTIFACT_TILING_HASH"
    printf 'artifact_source_kernel_sha256=%s\n' "$ARTIFACT_KERNEL_HASH"
    printf 'container_hostname=%s\n' "${HOSTNAME:-unknown}"
    printf 'container_name=%s\n' "${CONTAINER_NAME:-unknown}"
    printf 'container_image=%s\n' "${CONTAINER_IMAGE_DIGEST:-unknown}"
    printf 'container_cgroup_sha256=%s\n' "$(sha_file "$OUT_DIR/metadata/container_cgroup.txt")"
    printf 'cann_home=%s\n' "$ASCEND_HOME_PATH"
    printf 'cann_opp_path=%s\n' "$ASCEND_OPP_PATH"
    printf 'cann_version=%s\n' "$CANN_VERSION"
    printf 'cann_version_file=%s\n' "$CANN_VERSION_FILE"
    printf 'python=%s\n' "$PYTHON_PATH"
    printf 'python_version=%s\n' "$($PYTHON_PATH --version 2>&1)"
    printf 'msprof=%s\n' "$(command -v msprof)"
    printf 'device_logical=%s\n' "$DEVICE"
    printf 'npu_snapshot_scope=all devices visible in the task container; expected single-card container\n'
    printf 'spec_count=%s\n' "${#SPECS[@]}"
    printf 'spec_order_sha256=%s\n' "$SPEC_ORDER_HASH"
    for index in "${!SPECS[@]}"; do
        printf 'spec_%03d=%s\n' "$index" "${SPECS[$index]}"
    done
    printf 'initial_status=in_progress\n'
} > "$MANIFEST"

npu-smi info -m > "$OUT_DIR/metadata/npu_map.txt"
npu-smi info > "$OUT_DIR/metadata/npu_status_before.txt"
{
    printf 'npu_map_sha256=%s\n' "$(sha_file "$OUT_DIR/metadata/npu_map.txt")"
    printf 'npu_status_before_sha256=%s\n' "$(sha_file "$OUT_DIR/metadata/npu_status_before.txt")"
} >> "$MANIFEST"

ORIGINAL_PYTHONPATH="${PYTHONPATH-}"
export PYTHONPATH="$GREATER_DIR${ORIGINAL_PYTHONPATH:+:$ORIGINAL_PYTHONPATH}"
ORIGINAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
CUSTOM_OPAPI_DIR="$ASCEND_OPP_PATH/vendors/customize/op_api/lib"
export LD_LIBRARY_PATH="$CUSTOM_OPAPI_DIR${ORIGINAL_LD_LIBRARY_PATH:+:$ORIGINAL_LD_LIBRARY_PATH}"
export GREATER_DEV="$DEVICE"

if bash "$RUN_PATH" > "$OUT_DIR/metadata/install.log" 2>&1; then
    :
else
    install_rc=$?
    die "run package installation failed with exit code $install_rc; see metadata/install.log"
fi

INSTALLED_OPAPI="$CUSTOM_OPAPI_DIR/libcust_opapi.so"
[[ -f "$INSTALLED_OPAPI" ]] || die "installed libcust_opapi.so not found at $INSTALLED_OPAPI"
readelf --wide --dyn-syms "$INSTALLED_OPAPI" > "$OUT_DIR/metadata/libcust_opapi.symbols.txt"
grep -Eq '[[:space:]]aclnnGreater$' "$OUT_DIR/metadata/libcust_opapi.symbols.txt" \
    || die "installed libcust_opapi.so does not export aclnnGreater"
grep -Eq '[[:space:]]aclnnGreaterGetWorkspaceSize$' "$OUT_DIR/metadata/libcust_opapi.symbols.txt" \
    || die "installed libcust_opapi.so does not export aclnnGreaterGetWorkspaceSize"
INSTALLED_OPAPI_HASH="$(sha_file "$INSTALLED_OPAPI")"
printf 'installed_opapi_sha256=%s\n' "$INSTALLED_OPAPI_HASH" >> "$MANIFEST"

INSTALLED_TILING="$ASCEND_OPP_PATH/vendors/customize/op_impl/ai_core/tbe/op_tiling/liboptiling.so"
INSTALLED_MASTER="$ASCEND_OPP_PATH/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so"
INSTALLED_KERNEL_DIR="$ASCEND_OPP_PATH/vendors/customize/op_impl/ai_core/tbe/kernel/ascend910b/greater"
[[ -f "$INSTALLED_TILING" ]] || die "installed liboptiling.so not found at $INSTALLED_TILING"
[[ -f "$INSTALLED_MASTER" ]] || die "installed opmaster library not found at $INSTALLED_MASTER"
[[ -d "$INSTALLED_KERNEL_DIR" ]] || die "installed Greater Kernel directory not found"
INSTALLED_TILING_HASH="$(sha_file "$INSTALLED_TILING")"
INSTALLED_MASTER_HASH="$(sha_file "$INSTALLED_MASTER")"
find "$INSTALLED_KERNEL_DIR" -maxdepth 1 -type f \
    \( -name '*.o' -o -name '*.json' \) -exec sha256sum -- {} + \
    | LC_ALL=C sort > "$OUT_DIR/metadata/installed_kernel_sha256.txt"
[[ "$(wc -l < "$OUT_DIR/metadata/installed_kernel_sha256.txt")" -eq 10 ]] \
    || die "expected ten installed Kernel object/JSON files"
INSTALLED_KERNEL_TREE_HASH="$(sha_file "$OUT_DIR/metadata/installed_kernel_sha256.txt")"
{
    printf 'installed_tiling_sha256=%s\n' "$INSTALLED_TILING_HASH"
    printf 'installed_master_sha256=%s\n' "$INSTALLED_MASTER_HASH"
    printf 'installed_kernel_tree_sha256=%s\n' "$INSTALLED_KERNEL_TREE_HASH"
} >> "$MANIFEST"

for spec in "${SPECS[@]}"; do
    CURRENT_SPEC="$spec"
    SPEC_DIR="$OUT_DIR/specs/$spec"
    PROFILE_DIR="$SPEC_DIR/profile"
    mkdir -- "$SPEC_DIR"

    echo "=== profiling $spec on logical NPU $DEVICE ==="
    if timeout "${MSPROF_TIMEOUT_SECONDS:-600}" msprof \
        --application="$PYTHON_PATH $PROF_MATRIX $spec" \
        --output="$PROFILE_DIR" \
        --aic-metrics=PipeUtilization \
        > "$SPEC_DIR/app.log" 2> "$SPEC_DIR/msprof.err"; then
        :
    else
        msprof_rc=$?
        die "msprof failed for $spec with exit code $msprof_rc; see $SPEC_DIR/msprof.err"
    fi

    if ! grep -F "[$spec]" "$SPEC_DIR/app.log" > "$SPEC_DIR/accuracy.txt"; then
        die "accuracy line is missing for $spec"
    fi
    accuracy_count="$(wc -l < "$SPEC_DIR/accuracy.txt")"
    [[ "$accuracy_count" -eq 1 ]] || die "expected one accuracy line for $spec, found $accuracy_count"
    grep -Fq 'acc=PASS' "$SPEC_DIR/accuracy.txt" || die "accuracy did not pass for $spec"

    mapfile -d '' -t summary_files < <(find "$PROFILE_DIR" -type f -name 'op_summary*.csv' -print0)
    ((${#summary_files[@]} == 1)) \
        || die "expected one op_summary CSV for $spec, found ${#summary_files[@]}"
    printf '%s\n' "${summary_files[0]}" > "$SPEC_DIR/op_summary_path.txt"
done

[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$GIT_HEAD" ]] || die "Git HEAD changed during collection"
[[ "$(compute_dirty_hash)" == "$DIRTY_HASH" ]] || die "working-tree content changed during collection"
[[ "$(sha_file "$SOURCE_HOST")" == "$HOST_HASH" ]] || die "Host source changed during collection"
[[ "$(sha_file "$SOURCE_TILING")" == "$TILING_HASH" ]] || die "Tiling source changed during collection"
[[ "$(sha_file "$SOURCE_KERNEL")" == "$KERNEL_HASH" ]] || die "Kernel source changed during collection"
[[ "$(sha_file "$SOURCE_CALLER")" == "$CALLER_HASH" ]] || die "caller source changed during collection"
[[ "$(sha_file "$SOURCE_HELPER")" == "$HELPER_HASH" ]] || die "helper source changed during collection"
[[ "$(sha_file "$PYBIND_PATH")" == "$PYBIND_HASH" ]] || die "pybind changed during collection"
[[ "$(sha_file "$RUN_PATH")" == "$RUN_HASH" ]] || die "run package changed during collection"
[[ "$(sha_file "$ARTIFACT_SOURCE_MANIFEST")" == "$ARTIFACT_SOURCE_MANIFEST_HASH" ]] \
    || die "artifact source manifest changed during collection"
[[ "$(sha_file "$INSTALLED_OPAPI")" == "$INSTALLED_OPAPI_HASH" ]] \
    || die "installed libcust_opapi.so changed during collection"
[[ "$(sha_file "$INSTALLED_TILING")" == "$INSTALLED_TILING_HASH" ]] \
    || die "installed liboptiling.so changed during collection"
[[ "$(sha_file "$INSTALLED_MASTER")" == "$INSTALLED_MASTER_HASH" ]] \
    || die "installed opmaster library changed during collection"
CURRENT_KERNEL_TREE_HASH="$(
    find "$INSTALLED_KERNEL_DIR" -maxdepth 1 -type f \
        \( -name '*.o' -o -name '*.json' \) -exec sha256sum -- {} + \
        | LC_ALL=C sort | sha256sum | awk '{print $1}'
)"
[[ "$CURRENT_KERNEL_TREE_HASH" == "$INSTALLED_KERNEL_TREE_HASH" ]] \
    || die "installed Kernel tree changed during collection"

npu-smi info > "$OUT_DIR/metadata/npu_status_after.txt"
printf 'npu_status_after_sha256=%s\n' \
    "$(sha_file "$OUT_DIR/metadata/npu_status_after.txt")" >> "$MANIFEST"

if "$PYTHON_PATH" "$PARSER" --out "$OUT_DIR" > "$OUT_DIR/metadata/parse.log" 2>&1; then
    :
else
    parse_rc=$?
    die "strict parsing failed with exit code $parse_rc; see metadata/parse.log"
fi

CURRENT_SPEC="none"
RUN_STATUS="complete"
echo "strict collection complete: $OUT_DIR"
echo "summary: $OUT_DIR/summary.csv"
