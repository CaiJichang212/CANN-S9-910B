#!/usr/bin/env bash
# Build Greater in the required s8 image and create an official submission zip.
#
# Usage:
#   bash build_and_pack.sh
#
# Output:
#   releases/Greater-YYYYmmdd_HHMMSS/{Greater-YYYYmmdd_HHMMSS.zip,manifest.yaml}

set -Eeuo pipefail

readonly OP_NAME="Greater"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly HW_S9_DIR="$(dirname "$SCRIPT_DIR")"
readonly OP_PROJECT="$SCRIPT_DIR/$OP_NAME/op_project/custom_greater"
readonly ZIP_SCRIPT="$HW_S9_DIR/zip_op.sh"
readonly RELEASES_DIR="$SCRIPT_DIR/releases"
readonly RELEASE_INDEX="$RELEASES_DIR/index.csv"
readonly PACK_IMAGE_DEFAULT="swr.cn-southwest-2.myhuaweicloud.com/fuyangchenghu/cann8.5:s8"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

sha_file() {
    sha256sum -- "$1" | awk '{print $1}'
}

validate_zip() {
    local zip_path="$1"
    local run_path="$2"
    local run_name
    run_name="$(basename "$run_path")"

    command -v unzip >/dev/null 2>&1 || die "unzip is required"
    command -v cmp >/dev/null 2>&1 || die "cmp is required"
    command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"

    local -a expected_entries=(
        "${OP_NAME}_zip/"
        "${OP_NAME}_zip/op_host/"
        "${OP_NAME}_zip/op_host/CMakeLists.txt"
        "${OP_NAME}_zip/op_host/greater.cpp"
        "${OP_NAME}_zip/op_host/greater_tiling.h"
        "${OP_NAME}_zip/op_kernel/"
        "${OP_NAME}_zip/op_kernel/CMakeLists.txt"
        "${OP_NAME}_zip/op_kernel/greater.cpp"
        "${OP_NAME}_zip/${run_name}"
    )
    local -a actual_entries=()
    mapfile -t actual_entries < <(unzip -Z1 "$zip_path")

    [[ "${#actual_entries[@]}" -eq "${#expected_entries[@]}" ]] ||
        die "unexpected zip entry count: ${#actual_entries[@]}"

    declare -A expected_map=()
    local entry
    for entry in "${expected_entries[@]}"; do
        expected_map["$entry"]=1
    done
    for entry in "${actual_entries[@]}"; do
        [[ -n "${expected_map[$entry]+x}" ]] || die "unexpected zip entry: $entry"
    done

    unzip -p "$zip_path" "${OP_NAME}_zip/op_host/CMakeLists.txt" |
        cmp - "$OP_PROJECT/op_host/CMakeLists.txt"
    unzip -p "$zip_path" "${OP_NAME}_zip/op_host/greater.cpp" |
        cmp - "$OP_PROJECT/op_host/greater.cpp"
    unzip -p "$zip_path" "${OP_NAME}_zip/op_host/greater_tiling.h" |
        cmp - "$OP_PROJECT/op_host/greater_tiling.h"
    unzip -p "$zip_path" "${OP_NAME}_zip/op_kernel/CMakeLists.txt" |
        cmp - "$OP_PROJECT/op_kernel/CMakeLists.txt"
    unzip -p "$zip_path" "${OP_NAME}_zip/op_kernel/greater.cpp" |
        cmp - "$OP_PROJECT/op_kernel/greater.cpp"

    local packaged_run_sha
    packaged_run_sha="$(unzip -p "$zip_path" "${OP_NAME}_zip/${run_name}" | sha256sum | awk '{print $1}')"
    [[ "$packaged_run_sha" == "$(sha_file "$run_path")" ]] ||
        die "packaged run hash does not match build output"

    unzip -tqq "$zip_path"
}

validate_release_layout() {
    local release_root="$1"
    local release_id
    release_id="$(basename "$release_root")"
    local -a entries=()
    mapfile -t entries < <(find "$release_root" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)
    [[ "${#entries[@]}" -eq 2 ]] ||
        die "release must contain exactly two entries, found ${#entries[@]}"
    [[ -f "$release_root/manifest.yaml" && -f "$release_root/$release_id.zip" ]] ||
        die "unexpected release layout: ${entries[*]}"
}

write_release_metadata() {
    local release_id="$1"
    local final_zip="$2"
    local run_path="$3"
    local release_dir
    release_dir="$(dirname "$final_zip")"
    local manifest_path="$release_dir/manifest.yaml"
    local source_commit="${PACK_SOURCE_COMMIT:-unknown}"
    local source_dirty="${PACK_SOURCE_DIRTY:-unknown}"
    local created_at="${PACK_CREATED_AT:-unknown}"
    local zip_sha run_sha run_name
    zip_sha="$(sha_file "$final_zip")"
    run_sha="$(sha_file "$run_path")"
    run_name="$(basename "$run_path")"

    printf '%s\n' \
        'schema_version: 1' \
        'operator: Greater' \
        "release_id: $release_id" \
        "created_at: $created_at" \
        'status: pending_official_submission' \
        'source:' \
        "  commit: $source_commit" \
        "  dirty: $source_dirty" \
        "  host_sha256: $(sha_file "$OP_PROJECT/op_host/greater.cpp")" \
        "  tiling_sha256: $(sha_file "$OP_PROJECT/op_host/greater_tiling.h")" \
        "  kernel_sha256: $(sha_file "$OP_PROJECT/op_kernel/greater.cpp")" \
        'build:' \
        '  cann_version: 8.5.0' \
        '  target: ascend910b' \
        "  image: ${PACK_IMAGE:-$PACK_IMAGE_DEFAULT}" \
        'package:' \
        "  path: $release_id.zip" \
        "  sha256: $zip_sha" \
        "  run_filename: $run_name" \
        "  run_sha256: $run_sha" \
        '  validation: exact_manifest_and_source_match' \
        > "$manifest_path"

    [[ -f "$RELEASE_INDEX" ]] || die "missing release index: $RELEASE_INDEX"
    [[ "$(sed -n '1p' "$RELEASE_INDEX")" == \
        "release_id,created_at,status,source_commit,package,zip_sha256,run_sha256,manifest,official_feedback" ]] ||
        die "unexpected release index schema"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$release_id" "$created_at" "pending_official_submission" "$source_commit" \
        "releases/$release_id/$release_id.zip" "$zip_sha" "$run_sha" \
        "releases/$release_id/manifest.yaml" "" \
        >> "$RELEASE_INDEX"
}

build_inside_s8() {
    local timestamp="${PACK_TIMESTAMP:-}"
    [[ "$timestamp" =~ ^[0-9]{8}_[0-9]{6}$ ]] || die "invalid PACK_TIMESTAMP: $timestamp"
    [[ -f "$ZIP_SCRIPT" ]] || die "missing official zip script: $ZIP_SCRIPT"
    [[ -f "$OP_PROJECT/op_host/greater.cpp" ]] || die "missing Greater Host source"
    [[ -f "$OP_PROJECT/op_host/greater_tiling.h" ]] || die "missing Greater TilingData source"
    [[ -f "$OP_PROJECT/op_kernel/greater.cpp" ]] || die "missing Greater Kernel source"

    # The s8 image defaults to CANN 7.0. Submission builds must explicitly use 8.5.0.
    export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
    set +u
    source /home/ma-user/Ascend/cann-8.5.0/set_env.sh
    set -u
    unset BASE_LIBS_PATH
    export ASCEND_HOME_PATH=/home/ma-user/Ascend/cann-8.5.0
    export ASCEND_AICPU_PATH="$ASCEND_HOME_PATH"
    export ASCEND_OPP_PATH="$ASCEND_HOME_PATH/opp"
    export PATH="/home/ma-user/cmake-3.28.3-linux-aarch64/bin:$PATH"

    local version_file="$ASCEND_HOME_PATH/compiler/version.info"
    [[ -f "$version_file" ]] || die "missing CANN version file: $version_file"
    [[ "$(sed -n 's/^Version=//p' "$version_file")" == "8.5.0" ]] ||
        die "submission build is not using CANN 8.5.0"
    [[ "$(command -v cmake)" == "/home/ma-user/cmake-3.28.3-linux-aarch64/bin/cmake" ]] ||
        die "submission build is not using the required CMake"
    [[ "$(cmake --version | sed -n '1s/.* //p')" == "3.28.3" ]] ||
        die "submission build is not using CMake 3.28.3"

    echo "===== [1/4] Build Greater in s8 ====="
    cd "$OP_PROJECT"
    bash build.sh

    shopt -s nullglob
    local -a run_files=("$OP_PROJECT"/build_out/custom_opp_*.run)
    shopt -u nullglob
    [[ "${#run_files[@]}" -eq 1 ]] ||
        die "expected exactly one custom_opp_*.run, found ${#run_files[@]}"
    local run_path="${run_files[0]}"

    local release_id="${OP_NAME}-${timestamp}"
    local submission_root="/tmp/${release_id}-submission"
    local pack_op_dir="$submission_root/op/$OP_NAME"
    local zipfiles_dir="$submission_root/zipfiles"
    local release_root="$RELEASES_DIR/$release_id"
    local final_zip="$release_root/$release_id.zip"
    [[ ! -e "$submission_root" ]] || die "submission directory already exists: $submission_root"
    [[ ! -e "$release_root" ]] || die "release already exists: $release_root"

    echo "===== [2/4] Prepare official three-part layout ====="
    mkdir -p "$pack_op_dir/build_out" "$zipfiles_dir"
    cp -a "$OP_PROJECT/op_host" "$pack_op_dir/op_host"
    cp -a "$OP_PROJECT/op_kernel" "$pack_op_dir/op_kernel"
    cp -a "$run_path" "$pack_op_dir/build_out/$(basename "$run_path")"

    echo "===== [3/4] Package with official zip_op.sh ====="
    cd "$zipfiles_dir"
    bash "$ZIP_SCRIPT" "$OP_NAME"
    [[ -f "$zipfiles_dir/${OP_NAME}.zip" ]] || die "official zip script did not create ${OP_NAME}.zip"

    echo "===== [4/4] Validate submission ====="
    validate_zip "$zipfiles_dir/${OP_NAME}.zip" "$run_path"
    mkdir -p "$release_root"
    mv "$zipfiles_dir/${OP_NAME}.zip" "$final_zip"
    write_release_metadata "$release_id" "$final_zip" "$run_path"
    validate_release_layout "$release_root"

    echo "RELEASE_DIR=$release_root"
    echo "RUN_FILE=$run_path"
    echo "RUN_SHA256=$(sha_file "$run_path")"
    echo "ZIP_FILE=$final_zip"
    echo "ZIP_SHA256=$(sha_file "$final_zip")"
}

main() {
    if [[ "${1:-}" == "--inside-s8" ]]; then
        [[ "${PACK_IN_S8:-0}" == "1" ]] || die "--inside-s8 is reserved for the s8 container"
        build_inside_s8
        return
    fi
    [[ "$#" -eq 0 ]] || die "usage: bash build_and_pack.sh"

    command -v sudo >/dev/null 2>&1 || die "sudo is required"
    command -v docker >/dev/null 2>&1 || die "docker is required"

    local timestamp="${PACK_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
    [[ "$timestamp" =~ ^[0-9]{8}_[0-9]{6}$ ]] || die "invalid PACK_TIMESTAMP: $timestamp"
    local pack_image="${PACK_IMAGE:-$PACK_IMAGE_DEFAULT}"
    local container_name="greater-pack-${timestamp}"
    local release_id="${OP_NAME}-${timestamp}"
    local release_root="$RELEASES_DIR/$release_id"
    local final_zip="$release_root/$release_id.zip"
    local source_commit source_dirty created_at
    source_commit="$(git -C "$SCRIPT_DIR" rev-parse HEAD)"
    source_dirty=false
    [[ -z "$(git -C "$SCRIPT_DIR" status --porcelain)" ]] || source_dirty=true
    created_at="$(date --iso-8601=seconds)"

    [[ ! -e "$release_root" ]] || die "release already exists: $release_root"

    echo "Using s8 image: $pack_image"
    echo "Release ID: $release_id"
    sudo -n docker run --rm \
        --name "$container_name" \
        --network none \
        --user 0:0 \
        --mount "type=bind,src=${HW_S9_DIR},dst=${HW_S9_DIR}" \
        -w "$SCRIPT_DIR" \
        -e PACK_IN_S8=1 \
        -e PACK_TIMESTAMP="$timestamp" \
        -e PACK_SOURCE_COMMIT="$source_commit" \
        -e PACK_SOURCE_DIRTY="$source_dirty" \
        -e PACK_CREATED_AT="$created_at" \
        -e PACK_IMAGE="$pack_image" \
        --entrypoint /bin/bash \
        "$pack_image" \
        -lc 'bash "$PWD/build_and_pack.sh" --inside-s8'

    local uid gid
    uid="$(id -u)"
    gid="$(id -g)"
    sudo -n chown -R "$uid:$gid" "$OP_PROJECT/build_out" "$release_root"

    # Recheck on the host after ownership restoration.
    shopt -s nullglob
    local -a run_files=("$OP_PROJECT"/build_out/custom_opp_*.run)
    shopt -u nullglob
    [[ "${#run_files[@]}" -eq 1 ]] || die "host sees ${#run_files[@]} build outputs"
    validate_zip "$final_zip" "${run_files[0]}"
    validate_release_layout "$release_root"

    echo "===== Done ====="
    echo "ZIP_FILE=$final_zip"
    echo "ZIP_SHA256=$(sha_file "$final_zip")"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
