#!/usr/bin/env bash
set -euo pipefail
umask 077

source "$(dirname "$0")/env.sh"

mkdir -p "$TASK_HERE/metadata" "$TASK_HERE/correctness" "$PRIVATE_ROOT"

current_files=(
  "$TASK_ROOT/op/CustomOp/op_host/concat.cpp"
  "$TASK_ROOT/op/CustomOp/op_host/concat_tiling.h"
  "$TASK_ROOT/op/CustomOp/op_kernel/concat.cpp"
)
archive_files=(
  "$TASK_ROOT/Concat_20260722_102940_zip/op_host/concat.cpp"
  "$TASK_ROOT/Concat_20260722_102940_zip/op_host/concat_tiling.h"
  "$TASK_ROOT/Concat_20260722_102940_zip/op_kernel/concat.cpp"
)

for index in 0 1 2; do
  if ! cmp -s "${current_files[$index]}" "${archive_files[$index]}"; then
    echo "baseline source mismatch: ${current_files[$index]}" >&2
    exit 1
  fi
done

sha256sum "${current_files[@]}" "${archive_files[@]}" > "$TASK_HERE/metadata/prebuild_source_sha256.txt"
if [[ ! -f "$TASK_TMPDIR/build_root.log" ]]; then
  echo "missing root build log; run build_opp_root.sh first" >&2
  exit 1
fi
cp -f "$TASK_TMPDIR/build_root.log" "$TASK_HERE/metadata/build.log"

packages=("$TASK_TMPDIR/CustomOp_build_src/build_out"/custom_opp_*.run)
if [[ ${#packages[@]} -ne 1 || ! -f "${packages[0]}" ]]; then
  echo "expected one rebuilt custom_opp package, got ${#packages[@]}" >&2
  exit 1
fi

if [[ -d "$PRIVATE_ROOT/opp" ]]; then
  chmod -R u+w "$PRIVATE_ROOT/opp"
fi
rm -rf "$PRIVATE_ROOT/package" "$PRIVATE_ROOT/opp" "$PRIVATE_ROOT/wheel_src" \
       "$PRIVATE_ROOT/wheel_build" "$PRIVATE_ROOT/wheel_dist" "$WHEEL_SITE" "$TORCH_EXTENSIONS_DIR"
mkdir -p "$PRIVATE_ROOT/package" "$PRIVATE_ROOT/opp" "$PRIVATE_ROOT/wheel_src" \
         "$PRIVATE_ROOT/wheel_build" "$PRIVATE_ROOT/wheel_dist" "$WHEEL_SITE" "$TORCH_EXTENSIONS_DIR"
cp -f "${packages[0]}" "$PRIVATE_ROOT/package/"
private_packages=("$PRIVATE_ROOT/package"/custom_opp_*.run)
"${private_packages[0]}" --quiet --install-path="$PRIVATE_ROOT/opp" >> "$TASK_HERE/metadata/build.log" 2>&1

cp -f "$TASK_ROOT/Concat/setup.py" "$PRIVATE_ROOT/wheel_src/"
cp -a "$TASK_ROOT/Concat/extension" "$PRIVATE_ROOT/wheel_src/"
cp -a "$TASK_ROOT/Concat/common" "$PRIVATE_ROOT/wheel_src/"
(
  cd "$PRIVATE_ROOT/wheel_src"
  "$PYTHON_BIN" setup.py build --build-base="$PRIVATE_ROOT/wheel_build" \
    bdist_wheel --dist-dir="$PRIVATE_ROOT/wheel_dist"
) >> "$TASK_HERE/metadata/build.log" 2>&1

wheels=("$PRIVATE_ROOT/wheel_dist"/custom_ops*.whl)
if [[ ${#wheels[@]} -ne 1 || ! -f "${wheels[0]}" ]]; then
  echo "expected one custom_ops wheel, got ${#wheels[@]}" >&2
  exit 1
fi
"$PYTHON_BIN" -m pip install --no-deps --target "$WHEEL_SITE" "${wheels[0]}" \
  >> "$TASK_HERE/metadata/build.log" 2>&1

"$PYTHON_BIN" "$TASK_HERE/cases.py"
"$PYTHON_BIN" "$TASK_HERE/tiling_model.py"
"$PYTHON_BIN" "$TASK_HERE/scripts/collect_metadata.py"

echo "PRIVATE_ENV_READY package=${private_packages[0]} vendor=$PRIVATE_VENDOR wheel=${wheels[0]}"
