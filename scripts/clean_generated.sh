#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} != "" && ${1:-} != "--apply" ]]; then
    echo "Usage: $0 [--apply]" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
expected_root=$(cd -- "$script_dir/.." && pwd -P)
repo_root=$(git -C "$expected_root" rev-parse --show-toplevel)
if [[ $repo_root != "$expected_root" ]]; then
    echo "Refusing to clean unexpected Git root: $repo_root" >&2
    exit 1
fi

git_cmd=(git -c "safe.directory=$repo_root" -C "$repo_root")
apply=false
[[ ${1:-} == "--apply" ]] && apply=true
count=0

while IFS= read -r -d '' absolute_path; do
    relative_path=${absolute_path#"$repo_root"/}
    case "$relative_path" in
        *_zip|*_zip/*|releases/*.zip|releases/*/*.zip)
            continue
            ;;
    esac

    if ! "${git_cmd[@]}" check-ignore -q -- "$relative_path"; then
        continue
    fi
    if [[ -n $("${git_cmd[@]}" ls-files -- "$relative_path") ]]; then
        echo "skip tracked: $relative_path" >&2
        continue
    fi

    ((count += 1))
    if $apply; then
        echo "remove: $relative_path"
        "${git_cmd[@]}" clean -fdX -- "$relative_path"
    else
        echo "would remove: $relative_path"
    fi
done < <(
    find "$repo_root" -xdev -mindepth 1 \
        -path "$repo_root/.git" -prune -o \
        -type d \( \
            -name build -o -name build_out -o -name dist -o \
            -name __pycache__ -o -name '*.egg-info' -o \
            -name kernel_meta -o -name extra-info -o \
            -name raw -o -name private -o -name profile -o \
            -name 'PROF_*' -o -name opp -o -name '.local_opp' -o \
            -name '.local_python' -o -name 'npu_opp.*' -o \
            -name 'private_opp.*' -o -name '.submit_opp_*' -o \
            -name final_opp -o -name '.final_opp.*' -o \
            -name '.indexadd_opp.*' \
        \) -prune -print0 -o \
        -type f \( \
            -name '*.pyc' -o -name '*.so' -o \
            -name '*_profiling_output_record' -o \
            -name '.indexadd_opp_path' \
        \) -print0
)

if $apply; then
    echo "Removed $count ignored target(s)."
else
    echo "Dry run: $count ignored target(s). Re-run with --apply to remove them."
fi
