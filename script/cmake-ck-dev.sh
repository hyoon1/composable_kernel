#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# exit when a command exits with non-zero status; also when an unbound variable is referenced
set -eu
# pipefail is supported by many shells, not supported by sh and dash
set -o pipefail 2>/dev/null | true
# when treating a string as a sequence, do not split on spaces
IFS=$(printf '\n\t')

# clean the build system files
find . -name CMakeFiles     -type d -exec rm -rfv {} +
find . -name CMakeCache.txt -type f -exec rm -rv  {} +

if [ $# -ge 1 ]; then
    MY_PROJECT_SOURCE="$1"
    shift 1
else
    MY_PROJECT_SOURCE=".."
fi

GPU_TARGETS="gfx908;gfx90a;gfx942"

if [ $# -ge 1 ]; then
    case "$1" in
        gfx*)
            GPU_TARGETS="$1"
            shift 1
            echo "GPU targets provided: $GPU_TARGETS"
            REST_ARGS=("$@")
            ;;
        *)
            REST_ARGS=("$@")
            ;;
    esac
else
    REST_ARGS=("$@")
fi

is_truthy() {
    case "${1:-}" in
        1 | true | TRUE | on | ON | yes | YES) return 0 ;;
        *) return 1 ;;
    esac
}

# Optional dev knobs for generating ISA / resource-usage reports.
# Examples:
#   CK_SAVE_TEMPS=1 CK_KERNEL_RESOURCE_USAGE=1 ../script/cmake-ck-dev.sh .. gfx1201
#   CK_SAVE_TEMPS=1 CK_SAVE_TEMPS_MODE=cwd ../script/cmake-ck-dev.sh .. gfx1201
#   CK_SAVE_TEMPS=1 CK_KERNEL_RESOURCE_USAGE=1 CK_DEBUG_SYMBOLS=1 ../script/cmake-ck-dev.sh .. gfx1201
CK_SAVE_TEMPS="${CK_SAVE_TEMPS:-0}"
CK_SAVE_TEMPS_MODE="${CK_SAVE_TEMPS_MODE:-obj}" # obj|cwd
CK_KERNEL_RESOURCE_USAGE="${CK_KERNEL_RESOURCE_USAGE:-0}"
CK_DEBUG_SYMBOLS="${CK_DEBUG_SYMBOLS:-0}"

EXTRA_HIP_FLAGS=""
if is_truthy "${CK_SAVE_TEMPS}"; then
    if [ "${CK_SAVE_TEMPS_MODE}" = "cwd" ]; then
        EXTRA_HIP_FLAGS="${EXTRA_HIP_FLAGS} --save-temps"
    else
        EXTRA_HIP_FLAGS="${EXTRA_HIP_FLAGS} -save-temps=obj"
    fi
fi
if is_truthy "${CK_KERNEL_RESOURCE_USAGE}"; then
    EXTRA_HIP_FLAGS="${EXTRA_HIP_FLAGS} -Rpass-analysis=kernel-resource-usage"
fi
if is_truthy "${CK_DEBUG_SYMBOLS}"; then
    EXTRA_HIP_FLAGS="${EXTRA_HIP_FLAGS} -g"
fi

if [ -n "${EXTRA_HIP_FLAGS}" ]; then
    appended_hip_flags=0
    for i in "${!REST_ARGS[@]}"; do
        case "${REST_ARGS[$i]}" in
            -DCMAKE_HIP_FLAGS=*)
                REST_ARGS[$i]="${REST_ARGS[$i]} ${EXTRA_HIP_FLAGS# }"
                appended_hip_flags=1
                break
                ;;
        esac
    done
    if [ "${appended_hip_flags}" -eq 0 ]; then
        REST_ARGS+=("-DCMAKE_HIP_FLAGS=${EXTRA_HIP_FLAGS# }")
    fi
fi

cmake "${MY_PROJECT_SOURCE}" --preset dev -DGPU_TARGETS="$GPU_TARGETS" "${REST_ARGS[@]}"
