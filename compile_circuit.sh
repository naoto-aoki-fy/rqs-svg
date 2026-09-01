#!/usr/bin/env bash
set -euo pipefail
: "${CC:=gcc}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR_QCS_H="${SCRIPT_DIR}/include"

"$CC" -fPIC -shared -I"${DIR_QCS_H}" -std=c11 "$@"
