#!/usr/bin/env bash
set -euo pipefail
: "${CXX:=g++}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR_QCS_H="${SCRIPT_DIR}/include"

"$CXX" -fPIC -shared -I"${DIR_QCS_H}" -std=c++17 "$@"
