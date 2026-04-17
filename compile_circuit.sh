#!/usr/bin/env bash
set -euo pipefail
: "${CXX:=g++}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR_QCS_HPP="${SCRIPT_DIR}/include"

"$CXX" -fPIC -shared -I"${DIR_QCS_HPP}" -std=c++17 "$@"
