#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR_QCS_HPP="${SCRIPT_DIR}/include"

g++ -fPIC -shared -I"${DIR_QCS_HPP}" -std=c++17 "$@"
