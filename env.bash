_qcs_repo_dir="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
_qcs_lib_dir="${_qcs_repo_dir}/lib"
_qcs_include_dir="${_qcs_repo_dir}/include"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${_qcs_lib_dir}"
export CPATH="${CPATH:+${CPATH}:}${_qcs_include_dir}"
unset _qcs_repo_dir _qcs_lib_dir _qcs_include_dir
