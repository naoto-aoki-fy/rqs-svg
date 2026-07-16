NVCC_GENCODE_FLAGS = -gencode=arch=compute_100,code=sm_100
NVCCOPTIONS = -I/path/to/include -L/path/to/lib -Xlinker -rpath -Xlinker /path/to/lib -lfoobar
