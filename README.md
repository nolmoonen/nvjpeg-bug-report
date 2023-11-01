# nvjpeg-bug-report

Demonstrates `nvjpegEncodeImage` failing with status code `NVJPEG_STATUS_EXECUTION_FAILED` when encoding RGB data (`NVJPEG_INPUT_RGB`) to a chroma-subsampled format (e.g. `NVJPEG_CSS_410`) for a specific size (e.g. 101x101).

Tested with CUDA 12.2 driver version 535.113.01 on Ubuntu 22.04.02.

The Compute Sanitizer reports an "Invalid `__global__` read of size 4 bytes at .. in `void nvjpeg::format_to_ycbcr_kernel`..".
