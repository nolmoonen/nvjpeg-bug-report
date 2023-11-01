#include <cuda_runtime_api.h>
#include <nvjpeg.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA error \"" << cudaGetErrorString(error) << "\" ("      \
                << static_cast<int>(error) << ") at " __FILE__ ":" << __LINE__ \
                << "\n";                                                       \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define CHECK_NVJPEG(call)                                                     \
  do {                                                                         \
    nvjpegStatus_t status = call;                                              \
    if (status != NVJPEG_STATUS_SUCCESS) {                                     \
      std::cerr << "nvJPEG error " << static_cast<int>(status)                 \
                << " at " __FILE__ ":" << __LINE__ << "\n";                    \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

int main() {
  // encoder initialization
  cudaStream_t stream = cudaStreamDefault;
  nvjpegHandle_t nv_handle;
  CHECK_NVJPEG(nvjpegCreateSimple(&nv_handle));
  nvjpegEncoderState_t nv_enc_state;
  CHECK_NVJPEG(nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, stream));
  nvjpegEncoderParams_t nv_enc_params;
  CHECK_NVJPEG(nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, stream));

  // size is relevant, only some specific sizes fail,
  //   e.g. 100x100 passes but 101x101 fails
  const int size_x = 101;
  const int size_y = 101;
  const size_t image_pixels = size_x * size_y * 3;

  // set some pattern for testing for testing purposes
  std::vector<unsigned char> h_image(image_pixels);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < size_y; ++y) {
      for (int x = 0; x < size_x; ++x) {
        const size_t idx = c * size_x * size_y + y * size_x + x;
        h_image[idx] = idx ^ c;
      }
    }
  }

  // allocate and initialize image
  nvjpegImage_t nv_image;
  for (int c = 0; c < 3; ++c) {
    const size_t channel_size = size_x * size_y;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&nv_image.channel[c]),
                          channel_size));
    CHECK_CUDA(cudaMemcpy(nv_image.channel[c],
                          h_image.data() + channel_size * c, channel_size,
                          cudaMemcpyHostToDevice));
    nv_image.pitch[c] = size_x;
  }

  // encode the RGB image to a chroma-subsampled JPEG
  //   any chroma-subsampled format results in NVJPEG_STATUS_EXECUTION_FAILED
  CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(
      nv_enc_params, NVJPEG_CSS_410, cudaStreamDefault));
  CHECK_NVJPEG(nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params,
                                 &nv_image, NVJPEG_INPUT_RGB, size_x, size_y,
                                 cudaStreamDefault));

  size_t length = 0;
  CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, nullptr,
                                             &length, cudaStreamDefault));
  CHECK_CUDA(cudaDeviceSynchronize());
  std::vector<char> jpeg(length);
  CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(
      nv_handle, nv_enc_state, reinterpret_cast<unsigned char *>(jpeg.data()),
      &length, cudaStreamDefault));

  // write stream to file
  CHECK_CUDA(cudaDeviceSynchronize());
  std::stringstream ss;
  ss << "out_" << size_x << "_" << size_y << ".jpg";
  std::ofstream output_file(ss.str(), std::ios::out | std::ios::binary);
  output_file.write(jpeg.data(), length);
  output_file.close();

  for (int c = 0; c < 3; ++c) {
    CHECK_CUDA(cudaFree(nv_image.channel[c]));
  }

  CHECK_NVJPEG(nvjpegEncoderParamsDestroy(nv_enc_params));
  CHECK_NVJPEG(nvjpegEncoderStateDestroy(nv_enc_state));
  CHECK_NVJPEG(nvjpegDestroy(nv_handle));
}
