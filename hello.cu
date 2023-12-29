#include <chrono>
#include <cmath>
#include <iostream>

__global__ void add(int n, float* a, float* b) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    b[i] = a[i] + b[i];
  }
}

int main() {
  int N = 1 << 28;
  float* x = new float[N];
  float* y = new float[N];

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  float *d_x, *d_y;
  cudaMalloc(&d_x, N * sizeof(float));
  cudaMalloc(&d_y, N * sizeof(float));

  cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 1024;
  int numBlocks = 32;
  std::cout << "Launching kernel with " << numBlocks << " blocks of "
            << blockSize << " threads each" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; i++) {
    add<<<numBlocks, blockSize>>>(N, d_x, d_y);
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
      std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Time taken: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                   .count()
            << " ns" << std::endl;

  cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = std::max(maxError, std::abs(y[i] - 3.0f));
  }
  std::cout << "Max error: " << maxError << std::endl;

  cudaFree(d_x);
  cudaFree(d_y);
  delete[] x;
  delete[] y;
}
