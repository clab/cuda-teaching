#include <cstdlib>
#include <cublas_v2.h>
#include <curand.h>

#include <stdexcept>
#include <iostream>
#include <string>
#include <chrono>

#define CUDA_CHECK(stmt) do {                              \
    cudaError_t err = stmt;                                \
    if (err != cudaSuccess) {                              \
      std::cerr << "CUDA failure in " << #stmt << std::endl\
                << cudaGetErrorString(err) << std::endl;   \
      throw std::runtime_error(#stmt);                     \
    }                                                      \
  } while(0)

#define CUBLAS_CHECK(stmt) do {                            \
    cublasStatus_t stat = stmt;                            \
    if (stat != CUBLAS_STATUS_SUCCESS) {                   \
      std::cerr << "CUBLAS failure in " << #stmt           \
                << std::endl << stat << std::endl;         \
      throw std::runtime_error(#stmt);                     \
    }                                                      \
  } while(0)

using namespace std;
cublasHandle_t handle;
curandGenerator_t prng;

void GPU_randomize(float *M, int r, int c) { curandGenerateUniform(prng, M, r * c); }

struct T {
  explicit T(const std::string& x) : x(x), start(std::chrono::high_resolution_clock::now()) {}
  ~T() {
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(end - start);
    cerr << x << " took " << time_span.count() << "s\n";
  }
  void report() { cerr << x << endl; }
  const std::string x;
  std::chrono::high_resolution_clock::time_point start;
};

void Initialize_GPU(int& argc, char**& argv) {
  int nDevices;
  CUDA_CHECK(cudaGetDeviceCount(&nDevices));
  if (nDevices < 1) {
    cerr << "No GPUs found\n";
    abort();
  }
  size_t free_bytes, total_bytes, max_free = 0;
  int selected = 0;
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
    cerr << "Device Number: " << i << endl;
    cerr << "  Device name: " << prop.name << endl;
    cerr << "  Memory Clock Rate (KHz): " << prop.memoryClockRate << endl;
    cerr << "  Memory Bus Width (bits): " << prop.memoryBusWidth << endl;
    cerr << "  Peak Memory Bandwidth (GB/s): " << (2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6) << endl << endl;
    CUDA_CHECK(cudaSetDevice(i));
    CUDA_CHECK(cudaMemGetInfo( &free_bytes, &total_bytes ));
    CUDA_CHECK(cudaDeviceReset());
    cerr << "  Memory Free (MB): " << (int)free_bytes/1.0e6 << "/" << (int)total_bytes/1.0e6 << endl << endl;
    if(free_bytes > max_free) {
        max_free = free_bytes;
        selected = i;
    }
  }
  cerr << "**USING DEVICE: " << selected << endl;
  CUDA_CHECK(cudaSetDevice(selected));
  CUBLAS_CHECK(cublasCreate(&handle));
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);  // TODO check for errors
  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock()); // TODO check for errors
}

int main(int argc, char**argv) {
  Initialize_GPU(argc, argv);
  const int msize = 128;
  for (unsigned d1 = 8; d1 < 15; ++d1) {
    for (unsigned d2 = 8; d2 < 15; ++d2) {
      const int hsize = pow(2, d1);
      const int xsize = pow(2, d2);
      cerr << "DIM: " << d1 << ' ' << d2 << ' ' << hsize << ' ' << xsize << endl;

      float *W, *X, *Y;
      cudaMalloc(&W, hsize * xsize * sizeof(float));
      cudaMalloc(&X, xsize * msize * sizeof(float));
      cudaMalloc(&Y, hsize * msize * sizeof(float));
      GPU_randomize(W, hsize, xsize);
      GPU_randomize(X, xsize, msize);
      GPU_randomize(Y, hsize, msize);

      { T t("GPU MM");
        //Y(hsize, msize) = W(hsize, xsize) * X(xsize, msize)
        int lda = hsize,ldb=xsize,ldc=hsize;
        const float alf = 1;
        const float bet = 0;
        const float *alpha = &alf;
        const float *beta = &bet;
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, hsize, msize, xsize, alpha, W, lda, X, ldb, beta, Y, ldc));
        CUDA_CHECK(cudaThreadSynchronize());
      }

      cudaFree(W);
      cudaFree(X);
      cudaFree(Y);
    }
  }
}

