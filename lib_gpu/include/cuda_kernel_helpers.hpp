#pragma once
#include <cuda_runtime_api.h>
#include <cassert>

namespace lib_gpu {
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

class CudaKernelHelpers {
 public:
  static __device__ void inplace_reverse_prefixsum(unsigned int *odata,
                                                   int n) {
    int thid = threadIdx.x;
    int offset = 1;

    for (int d = n >> 1; d > 0; d >>= 1) {
      __syncthreads();
      for (int i = thid; i < d; i += blockDim.x) {
        int ai = offset * (2 * i + 1) - 1;
        int bi = offset * (2 * i + 2) - 1;
        odata[ai] += odata[bi];
      }
      offset *= 2;
    }

    if (thid == 0) odata[0] = 0;

    for (int d = 1; d < n; d *= 2) {
      offset >>= 1;
      __syncthreads();
      for (int i = thid; i < d; i += blockDim.x) {
        int ai = offset * (2 * i + 1) - 1;
        int bi = offset * (2 * i + 2) - 1;
        unsigned int t = odata[bi];
        odata[bi] = odata[ai];
        odata[ai] += t;
      }
    }

    __syncthreads();
  }

  static __device__ void prescan(float *odata, float *idata, float *temp,
                                 int n) {
    int thid = threadIdx.x;
    int offset = 1;

    int ai = thid;
    int bi = thid + (n / 2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    temp[ai + bankOffsetA] = idata[ai];
    temp[bi + bankOffsetB] = idata[bi];

    for (int d = n >> 1; d > 0; d >>= 1) {
      __syncthreads();
      if (thid < d) {
        int ai = offset * (2 * thid + 1) - 1;
        int bi = offset * (2 * thid + 2) - 1;
        ai += CONFLICT_FREE_OFFSET(ai);
        bi += CONFLICT_FREE_OFFSET(bi);

        temp[bi] += temp[ai];
      }
      offset *= 2;

      if (thid == 0) temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;

      for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
          int ai = offset * (2 * thid + 1) - 1;
          int bi = offset * (2 * thid + 2) - 1;
          ai += CONFLICT_FREE_OFFSET(ai);
          bi += CONFLICT_FREE_OFFSET(bi);

          float t = temp[ai];
          temp[ai] = temp[bi];
          temp[bi] += t;
        }
      }
      __syncthreads();

      odata[ai] = temp[ai + bankOffsetA];
      odata[bi] = temp[bi + bankOffsetB];
    }
  }

  static __device__ void prefixsum(float *odata, float *idata, int n) {
    int thid = threadIdx.x;
    int offset = 1;

    int ai = thid;
    int bi = thid + (n / 2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    for (int i = 0; i < n; i += blockDim.x) {
      odata[i + ai + bankOffsetA] = idata[i + ai];
      odata[i + bi + bankOffsetB] = idata[i + bi];
    }

    for (int d = n >> 1; d > 0; d >>= 1) {
      __syncthreads();
      if (thid < d) {
        int ai = offset * (2 * thid + 1) - 1;
        int bi = offset * (2 * thid + 2) - 1;
        ai += CONFLICT_FREE_OFFSET(ai);
        bi += CONFLICT_FREE_OFFSET(bi);

        odata[bi] += odata[ai];
      }
      offset *= 2;

      if (thid == 0) odata[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;

      for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
          int ai = offset * (2 * thid + 1) - 1;
          int bi = offset * (2 * thid + 2) - 1;
          ai += CONFLICT_FREE_OFFSET(ai);
          bi += CONFLICT_FREE_OFFSET(bi);

          float t = odata[ai];
          odata[ai] = odata[bi];
          odata[bi] += t;
        }
      }
      __syncthreads();

      odata[ai] = odata[ai + bankOffsetA];
      odata[bi] = odata[bi + bankOffsetB];
    }
  }

  static __device__ void inplace_prefixsum(unsigned int *data, int n) {
    int thid = threadIdx.x;
    int offset = 1;

    for (int d = n >> 1; d > 0; d >>= 1) {
      __syncthreads();
      for (int i = thid; i < d; i += blockDim.x) {
        int ai = offset * (2 * i + 1) - 1;
        int bi = offset * (2 * i + 2) - 1;
        data[bi] += data[ai];
      }
      offset *= 2;
    }

    if (thid == 0) data[n - 1] = 0;

    for (int d = 1; d < n; d *= 2) {
      offset >>= 1;
      __syncthreads();
      for (int i = thid; i < d; i += blockDim.x) {
        int ai = offset * (2 * i + 1) - 1;
        int bi = offset * (2 * i + 2) - 1;
        unsigned int t = data[ai];
        data[ai] = data[bi];
        data[bi] += t;
      }
    }

    __syncthreads();
  }

 private:
  CudaKernelHelpers() {}
  ~CudaKernelHelpers() {}
};
}