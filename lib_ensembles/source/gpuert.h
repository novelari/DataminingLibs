#pragma once
#include "gpudte.h"

#include "../../lib_algorithms/include/ml_algorithm.h"
#include "../../lib_ensembles/source/gpudte_algorithm_shared.h"

namespace lib_ensembles {
template <typename T>
class GpuErt : public GpuDte<T> {
 public:
  GpuErt() {}
  ~GpuErt() {}

  void CallCudaKernel(int blocks, int block_size,
                      GpuDteAlgorithmShared::GpuParams<T> &params,
                      GpuDteAlgorithmShared::GpuDteKernelId id) override;

  __device__ void gpuert_setup_kernel(
      GpuDteAlgorithmShared::GpuParams<T> *params, unsigned long long seed);
  __device__ void gpuert_initialize_tree_batch(
      GpuDteAlgorithmShared::GpuParams<T> *params);
  __device__ void gpuert_find_split(
      GpuDteAlgorithmShared::GpuParams<T> *params);
  __device__ void gpuert_perform_split(
      GpuDteAlgorithmShared::GpuParams<T> *params);
  __device__ void gpuert_predict(GpuDteAlgorithmShared::GpuParams<T> *params);

  __device__ T eval_numeric_attribute(
      GpuDteAlgorithmShared::GpuParams<T> *params,
      GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T> &node,
      GpuDteAlgorithmShared::gpuDTE_TmpNodeValues<T> &tmp_node, T *curr_dist,
      int att_type);

  __device__ T varianceCalculation(
      GpuDteAlgorithmShared::GpuParams<T> *params,
      GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T> &node,
      GpuDteAlgorithmShared::gpuDTE_TmpNodeValues<T> &tmp_node, T *curr_dist);
};
}