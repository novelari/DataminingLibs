#pragma once
#include "gpudte.h"

#include "../../lib_algorithms/include/ml_algorithm.h"
#include "../../lib_ensembles/source/gpudte_algorithm_shared.h"

namespace lib_ensembles {
template <typename T>
class GpuRf : public GpuDte<T> {
 public:
  GpuRf() {}
  ~GpuRf() {}

  void CallCudaKernel(int blocks, int block_size,
                      GpuDteAlgorithmShared::GpuParams<T> &params,
                      GpuDteAlgorithmShared::GpuDteKernelId id) override;

 public:
  __device__ void gpurf_setup_kernel(
      GpuDteAlgorithmShared::GpuParams<T> *params);
  __device__ void gpurf_initialize_tree_batch(
      GpuDteAlgorithmShared::GpuParams<T> *params);
  __device__ void gpurf_find_split(GpuDteAlgorithmShared::GpuParams<T> *params);
  __device__ void gpurf_perform_split(
      GpuDteAlgorithmShared::GpuParams<T> *params);
  __device__ void gpurf_predict(GpuDteAlgorithmShared::GpuParams<T> *params);
  __device__ void gpurf_oob_estimate(
      GpuDteAlgorithmShared::GpuParams<T> *params);
  __device__ void gpurf_feature_importance(
      GpuDteAlgorithmShared::GpuParams<T> *params);

  __device__ void radix_sort_on_attribute(
      GpuDteAlgorithmShared::GpuParams<T> *params,
      GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T> &node,
      GpuDteAlgorithmShared::gpuDTE_TmpNodeValues<T> &tmp_node,
      unsigned int s_histograms[1024], unsigned int s_offsets[256]);

  __device__ T eval_numeric_attribute(
      GpuDteAlgorithmShared::GpuParams<T> *params,
      GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T> &node,
      GpuDteAlgorithmShared::gpuDTE_TmpNodeValues<T> &tmp_node, T *curr_dist,
      int att_type, unsigned int *s_histograms, unsigned int *s_offsets);

  __device__ T
  variance_calculation(GpuDteAlgorithmShared::GpuParams<T> *params,
                       GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T> &node,
                       GpuDteAlgorithmShared::gpuDTE_TmpNodeValues<T> &tmp_node,
                       T *curr_dist, T *s_histograms);
};
}