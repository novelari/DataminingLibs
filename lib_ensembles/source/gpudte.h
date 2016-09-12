#pragma once
#include "../../source_shared/include/global_defines.h"

#include "../../lib_ensembles/source/gpudte_algorithm.h"
#include "../../lib_ensembles/source/gpudte_algorithm_shared.h"

#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <cassert>

namespace lib_ensembles {
template <typename T>
class GpuDte {
 public:
  GpuDte() {}
  virtual ~GpuDte() {}

  virtual void CallCudaKernel(int blocks, int block_size,
                              GpuDteAlgorithmShared::GpuParams<T> &params,
                              GpuDteAlgorithmShared::GpuDteKernelId id) = 0;
  
  void CopyIterationInfo(GpuDteAlgorithmShared::gpuDTE_IterationInfo &info);
  void CopyDataStaticInfo(GpuDteAlgorithmShared::gpuDTE_DatasetInfo &data,
                          GpuDteAlgorithmShared::gpuDTE_StaticInfo &info);

  __device__ void GetConstPointers(GpuDteAlgorithmShared::gpuDTE_IterationInfo **iter,
                        GpuDteAlgorithmShared::gpuDTE_DatasetInfo **data,
                        GpuDteAlgorithmShared::gpuDTE_StaticInfo **stat);

  __device__ T get_data_point(int attribute, int instance, int nr_instances,
                              T *dataset);

  __device__ inline T lnFunc(T num);

  __device__ T entropy_conditioned_on_rows(T *matrix, int att_type,
                                           int nr_targets);

  __device__ T entropy_over_columns(T *matrix, int att_type, int nr_targets);

  __device__ T evaluate_nominal_attribute(
      GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T> &node,
      GpuDteAlgorithmShared::gpuDTE_TmpNodeValues<T> &tmp_node, T *curr_dist,
      int att_type, int nr_targets, bool tick_tock, int **indices_buffer,
      T *targer_data, int nr_instances, T *dataset);

  __device__ void gpudte_perform_split(
      GpuDteAlgorithmShared::gpuDTE_StaticInfo &static_info,
      GpuDteAlgorithmShared::gpuDTE_DatasetInfo &dataset_info,
      GpuDteAlgorithmShared::gpuDTE_IterationInfo &iteration_info,
      T **probability_buffers, T *probability_tmp, T *dataset,
      int *attribute_types, int *node_counts, int **indices_buffers,
      int *node_cursors,
      GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T> **node_buffers);

  __device__ void gpudte_predict(
      int tid, int nr_instances, int data_type, int nr_targets,
      GpuDteAlgorithmShared::gpuDTE_NodeHeader_Classify<T> *node_buffer,
      T *dataset, T *probability_buffer, T *predictions, int *attribute_types);

  __device__ float AtomicAdd(float *address, float value);
  __device__ int AtomicAdd(int *address, int value);
  __device__ unsigned int AtomicAdd(unsigned int *address, unsigned int value);
  __device__ double AtomicAdd(double *address, double value);
};
}