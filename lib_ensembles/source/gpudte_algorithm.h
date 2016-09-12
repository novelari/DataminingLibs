#pragma once

#include "../../lib_algorithms/include/lib_algorithms.h"
#include "gpudte_algorithm_shared.h"

namespace lib_ensembles {
template <typename T>
class GpuDte;
template <typename T>
class GpuDteAlgorithm : public lib_algorithms::MlAlgorithm<T> {
 public:
  GpuDteAlgorithm(sp<GpuDte<T>> func) : gpu_functions_(func) {}
  virtual ~GpuDteAlgorithm() {}

  sp<lib_models::MlModel> Fit(
      sp<lib_data::MlDataFrame<T>> data,
      sp<lib_algorithms::MlAlgorithmParams> params) override;
  sp<lib_data::MlResultData<T>> Predict(
      sp<lib_data::MlDataFrame<T>> data, sp<lib_models::MlModel> model,
      sp<lib_algorithms::MlAlgorithmParams> params) override;

  sp<lib_models::MlModel> AggregateModels(
      col_array<sp<lib_models::MlModel>> models) override;
  col_array<sp<lib_models::MlModel>> SplitModel(sp<lib_models::MlModel> model,
                                                const int parts) override;
  sp<lib_data::MlResultData<T>> AggregateResults(
      col_array<sp<lib_data::MlResultData<T>>> results) override;
  col_array<sp<lib_algorithms::MlAlgorithmParams>> SplitParameterPack(
      sp<lib_algorithms::MlAlgorithmParams> params, const int parts) override;

  sp<GpuDte<T>> gpu_functions_;

 private:
  class HostAllocFit {
   public:
    HostAllocFit(sp<lib_gpu::GpuDevice> dev, size_t targets);
    ~HostAllocFit();

    T *probability_cpy;
    GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T> *node_cpy;
    int *node_cursor_cpy;

    sp<lib_gpu::GpuDevice> dev_;
  };
  class HostAllocPredict {
   public:
    HostAllocPredict(sp<lib_gpu::GpuDevice> dev, size_t samples,
                     size_t targets);
    ~HostAllocPredict();

    T *predictions_cpy;

    sp<lib_gpu::GpuDevice> dev_;
  };

  void AllocateFit(sp<lib_gpu::GpuDevice> dev,
                   sp<lib_algorithms::MlAlgorithmParams> params,
                   GpuDteAlgorithmShared::GpuParams<T> *gpu_params,
                   sp<lib_data::MlDataFrame<T>> data,
                   GpuDteAlgorithmShared::gpuDTE_StaticInfo &static_info,
                   GpuDteAlgorithmShared::gpuDTE_DatasetInfo &dataset_info,
                   GpuDteAlgorithmShared::gpuDTE_IterationInfo &iteration_info);
  void AllocatePredict(
      sp<lib_gpu::GpuDevice> dev, sp<lib_algorithms::MlAlgorithmParams> params,
      GpuDteAlgorithmShared::GpuParams<T> *gpu_params,
      sp<lib_data::MlDataFrame<T>> data, sp<lib_models::MlModel> model,
      GpuDteAlgorithmShared::gpuDTE_StaticInfo &static_info,
      GpuDteAlgorithmShared::gpuDTE_DatasetInfo &dataset_info,
      GpuDteAlgorithmShared::gpuDTE_IterationInfo &iteration_info);

  void SwapBuffers(int *lhs, int *rhs);
  void StreamToCache(
      sp<lib_gpu::GpuDevice> dev, HostAllocFit &host_alloc, int src_id,
      int layer_id,
      col_array<col_array<GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T>>>
          &node_cache,
      col_array<int> &buffer_counts,
      GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T> *node_headers);
  void StreamFromCache(
      sp<lib_gpu::GpuDevice> dev, HostAllocFit &host_alloc, int dst_id,
      int layer_id,
      col_array<col_array<GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T>>>
          &node_cache,
      col_array<int> &buffer_counts,
      GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T> *node_headers);
};
}