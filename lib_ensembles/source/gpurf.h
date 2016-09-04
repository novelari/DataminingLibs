#pragma once
#include "gpudte.h"

#include "../../lib_algorithms/include/ml_algorithm.h"

namespace lib_ensembles {
class GpuRfStatic {
 public:
  enum GpuRfKernelId {
    kSetupKernel = 0,
    kInitTreeBatch,
    kFindSplit,
    kPerformSplit,
    kPredict,
    kOobEstimate,
    kFeatureImp,
  };

 private:
  GpuRfStatic() = delete;
  ~GpuRfStatic() = delete;
};

template <typename T>
class GpuRf : public lib_algorithms::MlAlgorithm<T> {
 public:
  GpuRf();
  ~GpuRf();

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

 private:
  void AllocateFit(sp<lib_algorithms::MlAlgorithmParams> params,
                   GpuDte::GpuParams<T> *gpu_params,
                   sp<lib_data::MlDataFrame<T>> data);
  void AllocatePredict(sp<lib_algorithms::MlAlgorithmParams> params,
                       GpuDte::GpuParams<T> *gpu_params,
                       sp<lib_data::MlDataFrame<T>> data,
                       sp<lib_models::MlModel> model);

  void FreeParams(GpuDte::GpuParams<T> **dev_ptr);
  GpuDte::GpuParams<T> *CreateParams();

  void SwapBuffers(int *lhs, int *rhs);
  void StreamToCache(
      int src_id, int layer_id,
      col_array<col_array<GpuDte::gpuDTE_NodeHeader_Train<T>>> &node_cache,
      col_array<int> &buffer_counts,
      GpuDte::gpuDTE_NodeHeader_Train<T> *node_headers);
  void StreamFromCache(
      int dst_id, int layer_id,
      col_array<col_array<GpuDte::gpuDTE_NodeHeader_Train<T>>> &node_cache,
      col_array<int> &buffer_counts,
      GpuDte::gpuDTE_NodeHeader_Train<T> *node_headers);

  GpuDte::gpuDTE_StaticInfo static_info_;
  GpuDte::gpuDTE_DatasetInfo dataset_info_;
  GpuDte::gpuDTE_IterationInfo iteration_info_;

 public:
  __device__ void gpurf_setup_kernel(GpuDte::GpuParams<T> *params);
  __device__ void gpurf_initialize_tree_batch(GpuDte::GpuParams<T> *params);
  __device__ void gpurf_find_split(GpuDte::GpuParams<T> *params);
  __device__ void gpurf_perform_split(GpuDte::GpuParams<T> *params);
  __device__ void gpurf_predict(GpuDte::GpuParams<T> *params);
  __device__ void gpurf_oob_estimate(GpuDte::GpuParams<T> *params);
  __device__ void gpurf_feature_importance(GpuDte::GpuParams<T> *params);

  __device__ void radix_sort_on_attribute(
      GpuDte::GpuParams<T> *params, GpuDte::gpuDTE_NodeHeader_Train<T> &node,
      GpuDte::gpuDTE_TmpNodeValues<T> &tmp_node,
      unsigned int s_histograms[1024], unsigned int s_offsets[256]);

  __device__ T eval_numeric_attribute(GpuDte::GpuParams<T> *params,
                                      GpuDte::gpuDTE_NodeHeader_Train<T> &node,
                                      GpuDte::gpuDTE_TmpNodeValues<T> &tmp_node,
                                      T *curr_dist, int att_type,
                                      unsigned int *s_histograms,
                                      unsigned int *s_offsets);

  __device__ T variance_calculation(GpuDte::GpuParams<T> *params,
                                    GpuDte::gpuDTE_NodeHeader_Train<T> &node,
                                    GpuDte::gpuDTE_TmpNodeValues<T> &tmp_node,
                                    T *curr_dist, T *s_histograms);
};
}