#pragma once
#include <curand_kernel.h>
#include "gpudte.h"

namespace lib_algorithms {
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
class GpuRf : public GpuDte<T> {
 public:
  GpuRf();
  ~GpuRf();

  sp<lib_models::MlModel> Fit(sp<lib_data::MlDataFrame<T>> data,
                              sp<MlAlgorithmParams> params) override;
  sp<lib_data::MlResultData<T>> Predict(sp<lib_data::MlDataFrame<T>> data,
                                        sp<lib_models::MlModel> model,
                                        sp<MlAlgorithmParams> params) override;

  sp<lib_models::MlModel> AggregateModels(
      col_array<sp<lib_models::MlModel>> models) override;
  sp<lib_data::MlResultData<T>> AggregateResults(
	  col_array<sp<lib_data::MlResultData<T>>> results) override;
  col_array<sp<MlAlgorithmParams>> SplitParameterPack(
      sp<MlAlgorithmParams> params, const int parts) override;

  struct GpuParams {
    void init();
    void finalize();

    gpuDTE_StaticInfo *static_info;
    gpuDTE_DatasetInfo *dataset_info;
    gpuDTE_IterationInfo *iteration_info;

    // Global buffers
    int *node_cursors;
    T *probability_buffers[2];
    gpuDTE_NodeHeader_Train *node_buffers[3];
    gpuDTE_NodeHeader_Classify *node_buffers_classify;

    gpuDTE_TmpNodeValues *node_tmp_buffer;
    T *probability_tmp_buffer;
    int *indices_buffer[2];
    bool *indices_inbag;
    int *target_starts;

    curandStateMRG32k3a *random_states;

    // Dataset buffers
    T *dataset;
    T *target_data;
    int *attribute_type;

    // Prediction buffer
    T *predictions;
    int *oobCounts;
    T *mse;
  };

 private:
  void AllocateFit(sp<MlAlgorithmParams> params, GpuParams *gpu_params,
                   sp<lib_data::MlDataFrame<T>> data);
  void AllocatePredict(sp<MlAlgorithmParams> params, GpuParams *gpu_params,
                       sp<lib_data::MlDataFrame<T>> data,
                       sp<lib_models::MlModel> model);

  void FreeParams(GpuParams **dev_ptr);
  GpuParams *CreateParams();

  void SwapBuffers(int *lhs, int *rhs);
  void StreamToCache(int src_id, int layer_id,
                     col_array<col_array<gpuDTE_NodeHeader_Train>> &node_cache,
                     col_array<int> &buffer_counts,
                     gpuDTE_NodeHeader_Train *node_headers);
  void StreamFromCache(
      int dst_id, int layer_id,
      col_array<col_array<gpuDTE_NodeHeader_Train>> &node_cache,
      col_array<int> &buffer_counts, gpuDTE_NodeHeader_Train *node_headers);

 public:
  static __device__ void gpurf_setup_kernel(GpuParams *params);
  static __device__ void gpurf_initialize_tree_batch(GpuParams *params);
  static __device__ void gpurf_find_split(GpuParams *params);
  static __device__ void gpurf_perform_split(GpuParams *params);
  static __device__ void gpurf_predict(GpuParams *params);
  static __device__ void gpurf_oob_estimate(GpuParams *params);
  static __device__ void gpurf_feature_importance(GpuParams *params);

  static __device__ void radix_sort_on_attribute(
      GpuParams *params, gpuDTE_NodeHeader_Train &node,
      gpuDTE_TmpNodeValues &tmp_node, unsigned int s_histograms[1024],
      unsigned int s_offsets[256]);

  static __device__ T eval_numeric_attribute(GpuParams *params,
                                             gpuDTE_NodeHeader_Train &node,
                                             gpuDTE_TmpNodeValues &tmp_node,
                                             T *curr_dist, int att_type,
                                             unsigned int *s_histograms,
                                             unsigned int *s_offsets);

  static __device__ T variance_calculation(GpuParams *params,
                                           gpuDTE_NodeHeader_Train &node,
                                           gpuDTE_TmpNodeValues &tmp_node,
                                           T *curr_dist, T *s_histograms);
};
}