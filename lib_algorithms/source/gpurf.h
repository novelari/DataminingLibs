#pragma once
#include "../../source_shared/include/global_defines.h"

#include <cuda_runtime_api.h>
#include <cassert>
#include <curand_kernel.h>
#include "../include/ml_algorithm.h"

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
class GpuRf : public MlAlgorithm<T> {
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

#define block_size_ 64
#define max_blocks_ 1024
#define max_nominal_ 2
#define type_classification_ 0
#define type_regression_ 1
#define node_id_ 2
#define new_nodes_ 1
#define work_cursor_ 0
#define flt_max 3.402823466e+38F

  struct gpuDTE_StaticInfo {
    int total_trees;
    int loaded_trees;
    int node_buffer_size;
    int max_node_size;
    int max_node_depth;
    int nr_features;
    int max_class_count;
    bool balanced_sampling;
  };

  struct gpuDTE_IterationInfo {
    int threads_launched;
    int read_buffer_id;
    int write_buffer_id;
    int prob_buffer_id;
    int depth;
    int tree_offset;
    int node_offset;
    bool first_part;
    bool last_part;
    bool tick_tock;
  };

  struct gpuDTE_DatasetInfo {
    int data_type;
    int nr_instances;
    int nr_attributes;
    int nr_target_values;
  };

  struct gpuDTE_TmpNodeValues {
    T tmp_split;
    T tmp_score;
    int tmp_attribute;
  };

  struct gpuDTE_NodeHeader_Train {
    int trackinid;
    int parent_id;

    int node_index_start;
    int node_index_count;

    int attribute;
    T split_point;
  };

  struct gpuDTE_NodeHeader_Classify {
    int child_start;
    int child_count;
    int probability_start;

    int attribute;
    T split_point;
  };

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
  __device__ T get_data_point(int attribute, int instance, int nr_instances,
                              T *dataset);

  __device__ inline T lnFunc(T num);

  __device__ T entropy_conditioned_on_rows(T *matrix, int att_type,
                                           int nr_targets);

  __device__ T entropy_over_columns(T *matrix, int att_type, int nr_targets);

  __device__ T evaluate_nominal_attribute(gpuDTE_NodeHeader_Train &node,
                                          gpuDTE_TmpNodeValues &tmp_node,
                                          T *curr_dist, int att_type,
                                          int nr_targets, bool tick_tock,
                                          int **indices_buffer, T *targer_data,
                                          int nr_instances, T *dataset);

  __device__ void gpudte_perform_split(gpuDTE_StaticInfo &static_info,
                                       gpuDTE_DatasetInfo &dataset_info,
                                       gpuDTE_IterationInfo &iteration_info,
                                       T **probability_buffers,
                                       T *probability_tmp, T *dataset,
                                       int *attribute_types, int *node_counts,
                                       int **indices_buffers, int *node_cursors,
                                       gpuDTE_NodeHeader_Train **node_buffers);

  __device__ void gpudte_predict(int tid, int nr_instances, int data_type,
                                 int nr_targets,
                                 gpuDTE_NodeHeader_Classify *node_buffer,
                                 T *dataset, T *probability_buffer,
                                 T *predictions, int *attribute_types);

  __device__ float AtomicAdd(float *address, float value);
  __device__ int AtomicAdd(int *address, int value);
  __device__ unsigned int AtomicAdd(unsigned int *address, unsigned int value);
  __device__ double AtomicAdd(double *address, double value);

  __device__ void gpurf_setup_kernel(GpuParams *params);
  __device__ void gpurf_initialize_tree_batch(GpuParams *params);
  __device__ void gpurf_find_split(GpuParams *params);
  __device__ void gpurf_perform_split(GpuParams *params);
  __device__ void gpurf_predict(GpuParams *params);
  __device__ void gpurf_oob_estimate(GpuParams *params);
  __device__ void gpurf_feature_importance(GpuParams *params);

  __device__ void radix_sort_on_attribute(GpuParams *params,
                                          gpuDTE_NodeHeader_Train &node,
                                          gpuDTE_TmpNodeValues &tmp_node,
                                          unsigned int s_histograms[1024],
                                          unsigned int s_offsets[256]);

  __device__ T eval_numeric_attribute(GpuParams *params,
                                      gpuDTE_NodeHeader_Train &node,
                                      gpuDTE_TmpNodeValues &tmp_node,
                                      T *curr_dist, int att_type,
                                      unsigned int *s_histograms,
                                      unsigned int *s_offsets);

  __device__ T variance_calculation(GpuParams *params,
                                    gpuDTE_NodeHeader_Train &node,
                                    gpuDTE_TmpNodeValues &tmp_node,
                                    T *curr_dist, T *s_histograms);
};
}