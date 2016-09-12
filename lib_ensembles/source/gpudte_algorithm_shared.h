#pragma once
#include "../../source_shared/include/global_defines.h"

#include "../../lib_gpu/include/gpu_device.h"

struct curandStateMRG32k3a;

namespace lib_ensembles {
#define block_size_ 64
#define max_blocks_ 1024
#define max_tree_batch_ 100
#define max_nominal_ 2
#define type_classification_ 0
#define type_regression_ 1
#define node_id_ 2
#define new_nodes_ 1
#define work_cursor_ 0
#define flt_max 3.402823466e+38F

class GpuDteAlgorithmShared {
 public:
  enum GpuDteKernelId {
    kSetupKernel = 0,
    kInitTreeBatch,
    kFindSplit,
    kPerformSplit,
    kPredict,
    kOobEstimate,
    kFeatureImp,
  };

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

  template <typename T>
  struct gpuDTE_TmpNodeValues {
    T tmp_split;
    T tmp_score;
    int tmp_attribute;
  };

  template <typename T>
  struct gpuDTE_NodeHeader_Train {
    int tracking_id;
    int parent_id;

    int node_index_start;
    int node_index_count;

    int attribute;
    T split_point;
  };

  template <typename T>
  struct gpuDTE_NodeHeader_Classify {
    int child_start;
    int child_count;
    int probability_start;

    int attribute;
    T split_point;
  };

  template <typename T>
  class GpuParams {
   public:
    GpuParams();
    ~GpuParams() {}

    void finalize(sp<lib_gpu::GpuDevice> dev);

    gpuDTE_StaticInfo *static_info;
    gpuDTE_DatasetInfo *dataset_info;
    gpuDTE_IterationInfo *iteration_info;

    // Global buffers
    int *node_cursors;
    T *probability_buffers[2];
    gpuDTE_NodeHeader_Train<T> *node_buffers[3];
    gpuDTE_NodeHeader_Classify<T> *node_buffers_classify;

    gpuDTE_TmpNodeValues<T> *node_tmp_buffer;
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
};
}