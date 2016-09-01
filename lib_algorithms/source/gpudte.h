#pragma once
#include "../../source_shared/include/global_defines.h"

#include <cuda_runtime_api.h>
#include <cassert>
#include "../include/ml_algorithm.h"

namespace lib_algorithms {
template <typename T>
class GpuDte : public MlAlgorithm<T> {
 public:
  GpuDte();
  virtual ~GpuDte();

 protected:
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

  static __device__ T get_data_point(int attribute, int instance,
                                     int nr_instances, T* dataset);

  static __device__ inline T lnFunc(T num);

  static __device__ T entropy_conditioned_on_rows(T* matrix, int att_type,
                                                  int nr_targets);

  static __device__ T entropy_over_columns(T* matrix, int att_type,
                                           int nr_targets);

  static __device__ T evaluate_nominal_attribute(
      gpuDTE_NodeHeader_Train& node, gpuDTE_TmpNodeValues& tmp_node,
      T* curr_dist, int att_type, int nr_targets, bool tick_tock,
      int** indices_buffer, T* targer_data, int nr_instances, T* dataset);

  static __device__ void gpudte_perform_split(
      gpuDTE_StaticInfo& static_info, gpuDTE_DatasetInfo& dataset_info,
      gpuDTE_IterationInfo& iteration_info, T** probability_buffers,
      T* probability_tmp, T* dataset, int* attribute_types, int* node_counts,
      int** indices_buffers, int* node_cursors,
      gpuDTE_NodeHeader_Train** node_buffers);

  static __device__ void gpudte_predict(int tid, int nr_instances,
                                        int data_type, int nr_targets,
                                        gpuDTE_NodeHeader_Classify* node_buffer,
                                        T* dataset, T* probability_buffer,
                                        T* predictions, int* attribute_types);

  static __device__ float AtomicAdd(float* address, float value);
  static __device__ int AtomicAdd(int* address, int value);
  static __device__ unsigned int AtomicAdd(unsigned int* address,
                                           unsigned int value);
  static __device__ double AtomicAdd(double* address, double value);
};
}