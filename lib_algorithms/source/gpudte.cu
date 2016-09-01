#define DLLExport
#define TestExport

#include "gpudte.h"

namespace lib_algorithms {
template <typename T>
GpuDte<T>::GpuDte() {}
template <typename T>
GpuDte<T>::~GpuDte() {}
template <typename T>
__device__ T GpuDte<T>::get_data_point(int attribute, int instance,
                                       int nr_instances, T* dataset) {
  return dataset[attribute * nr_instances + instance];
}

template <typename T>
__device__ T GpuDte<T>::lnFunc(T num) {
  if (num <= 1e-6) return 0;
  return num * log(num);
}

template <typename T>
__device__ T GpuDte<T>::entropy_conditioned_on_rows(T* matrix, int att_type,
                                                    int nr_targets) {
  int x = (att_type > max_nominal_) ? 2 : 0;
  int y = nr_targets;

  T returnValue = 0, sumForRow, total = 0;
  for (int i = 0; i < x; i++) {
    sumForRow = 0;
    for (int ii = 0; ii < y; ii++) {
      returnValue = returnValue + GpuDte::lnFunc(matrix[y * i + ii]);
      sumForRow += matrix[y * i + ii];
    }
    returnValue = returnValue - GpuDte::lnFunc(sumForRow);
    total += sumForRow;
  }
  if (total < 1.0e-6) {
    return 0;
  }
  return -returnValue / (total * log(T(2.0)));
}

template <typename T>
__device__ T GpuDte<T>::entropy_over_columns(T* matrix, int att_type,
                                             int nr_targets) {
  int x = (att_type > max_nominal_) ? 2 : att_type;
  int y = nr_targets;

  T returnValue = 0, sumForColumn, total = 0;
  for (int ii = 0; ii < y; ii++) {
    sumForColumn = 0;
    for (int i = 0; i < x; i++) {
      sumForColumn += matrix[y * i + ii];
    }
    returnValue = returnValue - GpuDte::lnFunc(sumForColumn);
    total += sumForColumn;
  }
  if (total < 1.0e-6) {
    return 0;
  }
  return (returnValue + GpuDte::lnFunc(total)) / (total * log(T(2.0)));
}

template <typename T>
__device__ T GpuDte<T>::evaluate_nominal_attribute(
    gpuDTE_NodeHeader_Train& node, gpuDTE_TmpNodeValues& tmp_node, T* curr_dist,
    int att_type, int nr_targets, bool tick_tock, int** indices_buffer,
    T* targer_data, int nr_instances, T* dataset) {
  int numInds = node.node_index_count;
  int nodeIndStart = node.node_index_start;
  T weight = 1;
  int inst;
  T val;

  // Split on median value
  for (int i = threadIdx.x; i < numInds; i += blockDim.x) {
    inst = indices_buffer[tick_tock ? 0 : 1][nodeIndStart + i];

    val = GpuDte::get_data_point(tmp_node.tmp_attribute, inst, nr_instances,
                                 dataset);

    if (val != -FLT_MAX)
      AtomicAdd(&curr_dist[nr_targets * int(val) + int(targer_data[inst])],
                weight);
    else
      AtomicAdd(&curr_dist[int(targer_data[inst])], weight);
  }

  __syncthreads();

  T response;
  if (threadIdx.x == 0) {
    T prior = entropy_over_columns(curr_dist, att_type, nr_targets);
    T posterior = entropy_conditioned_on_rows(curr_dist, att_type, nr_targets);
    response = prior - posterior;
  }

  return response;
}

template <typename T>
__device__ void GpuDte<T>::gpudte_perform_split(
    gpuDTE_StaticInfo& static_info, gpuDTE_DatasetInfo& dataset_info,
    gpuDTE_IterationInfo& iteration_info, T** probability_buffers,
    T* probability_tmp, T* dataset, int* attribute_types, int* node_counts,
    int** indices_buffers, int* node_cursors,
    gpuDTE_NodeHeader_Train** node_buffers) {
  __shared__ gpuDTE_NodeHeader_Train s_tree_node;
  __shared__ int s_attribute_type;

  if (threadIdx.x == 0) {
    s_tree_node = node_buffers[iteration_info.read_buffer_id]
                              [blockIdx.x + iteration_info.node_offset];
    s_attribute_type = attribute_types[s_tree_node.attribute];
    s_attribute_type = s_attribute_type >= max_nominal_ ? 2 : s_attribute_type;

    switch (dataset_info.data_type) {
      case type_classification_: {
        for (int ii = 0; ii < s_attribute_type; ++ii) {
          node_counts[ii] = 0;
          for (int i = 0; i < dataset_info.nr_target_values; ++i)
            node_counts[ii] +=
                probability_tmp[blockIdx.x * dataset_info.nr_target_values *
                                    max_nominal_ +
                                ii * dataset_info.nr_target_values + i];
        }

        int add = node_counts[0];
        int tmp;
        node_counts[0] = 0;
        for (int i = 1; i < s_attribute_type; i++) {
          tmp = node_counts[i];
          node_counts[i] = node_counts[i - 1] + add;
          add = tmp;
        }
        break;
      }
      case type_regression_: {
        node_counts[0] = 0;
        node_counts[1] = s_tree_node.node_index_count - 1;
        break;
      }
    }
  }

  __syncthreads();

  // Split node indices into backbuffer
  T data_point;
  int instance_ind, ind_position;
  for (int i = threadIdx.x; i < s_tree_node.node_index_count; i += blockDim.x) {
    instance_ind = indices_buffers[iteration_info.tick_tock ? 0 : 1]
                                  [s_tree_node.node_index_start + i];
    data_point = GpuDte::get_data_point(s_tree_node.attribute, instance_ind,
                                        dataset_info.nr_instances, dataset);

    switch (dataset_info.data_type) {
      case type_classification_:
        if (data_point < s_tree_node.split_point)
          ind_position = AtomicAdd(&node_counts[0], 1);
        else
          ind_position = AtomicAdd(&node_counts[1], 1);
        break;
      case type_regression_:
        if (data_point < s_tree_node.split_point)
          ind_position = AtomicAdd(&node_counts[0], 1);
        else
          ind_position = AtomicAdd(&node_counts[1], 1);
        break;
    }

    ind_position += s_tree_node.node_index_start;
    indices_buffers[iteration_info.tick_tock ? 1 : 0][ind_position] =
        instance_ind;
  }

  __syncthreads();

  // Create new nodes
  if (threadIdx.x == 0) {
    if (dataset_info.data_type == type_regression_)
      node_counts[1] = s_tree_node.node_index_count;

    gpuDTE_NodeHeader_Train child;
    child.parent_id = s_tree_node.trackinid;

    bool skip_split = false;
    int index_offset = 0;
    int child_count = 2;
    for (int i = 0; i < child_count; ++i) {
      child.node_index_count = node_counts[i] - index_offset;
      index_offset += node_counts[i];
      if (child.node_index_count <= 0) skip_split = true;
    }

    if (skip_split ||
        s_tree_node.node_index_count <= static_info.max_node_size ||
        static_info.max_node_depth <= iteration_info.depth) {
      s_tree_node.attribute = -1;
      s_tree_node.split_point = -1;
    } else {
      int child_start = AtomicAdd(&node_cursors[work_cursor_], child_count);
      AtomicAdd(&node_cursors[new_nodes_], child_count);

      index_offset = 0;
      child.node_index_start = s_tree_node.node_index_start;
      for (int i = 0; i < child_count; ++i) {
        child.trackinid = AtomicAdd(&node_cursors[node_id_], 1);
        child.node_index_count = node_counts[i] - index_offset;
        index_offset += node_counts[i];

        node_buffers[iteration_info.write_buffer_id][child_start + i] = child;
        child.node_index_start += child.node_index_count;
      }
    }

    // Set probabilities for original node
    for (int i = 0; i < dataset_info.nr_target_values; ++i)
      probability_buffers[iteration_info.prob_buffer_id]
                         [(blockIdx.x + iteration_info.node_offset) *
                              dataset_info.nr_target_values +
                          i] = 0;

    for (int i = 0; i < child_count; ++i) {
      switch (dataset_info.data_type) {
        case type_classification_:
          for (int ii = 0; ii < dataset_info.nr_target_values; ++ii) {
            probability_buffers[iteration_info.prob_buffer_id]
                               [(blockIdx.x + iteration_info.node_offset) *
                                    dataset_info.nr_target_values +
                                ii] +=
                probability_tmp[blockIdx.x * dataset_info.nr_target_values *
                                    max_nominal_ +
                                ii + dataset_info.nr_target_values * i];
          }
          break;
        case type_regression_:
          probability_buffers[iteration_info.prob_buffer_id]
                             [(blockIdx.x + iteration_info.node_offset) *
                              dataset_info.nr_target_values] +=
              probability_tmp[blockIdx.x * dataset_info.nr_target_values *
                                  max_nominal_ +
                              i] /
              T(child_count);
          break;
      }
    }

    node_buffers[iteration_info.read_buffer_id]
                [blockIdx.x + iteration_info.node_offset] = s_tree_node;
  }
}

template <typename T>
__device__ void GpuDte<T>::gpudte_predict(
    int tid, int nr_instances, int data_type, int nr_targets,
    gpuDTE_NodeHeader_Classify* node_buffer, T* dataset, T* probability_buffer,
    T* predictions, int* attribute_types) {
  // Access pattern
  // instances*trees: 5*5 = 25
  // tid:			0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
  // instances:		0 1 2 3 4 0 1 2 3 4 0  1  2  3  4  0  1  2  3
  // trees:			0 0 0 0 0 1 1 1 1 1 2  2  2  2  2  3  3  3  3

  int instanceId = tid % nr_instances;
  int treeId = T(tid) / T(nr_instances);
  T dataPoint;

  gpuDTE_NodeHeader_Classify tree_node = node_buffer[treeId];
  while (tree_node.child_count > 0) {
    dataPoint = GpuDte::get_data_point(tree_node.attribute, instanceId,
                                       nr_instances, dataset);
    if (dataPoint != -FLT_MAX) {
      tree_node = (dataPoint < tree_node.split_point)
                      ? node_buffer[tree_node.child_start]
                      : node_buffer[tree_node.child_start + 1];
    } else
      tree_node = node_buffer[tree_node.child_start];
  }

  switch (data_type) {
    case type_classification_: {
      int classSelector = 0;
      T max_prob = 0;
      for (int i = 0; i < nr_targets; ++i) {
        if (probability_buffer[tree_node.probability_start + i] > max_prob) {
          max_prob = probability_buffer[tree_node.probability_start + i];
          classSelector = i;
        }
      }

      T add = 1;
      AtomicAdd(&predictions[instanceId * nr_targets + classSelector], add);
      break;
    }
    case type_regression_:
      T mean = probability_buffer[tree_node.probability_start];
      AtomicAdd(&predictions[instanceId], mean);
      break;
  }
}

template <typename T>
__device__ float GpuDte<T>::AtomicAdd(float* address, float value) {
  return atomicAdd(address, value);
}

template <typename T>
__device__ int GpuDte<T>::AtomicAdd(int* address, int value) {
  return atomicAdd(address, value);
}

template <typename T>
__device__ unsigned int GpuDte<T>::AtomicAdd(unsigned int* address,
                                             unsigned int value) {
  return atomicAdd(address, value);
}

template <typename T>
__device__ double GpuDte<T>::AtomicAdd(double* address, double value) {
  unsigned long long oldval, newval, readback;

  oldval = __double_as_longlong(*address);
  newval = __double_as_longlong(__longlong_as_double(oldval) + value);

  while ((readback = atomicCAS((unsigned long long*)address, oldval, newval)) !=
         oldval) {
    oldval = readback;
    newval = __double_as_longlong(__longlong_as_double(oldval) + value);
  }

  return __longlong_as_double(oldval);
}

template GpuDte<float>::GpuDte();
template GpuDte<double>::GpuDte();

template __device__ float GpuDte<float>::get_data_point(int attribute,
                                                        int instance,
                                                        int nr_instances,
                                                        float* dataset);
template __device__ double GpuDte<double>::get_data_point(int attribute,
                                                          int instance,
                                                          int nr_instances,
                                                          double* dataset);

template __device__ float GpuDte<float>::lnFunc(float num);
template __device__ double GpuDte<double>::lnFunc(double num);

template __device__ float GpuDte<float>::entropy_conditioned_on_rows(
    float* matrix, int att_type, int nr_targets);
template __device__ double GpuDte<double>::entropy_conditioned_on_rows(
    double* matrix, int att_type, int nr_targets);

template __device__ float GpuDte<float>::entropy_over_columns(float* matrix,
                                                              int att_type,
                                                              int nr_targets);
template __device__ double GpuDte<double>::entropy_over_columns(double* matrix,
                                                                int att_type,
                                                                int nr_targets);

template __device__ float GpuDte<float>::evaluate_nominal_attribute(
    gpuDTE_NodeHeader_Train& node, gpuDTE_TmpNodeValues& tmp_node,
    float* curr_dist, int att_type, int nr_targets, bool tick_tock,
    int** indices_buffer, float* targer_data, int nr_instances, float* dataset);
template __device__ double GpuDte<double>::evaluate_nominal_attribute(
    gpuDTE_NodeHeader_Train& node, gpuDTE_TmpNodeValues& tmp_node,
    double* curr_dist, int att_type, int nr_targets, bool tick_tock,
    int** indices_buffer, double* targer_data, int nr_instances,
    double* dataset);

template __device__ void GpuDte<float>::gpudte_perform_split(
    gpuDTE_StaticInfo& static_info, gpuDTE_DatasetInfo& dataset_info,
    gpuDTE_IterationInfo& iteration_info, float** probability_buffers,
    float* probability_tmp, float* dataset, int* attribute_types,
    int* node_counts, int** indices_buffers, int* node_cursors,
    gpuDTE_NodeHeader_Train** node_buffers);
template __device__ void GpuDte<double>::gpudte_perform_split(
    gpuDTE_StaticInfo& static_info, gpuDTE_DatasetInfo& dataset_info,
    gpuDTE_IterationInfo& iteration_info, double** probability_buffers,
    double* probability_tmp, double* dataset, int* attribute_types,
    int* node_counts, int** indices_buffers, int* node_cursors,
    gpuDTE_NodeHeader_Train** node_buffers);

template __device__ void GpuDte<float>::gpudte_predict(
    int tid, int nr_instances, int data_type, int nr_targets,
    gpuDTE_NodeHeader_Classify* node_buffer, float* dataset,
    float* probability_buffer, float* predictions, int* attribute_types);
template __device__ void GpuDte<double>::gpudte_predict(
    int tid, int nr_instances, int data_type, int nr_targets,
    gpuDTE_NodeHeader_Classify* node_buffer, double* dataset,
    double* probability_buffer, double* predictions, int* attribute_types);

template __device__ float GpuDte<float>::AtomicAdd(float* address, float value);
template __device__ int GpuDte<float>::AtomicAdd(int* address, int value);
template __device__ unsigned int GpuDte<float>::AtomicAdd(unsigned int* address,
                                                          unsigned int value);
template __device__ double GpuDte<float>::AtomicAdd(double* address,
                                                    double value);

template __device__ float GpuDte<double>::AtomicAdd(float* address,
                                                    float value);
template __device__ int GpuDte<double>::AtomicAdd(int* address, int value);
template __device__ unsigned int GpuDte<double>::AtomicAdd(
    unsigned int* address, unsigned int value);
template __device__ double GpuDte<double>::AtomicAdd(double* address,
                                                     double value);
};