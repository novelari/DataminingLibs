#define DLLExport
#define TestExport

#include "gpudte.h"

#include "../../lib_gpu/source/gpu_device_cuda.h"

namespace lib_ensembles {
template <typename T>
__device__ T GpuDte::get_data_point(int attribute, int instance,
                                    int nr_instances, T* dataset) {
  return dataset[attribute * nr_instances + instance];
}

template <typename T>
__device__ T GpuDte::lnFunc(T num) {
  if (num <= 1e-6) return 0;
  return num * log(num);
}

template <typename T>
__device__ T GpuDte::entropy_conditioned_on_rows(T* matrix, int att_type,
                                                 int nr_targets) {
  int x = (att_type > max_nominal_) ? 2 : att_type;
  int y = nr_targets;

  T returnValue = 0, sumForRow, total = 0;
  for (int i = 0; i < x; i++) {
    sumForRow = 0;
    for (int ii = 0; ii < y; ii++) {
      returnValue = returnValue + lnFunc(matrix[y * i + ii]);
      sumForRow += matrix[y * i + ii];
    }
    returnValue = returnValue - lnFunc(sumForRow);
    total += sumForRow;
  }
  if (total < 1.0e-6) {
    return 0;
  }
  return -returnValue / (total * log(T(2.0)));
}

template <typename T>
__device__ T GpuDte::entropy_over_columns(T* matrix, int att_type,
                                          int nr_targets) {
  int x = (att_type > max_nominal_) ? 2 : att_type;
  int y = nr_targets;

  T returnValue = 0, sumForColumn, total = 0;
  for (int ii = 0; ii < y; ii++) {
    sumForColumn = 0;
    for (int i = 0; i < x; i++) {
      sumForColumn += matrix[y * i + ii];
    }
    returnValue = returnValue - lnFunc(sumForColumn);
    total += sumForColumn;
  }
  if (total < 1.0e-6) {
    return 0;
  }
  return (returnValue + lnFunc(total)) / (total * log(T(2.0)));
}

template <typename T>
__device__ T GpuDte::evaluate_nominal_attribute(
    gpuDTE_NodeHeader_Train<T>& node, gpuDTE_TmpNodeValues<T>& tmp_node,
    T* curr_dist, int att_type, int nr_targets, bool tick_tock,
    int** indices_buffer, T* targer_data, int nr_instances, T* dataset) {
  int numInds = node.node_index_count;
  int nodeIndStart = node.node_index_start;
  T weight = 1;
  int inst;
  T val;

  // Split on median value
  for (int i = threadIdx.x; i < numInds; i += blockDim.x) {
    inst = indices_buffer[tick_tock ? 0 : 1][nodeIndStart + i];

    val = get_data_point(tmp_node.tmp_attribute, inst, nr_instances, dataset);

    if (val != -flt_max)
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
__device__ void GpuDte::gpudte_perform_split(
    gpuDTE_StaticInfo& static_info, gpuDTE_DatasetInfo& dataset_info,
    gpuDTE_IterationInfo& iteration_info, T** probability_buffers,
    T* probability_tmp, T* dataset, int* attribute_types, int* node_counts,
    int** indices_buffers, int* node_cursors,
    gpuDTE_NodeHeader_Train<T>** node_buffers) {
  __shared__ gpuDTE_NodeHeader_Train<T> s_tree_node;
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
    data_point = get_data_point(s_tree_node.attribute, instance_ind,
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

    gpuDTE_NodeHeader_Train<T> child;
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
__device__ void GpuDte::gpudte_predict(
    int tid, int nr_instances, int data_type, int nr_targets,
    gpuDTE_NodeHeader_Classify<T>* node_buffer, T* dataset,
    T* probability_buffer, T* predictions, int* attribute_types) {
  // Access pattern
  // instances*trees: 5*5 = 25
  // tid:			0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
  // instances:		0 1 2 3 4 0 1 2 3 4 0  1  2  3  4  0  1  2  3
  // trees:			0 0 0 0 0 1 1 1 1 1 2  2  2  2  2  3  3  3  3

  int instanceId = tid % nr_instances;
  int treeId = T(tid) / T(nr_instances);
  T dataPoint;

  gpuDTE_NodeHeader_Classify<T> tree_node = node_buffer[treeId];
  while (tree_node.child_count > 0) {
    dataPoint =
        get_data_point(tree_node.attribute, instanceId, nr_instances, dataset);
    if (dataPoint != -flt_max) {
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

__device__ float GpuDte::AtomicAdd(float* address, float value) {
  return atomicAdd(address, value);
}

__device__ int GpuDte::AtomicAdd(int* address, int value) {
  return atomicAdd(address, value);
}

__device__ unsigned int GpuDte::AtomicAdd(unsigned int* address,
                                          unsigned int value) {
  return atomicAdd(address, value);
}

__device__ double GpuDte::AtomicAdd(double* address, double value) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old =
        atomicCAS(address_as_ull, assumed,
                  __double_as_longlong(value + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

template <typename T>
GpuDte::GpuParams<T>::GpuParams() {
  node_cursors = nullptr;
  probability_buffers[0] = nullptr;
  probability_buffers[1] = nullptr;
  node_buffers[0] = nullptr;
  node_buffers[1] = nullptr;
  node_buffers[2] = nullptr;
  node_buffers_classify = nullptr;
  node_tmp_buffer = nullptr;
  probability_tmp_buffer = nullptr;
  indices_buffer[0] = nullptr;
  indices_buffer[1] = nullptr;
  indices_inbag = nullptr;
  target_starts = nullptr;
  random_states = nullptr;
  dataset = nullptr;
  target_data = nullptr;
  attribute_type = nullptr;
  predictions = nullptr;
  oobCounts = nullptr;
  mse = nullptr;
}
template <typename T>
GpuDte::GpuParams<T>::~GpuParams() {}

template <typename T>
void GpuDte::GpuParams<T>::finalize(sp<lib_gpu::GpuDevice> dev) {
  dev->DeallocateMemory(node_cursors);
  dev->DeallocateMemory(probability_buffers[0]);
  dev->DeallocateMemory(probability_buffers[1]);
  dev->DeallocateMemory(node_buffers[0]);
  dev->DeallocateMemory(node_buffers[1]);
  dev->DeallocateMemory(node_buffers[2]);
  dev->DeallocateMemory(node_buffers_classify);
  dev->DeallocateMemory(node_tmp_buffer);
  dev->DeallocateMemory(probability_tmp_buffer);
  dev->DeallocateMemory(indices_buffer[0]);
  dev->DeallocateMemory(indices_buffer[1]);
  dev->DeallocateMemory(indices_inbag);
  dev->DeallocateMemory(target_starts);
  dev->DeallocateMemory(random_states);
  dev->DeallocateMemory(dataset);
  dev->DeallocateMemory(target_data);
  dev->DeallocateMemory(attribute_type);
  dev->DeallocateMemory(predictions);
  dev->DeallocateMemory(oobCounts);
  dev->DeallocateMemory(mse);
}

template GpuDte::GpuParams<float>::GpuParams();
template GpuDte::GpuParams<double>::GpuParams();
template GpuDte::GpuParams<float>::~GpuParams();
template GpuDte::GpuParams<double>::~GpuParams();
template void GpuDte::GpuParams<float>::finalize(sp<lib_gpu::GpuDevice> dev);
template void GpuDte::GpuParams<double>::finalize(sp<lib_gpu::GpuDevice> dev);

template __device__ void GpuDte::gpudte_predict(
    int tid, int nr_instances, int data_type, int nr_targets,
    gpuDTE_NodeHeader_Classify<float>* node_buffer, float* dataset,
    float* probability_buffer, float* predictions, int* attribute_types);
template __device__ void GpuDte::gpudte_predict(
    int tid, int nr_instances, int data_type, int nr_targets,
    gpuDTE_NodeHeader_Classify<double>* node_buffer, double* dataset,
    double* probability_buffer, double* predictions, int* attribute_types);

template __device__ void GpuDte::gpudte_perform_split(
    gpuDTE_StaticInfo& static_info, gpuDTE_DatasetInfo& dataset_info,
    gpuDTE_IterationInfo& iteration_info, float** probability_buffers,
    float* probability_tmp, float* dataset, int* attribute_types,
    int* node_counts, int** indices_buffers, int* node_cursors,
    gpuDTE_NodeHeader_Train<float>** node_buffers);

template __device__ void GpuDte::gpudte_perform_split(
    gpuDTE_StaticInfo& static_info, gpuDTE_DatasetInfo& dataset_info,
    gpuDTE_IterationInfo& iteration_info, double** probability_buffers,
    double* probability_tmp, double* dataset, int* attribute_types,
    int* node_counts, int** indices_buffers, int* node_cursors,
    gpuDTE_NodeHeader_Train<double>** node_buffers);

template __device__ float GpuDte::evaluate_nominal_attribute(
    gpuDTE_NodeHeader_Train<float>& node, gpuDTE_TmpNodeValues<float>& tmp_node,
    float* curr_dist, int att_type, int nr_targets, bool tick_tock,
    int** indices_buffer, float* targer_data, int nr_instances, float* dataset);

template __device__ double GpuDte::evaluate_nominal_attribute(
    gpuDTE_NodeHeader_Train<double>& node,
    gpuDTE_TmpNodeValues<double>& tmp_node, double* curr_dist, int att_type,
    int nr_targets, bool tick_tock, int** indices_buffer, double* targer_data,
    int nr_instances, double* dataset);

template __device__ float GpuDte::entropy_conditioned_on_rows(float* matrix,
                                                              int att_type,
                                                              int nr_targets);

template __device__ double GpuDte::entropy_conditioned_on_rows(double* matrix,
                                                               int att_type,
                                                               int nr_targets);

template __device__ float GpuDte::entropy_over_columns(float* matrix,
                                                       int att_type,
                                                       int nr_targets);
template __device__ double GpuDte::entropy_over_columns(double* matrix,
                                                        int att_type,
                                                        int nr_targets);

template __device__ float GpuDte::get_data_point(int attribute, int instance,
                                                 int nr_instances,
                                                 float* dataset);
template __device__ double GpuDte::get_data_point(int attribute, int instance,
                                                  int nr_instances,
                                                  double* dataset);

template __device__ float GpuDte::lnFunc(float num);

template __device__ double GpuDte::lnFunc(double num);
}