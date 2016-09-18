#define DLLExport
#define TestExport

#include "../../source_shared/include/global_defines.h"

#include "../../lib_core/include/lib_core.h"
#include "gpuert.h"

namespace lib_ensembles {
template <typename T>
__global__ void host_kernel(GpuErt<T> *gpu_algo,
                            GpuDteAlgorithmShared::GpuParams<T> params,
                            GpuDteAlgorithmShared::GpuDteKernelId id) {
  gpu_algo->GetConstPointers(&params.iteration_info, &params.dataset_info,
                             &params.static_info);
  switch (id) {
    case GpuDteAlgorithmShared::kSetupKernel:
      gpu_algo->gpuert_setup_kernel(&params, 123512);
      break;
    case GpuDteAlgorithmShared::kInitTreeBatch:
      gpu_algo->gpuert_initialize_tree_batch(&params);
      break;
    case GpuDteAlgorithmShared::kFindSplit:
      gpu_algo->gpuert_find_split(&params);
      break;
    case GpuDteAlgorithmShared::kPerformSplit:
      gpu_algo->gpuert_perform_split(&params);
      break;
    case GpuDteAlgorithmShared::kPredict:
      gpu_algo->gpuert_predict(&params);
      break;
    default:
      break;
  }
}

template <typename T>
void GpuErt<T>::CallCudaKernel(int blocks, int block_size,
                               GpuDteAlgorithmShared::GpuParams<T> &params,
                               GpuDteAlgorithmShared::GpuDteKernelId id) {
  host_kernel<T><<<blocks, block_size>>>(this, params, id);
}

template <typename T>
__device__ void GpuErt<T>::gpuert_setup_kernel(
    GpuDteAlgorithmShared::GpuParams<T> *params, unsigned long long seed) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  // Each thread gets same seed, a different sequence number, no offset
  if (id < params->iteration_info->threads_launched)
    curand_init(seed, id, 0, &params->random_states[id]);
}

template <typename T>
__device__ void GpuErt<T>::gpuert_initialize_tree_batch(
    GpuDteAlgorithmShared::GpuParams<T> *params) {
  __shared__ int s_indexCursor;

  if (threadIdx.x == 0) {
    if (params->static_info->balanced_sampling &&
        params->dataset_info->data_type == type_classification_)
      s_indexCursor = 0;
    else
      s_indexCursor = params->dataset_info->nr_instances;
  }

  __syncthreads();

  int treeOffset = params->dataset_info->nr_instances * blockIdx.x;
  if (params->dataset_info->data_type == type_classification_ &&
      params->static_info->balanced_sampling) {
    // Initialize indices main buffer
    int localCursor;
    int randVal;
    int stateId = (blockIdx.x * blockDim.x + threadIdx.x) %
                  params->static_info->node_buffer_size;
    curandStateMRG32k3a localState = params->random_states[stateId];
    for (int i = 0; i < params->dataset_info->nr_target_values; ++i) {
      int targetStart = params->target_starts[i];
      int targetEnd = (i == params->dataset_info->nr_target_values - 1)
                          ? params->dataset_info->nr_instances - 1
                          : params->target_starts[i + 1] - 1;

      for (int ii = threadIdx.x;
           ii < params->dataset_info->nr_instances /
                    params->dataset_info->nr_target_values;
           ii += blockDim.x) {
        localCursor = GpuDte<T>::AtomicAdd(&s_indexCursor, 1);
        if (targetEnd - targetStart > 0)
          randVal =
              targetStart + curand(&localState) % (targetEnd - targetStart);
        else
          randVal = targetStart;
        params->indices_buffer[0][treeOffset + localCursor] = randVal;
      }
    }
  } else {
    // Initialize indices main buffer
    for (int i = threadIdx.x; i < params->dataset_info->nr_instances;
         i += blockDim.x)
      params->indices_buffer[0][treeOffset + i] = i;
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T> root;
    root.parent_id = -2;
    root.attribute = -2;
    root.split_point = -2;
    root.tracking_id = GpuDte<T>::AtomicAdd(&params->node_cursors[node_id_], 1);
    root.node_index_start = treeOffset;
    root.node_index_count = s_indexCursor;

    params->node_buffers[params->iteration_info->read_buffer_id][blockIdx.x] =
        root;
  }
}
template <typename T>
__device__ void GpuErt<T>::gpuert_find_split(
    GpuDteAlgorithmShared::GpuParams<T> *params) {
  __shared__ T s_dynamic_shared[40];
  __shared__ GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T> s_tree_node;
  __shared__ GpuDteAlgorithmShared::gpuDTE_TmpNodeValues<T> s_tmp_node;
  __shared__ int s_attribute_type;
  __shared__ bool s_sensible_split;

  curandStateMRG32k3a localState;
  localState = params->random_states[(blockIdx.x * blockDim.x + threadIdx.x) %
                                     params->static_info->node_buffer_size];
  if (threadIdx.x == 0) {
    s_tree_node =
        params->node_buffers[params->iteration_info->read_buffer_id]
                            [blockIdx.x + params->iteration_info->node_offset];
    s_tmp_node =
        params
            ->node_tmp_buffer[blockIdx.x + params->iteration_info->node_offset];
    s_tmp_node.tmp_score = 0;
    s_sensible_split = false;
  }

  __syncthreads();

  bool firstFeature = true;
  int k = params->static_info->nr_features;
  int max_retries = k - params->dataset_info->nr_attributes < -10
                        ? -10
                        : k - params->dataset_info->nr_attributes;
  while ((k > max_retries) && (k-- > 0 || !s_sensible_split)) {
    if (threadIdx.x == 0) {
      s_tmp_node.tmp_attribute =
          curand(&localState) % params->dataset_info->nr_attributes;
      s_tmp_node.tmp_split = 0;

      s_attribute_type = params->attribute_type[s_tmp_node.tmp_attribute];
      s_attribute_type =
          s_attribute_type >= max_nominal_ ? 2 : s_attribute_type;
    }

    for (int i = threadIdx.x;
         i < params->dataset_info->nr_target_values * max_nominal_;
         i += blockDim.x)
      s_dynamic_shared[i] = 0;

    __syncthreads();

    if (threadIdx.x < 10) {
      T dat = get_data_point(
          s_tmp_node.tmp_attribute,
          params->indices_buffer[params->iteration_info->tick_tock ? 0 : 1]
                                [(curand(&localState) %
                                  s_tree_node.node_index_count) +
                                 s_tree_node.node_index_start],
          params->dataset_info->nr_instances, params->dataset);

      GpuDte<T>::AtomicAdd(&s_tmp_node.tmp_split, dat);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      s_tmp_node.tmp_split /= 10.0f;
    }

    __syncthreads();

    T response;
    switch (params->dataset_info->data_type) {
      case type_classification_:
        response = eval_numeric_attribute(params, s_tree_node, s_tmp_node,
                                          s_dynamic_shared, s_attribute_type);
        break;
      case type_regression_:
        response = varianceCalculation(params, s_tree_node, s_tmp_node,
                                       s_dynamic_shared);
        break;
    }

    if (threadIdx.x == 0) {
      if (s_tmp_node.tmp_score < response || firstFeature) {
        // Save splitpoint, attribute and distribution
        s_tmp_node.tmp_score = response;
        s_tree_node.split_point = s_tmp_node.tmp_split;
        s_tree_node.attribute = s_tmp_node.tmp_attribute;

        switch (params->dataset_info->data_type) {
          case type_classification_:
            for (int i = 0;
                 i < params->dataset_info->nr_target_values * max_nominal_; ++i)
              params->probability_tmp_buffer
                  [blockIdx.x * params->dataset_info->nr_target_values *
                       max_nominal_ +
                   i] = s_dynamic_shared[i];
            break;
          case type_regression_:
            params->probability_tmp_buffer
                [blockIdx.x * params->dataset_info->nr_target_values *
                 max_nominal_] = s_dynamic_shared[2];
            params->probability_tmp_buffer
                [blockIdx.x * params->dataset_info->nr_target_values *
                     max_nominal_ +
                 1] = s_dynamic_shared[3];
            break;
        }
      }

      if (s_tmp_node.tmp_score > 1e-3) s_sensible_split = true;

      firstFeature = false;
    }

    __syncthreads();
  }

  params->random_states[(blockIdx.x * blockDim.x + threadIdx.x) %
                        params->static_info->node_buffer_size] = localState;

  // Copy back result
  if (threadIdx.x == 0) {
    params->node_buffers[params->iteration_info->read_buffer_id]
                        [blockIdx.x + params->iteration_info->node_offset] =
        s_tree_node;
    params->node_tmp_buffer[blockIdx.x + params->iteration_info->node_offset] =
        s_tmp_node;
    // params->random_states[blockIdx.x] = localState;
  }
}
template <typename T>
__device__ void GpuErt<T>::gpuert_perform_split(
    GpuDteAlgorithmShared::GpuParams<T> *params) {
  __shared__ int s_node_counts[40];

  gpudte_perform_split(*(params->static_info), *(params->dataset_info),
                       *(params->iteration_info), params->probability_buffers,
                       params->probability_tmp_buffer, params->dataset,
                       params->attribute_type, s_node_counts,
                       params->indices_buffer, params->node_cursors,
                       params->node_buffers);
}
template <typename T>
__device__ void GpuErt<T>::gpuert_predict(
    GpuDteAlgorithmShared::GpuParams<T> *params) {
  if (threadIdx.x + blockIdx.x * blockDim.x >=
      params->iteration_info->threads_launched)
    return;

  int tid = params->iteration_info->tree_offset + threadIdx.x +
            blockIdx.x * blockDim.x;
  gpudte_predict(
      tid, params->dataset_info->nr_instances, params->dataset_info->data_type,
      params->dataset_info->nr_target_values, params->node_buffers_classify,
      params->dataset, params->probability_tmp_buffer, params->predictions,
      params->attribute_type);
}
template <typename T>
__device__ T GpuErt<T>::eval_numeric_attribute(
    GpuDteAlgorithmShared::GpuParams<T> *params,
    GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T> &node,
    GpuDteAlgorithmShared::gpuDTE_TmpNodeValues<T> &tmp_node, T *curr_dist,
    int att_type) {
  int numInds = node.node_index_count;
  int nodeIndStart = node.node_index_start;
  int weight = 1;
  int inst;
  T val;

  for (int i = threadIdx.x; i < numInds; i += blockDim.x) {
    inst = params->indices_buffer[params->iteration_info->tick_tock ? 0 : 1]
                                 [nodeIndStart + i];

    val = get_data_point(tmp_node.tmp_attribute, inst,
                         params->dataset_info->nr_instances, params->dataset);

    if (val != -flt_max)
      GpuDte<T>::AtomicAdd(&curr_dist[params->dataset_info->nr_target_values *
                                          ((val < tmp_node.tmp_split) ? 0 : 1) +
                                      int(params->target_data[inst])],
                           weight);
    else
      GpuDte<T>::AtomicAdd(&curr_dist[int(params->target_data[inst])], weight);
  }

  __syncthreads();

  T response = 0;
  if (threadIdx.x == 0) {
    int count;
    for (int i = 0; i < 2; ++i) {
      count = 0;
      for (int ii = 0; ii < params->dataset_info->nr_target_values; ++ii) {
        count += curr_dist[i * params->dataset_info->nr_target_values + ii];
      }
      if (count == 0) response = -flt_max;
    }

    if (response != -flt_max) {
      T prior = entropy_over_columns(curr_dist, att_type,
                                     params->dataset_info->nr_target_values);
      T posterior = entropy_conditioned_on_rows(
          curr_dist, att_type, params->dataset_info->nr_target_values);
      response = prior - posterior;
    }
  }

  return response;
}

template <typename T>
__device__ T GpuErt<T>::varianceCalculation(
    GpuDteAlgorithmShared::GpuParams<T> *params,
    GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T> &node,
    GpuDteAlgorithmShared::gpuDTE_TmpNodeValues<T> &tmp_node, T *curr_dist) {
  __shared__ T s_means[2];
  int numInds = node.node_index_count;
  int nodeIndStart = node.node_index_start;
  int attribute = tmp_node.tmp_attribute;
  int inst;
  T val;

  if (threadIdx.x < 2) {
    s_means[threadIdx.x] = 0;
    curr_dist[threadIdx.x] = 0;
  }

  __syncthreads();

  // Calculate mean values from split
  for (int i = threadIdx.x; i < numInds; i += blockDim.x) {
    inst = params->indices_buffer[params->iteration_info->tick_tock ? 0 : 1]
                                 [nodeIndStart + i];
    val = get_data_point(attribute, inst, params->dataset_info->nr_instances,
                         params->dataset);
    int t = node.node_index_count;
    if (val != -flt_max && t > 0) {
      GpuDte<T>::AtomicAdd(&curr_dist[(val < tmp_node.tmp_split) ? 0 : 1], 1);
      GpuDte<T>::AtomicAdd(&s_means[(val < tmp_node.tmp_split) ? 0 : 1],
                           params->target_data[inst]);
    }
  }

  __syncthreads();

  T gain = 0;
  if (threadIdx.x == 0) {
    for (int i = 0; i < 2; ++i) {
      if (curr_dist[i] == 0) gain = -flt_max;
    }

    if (gain != -flt_max) {
      curr_dist[2] = (curr_dist[0] != 0) ? s_means[0] / T(curr_dist[0]) : 0;
      curr_dist[3] = (curr_dist[1] != 0) ? s_means[1] / T(curr_dist[1]) : 0;

      T nLeft = curr_dist[0] == 0 ? 1 : curr_dist[0];
      T nRight = curr_dist[1] == 0 ? 1 : curr_dist[1];
      T diff = ((s_means[0] / nLeft) - (s_means[1] / nRight));

      gain = (nLeft * nRight * diff * diff / (nLeft + nRight));
    }
  }
  return gain;
}

template GpuErt<float>::GpuErt();
template GpuErt<double>::GpuErt();
}