#define DLLExport
#define TestExport

#include "../../lib_gpu/include/cuda_kernel_helpers.hpp"
#include "gpurf.h"

namespace lib_ensembles {
template <typename T>
__global__ void host_kernel(GpuRf<T>* gpu_algo,
                            GpuDteAlgorithmShared::GpuParams<T> params,
                            GpuDteAlgorithmShared::GpuDteKernelId type) {
  gpu_algo->GetConstPointers(&params.iteration_info, &params.dataset_info,
                             &params.static_info);
  switch (type) {
    case GpuDteAlgorithmShared::kSetupKernel:
      gpu_algo->gpurf_setup_kernel(&params);
      break;
    case GpuDteAlgorithmShared::kInitTreeBatch:
      gpu_algo->gpurf_initialize_tree_batch(&params);
      break;
    case GpuDteAlgorithmShared::kFindSplit:
      gpu_algo->gpurf_find_split(&params);
      break;
    case GpuDteAlgorithmShared::kPerformSplit:
      gpu_algo->gpurf_perform_split(&params);
      break;
    case GpuDteAlgorithmShared::kPredict:
      gpu_algo->gpurf_predict(&params);
      break;
    case GpuDteAlgorithmShared::kOobEstimate:
      gpu_algo->gpurf_oob_estimate(&params);
      break;
    case GpuDteAlgorithmShared::kFeatureImp:
      gpu_algo->gpurf_feature_importance(&params);
      break;
    default:
      break;
  }
}

template <typename T>
void GpuRf<T>::CallCudaKernel(int blocks, int block_size,
                              GpuDteAlgorithmShared::GpuParams<T>& params,
                              GpuDteAlgorithmShared::GpuDteKernelId id) {
  host_kernel<T><<<blocks, block_size>>>(this, params, id);
}

template <typename T>
__device__ void GpuRf<T>::gpurf_setup_kernel(
    GpuDteAlgorithmShared::GpuParams<T>* params) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  // Each thread gets same seed, a different sequence number, no offset
  if (id < params->iteration_info->threads_launched)
    curand_init(324123, id, 0, &params->random_states[id]);
}

template <typename T>
__device__ void GpuRf<T>::gpurf_initialize_tree_batch(
    GpuDteAlgorithmShared::GpuParams<T>* params) {
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
    // Initialize probability main buffer
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
        localCursor = AtomicAdd(&s_indexCursor, 1);
        if (targetEnd - targetStart > 0)
          randVal =
              targetStart + curand(&localState) % (targetEnd - targetStart);
        else
          randVal = targetStart;
        params->indices_buffer[0][treeOffset + localCursor] = randVal;
        params->indices_inbag[treeOffset + randVal] = true;
      }
    }
  } else {
    // Initialize indices main buffer
    int randVal;
    int stateId = (blockIdx.x * blockDim.x + threadIdx.x) %
                  params->static_info->node_buffer_size;
    curandStateMRG32k3a localState = params->random_states[stateId];
    for (int i = threadIdx.x; i < params->dataset_info->nr_instances;
         i += blockDim.x) {
      randVal = curand(&localState) % params->dataset_info->nr_instances;
      params->indices_buffer[0][treeOffset + i] = randVal;
      params->indices_inbag[treeOffset + randVal] = true;
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T> root;
    root.parent_id = -2;
    root.attribute = -2;
    root.split_point = -2;
    root.tracking_id = AtomicAdd(&params->node_cursors[node_id_], 1);
    root.node_index_start = treeOffset;
    root.node_index_count = s_indexCursor;

    params->node_buffers[params->iteration_info->read_buffer_id][blockIdx.x] =
        root;
  }
}

template <typename T>
__device__ void GpuRf<T>::gpurf_find_split(
    GpuDteAlgorithmShared::GpuParams<T>* params) {
  __shared__ T s_dynamic_shared[40];
  __shared__ unsigned int s_histograms[1024];
  __shared__ unsigned int s_offsets[256];
  __shared__ GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T> s_tree_node;
  __shared__ GpuDteAlgorithmShared::gpuDTE_TmpNodeValues<T> s_tmp_node;
  __shared__ int s_attribute_type;
  __shared__ bool s_sensible_split;

  curandStateMRG32k3a localState;
  if (threadIdx.x == 0) {
    localState = params->random_states[blockIdx.x];
    s_tree_node =
        params->node_buffers[params->iteration_info->read_buffer_id]
                            [blockIdx.x + params->iteration_info->node_offset];
    s_tmp_node =
        params
            ->node_tmp_buffer[blockIdx.x + params->iteration_info->node_offset];
    s_tmp_node.tmp_score = 0;
    s_sensible_split = false;
    for (int i = 0; i < params->dataset_info->nr_target_values * max_nominal_;
         ++i)
      s_dynamic_shared[i] = 0;
  }

  __syncthreads();

  // Find prior distribution
  for (int i = threadIdx.x; i < s_tree_node.node_index_count; i += blockDim.x) {
    int inst = params->indices_buffer[params->iteration_info->tick_tock ? 0 : 1]
                                     [s_tree_node.node_index_start + i];
    switch (params->dataset_info->data_type) {
      case type_classification_:
        AtomicAdd(&s_dynamic_shared[int(params->target_data[inst])], T(1));
        break;
      case type_regression_:
        AtomicAdd(&s_dynamic_shared[0], params->target_data[inst]);
        break;
    }
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

      s_attribute_type = params->attribute_type[s_tmp_node.tmp_attribute];
      s_attribute_type =
          s_attribute_type >= max_nominal_ ? 2 : s_attribute_type;
    }

    __syncthreads();

    // Sort indices on attribute
    radix_sort_on_attribute(params, s_tree_node, s_tmp_node, s_histograms,
                            s_offsets);

    __syncthreads();

    T response;
    switch (params->dataset_info->data_type) {
      case type_classification_:
        response = eval_numeric_attribute(params, s_tree_node, s_tmp_node,
                                          s_dynamic_shared, s_attribute_type,
                                          s_histograms, s_offsets);
        break;
      case type_regression_:
        response =
            GpuRf::variance_calculation(params, s_tree_node, s_tmp_node,
                                        s_dynamic_shared, (T*)s_histograms);
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

  // Copy back result
  if (threadIdx.x == 0) {
    params->node_buffers[params->iteration_info->read_buffer_id]
                        [blockIdx.x + params->iteration_info->node_offset] =
        s_tree_node;
    params->node_tmp_buffer[blockIdx.x + params->iteration_info->node_offset] =
        s_tmp_node;
    params->random_states[blockIdx.x] = localState;
  }
}

template <typename T>
__device__ void GpuRf<T>::gpurf_perform_split(
    GpuDteAlgorithmShared::GpuParams<T>* params) {
  __shared__ int s_node_counts[40];

  gpudte_perform_split(*(params->static_info), *(params->dataset_info),
                       *(params->iteration_info), params->probability_buffers,
                       params->probability_tmp_buffer, params->dataset,
                       params->attribute_type, s_node_counts,
                       params->indices_buffer, params->node_cursors,
                       params->node_buffers);
}

template <typename T>
__device__ void GpuRf<T>::gpurf_predict(
    GpuDteAlgorithmShared::GpuParams<T>* params) {
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
__device__ void GpuRf<T>::gpurf_oob_estimate(
    GpuDteAlgorithmShared::GpuParams<T>* params) {
  if (threadIdx.x + blockIdx.x * blockDim.x >=
      params->iteration_info->threads_launched)
    return;

  int tid = params->iteration_info->tree_offset + threadIdx.x +
            blockIdx.x * blockDim.x;
  int instanceId = tid % params->dataset_info->nr_instances;
  int treeId = T(tid) / T(params->dataset_info->nr_instances);
  bool inBag =
      params->indices_inbag[params->dataset_info->nr_instances * treeId +
                            instanceId];
  if (inBag) return;

  T dataPoint;
  GpuDteAlgorithmShared::gpuDTE_NodeHeader_Classify<T> tree_node =
      params->node_buffers_classify[treeId];
  while (tree_node.child_count != 0) {
    int attribute_type = params->attribute_type[tree_node.attribute];

    dataPoint =
        get_data_point(tree_node.attribute, instanceId,
                       params->dataset_info->nr_instances, params->dataset);
    if (attribute_type > max_nominal_ ||
        params->dataset_info->data_type == type_regression_) {
      if (dataPoint != -flt_max)
        tree_node =
            (dataPoint < tree_node.split_point)
                ? params->node_buffers_classify[tree_node.child_start]
                : params->node_buffers_classify[tree_node.child_start + 1];
      else
        tree_node = params->node_buffers_classify[tree_node.child_start];
    } else {
      if (dataPoint != -flt_max)
        tree_node =
            params
                ->node_buffers_classify[tree_node.child_start + int(dataPoint)];
      else
        tree_node = params->node_buffers_classify[tree_node.child_start];
    }
  }

  switch (params->dataset_info->data_type) {
    case type_classification_: {
      int classSelector = 0;
      T max_prob = 0;
      for (int i = 0; i < params->dataset_info->nr_target_values; ++i) {
        if (params->probability_tmp_buffer[tree_node.probability_start + i] >
            max_prob) {
          max_prob =
              params->probability_tmp_buffer[tree_node.probability_start + i];
          classSelector = i;
        }
      }

      if (params->target_data[instanceId] == classSelector)
        AtomicAdd(&params->oobCounts[0], 1);
      else
        AtomicAdd(&params->oobCounts[1], 1);
      break;
    }
    case type_regression_:
      AtomicAdd(&params->mse[0],
                params->probability_tmp_buffer[tree_node.probability_start]);
      break;
  }
}

template <typename T>
__device__ void GpuRf<T>::gpurf_feature_importance(
    GpuDteAlgorithmShared::GpuParams<T>* params) {
  if (threadIdx.x + blockIdx.x * blockDim.x >=
      params->iteration_info->threads_launched)
    return;

  int randAttribute = params->iteration_info->depth;
  int tid = params->iteration_info->tree_offset + threadIdx.x +
            blockIdx.x * blockDim.x;
  int instanceId = tid % params->dataset_info->nr_instances;
  int treeId = T(tid) / T(params->dataset_info->nr_instances);
  bool inBag =
      params->indices_inbag[params->dataset_info->nr_instances * treeId +
                            instanceId];
  if (inBag) return;

  int instance;
  curandStateMRG32k3a localState = params->random_states[blockIdx.x];
  T dataPoint;
  GpuDteAlgorithmShared::gpuDTE_NodeHeader_Classify<T> tree_node =
      params->node_buffers_classify[treeId];
  while (tree_node.child_count != 0) {
    int attribute_type = params->attribute_type[tree_node.attribute];

    if (randAttribute == tree_node.attribute)
      instance = curand(&localState) % params->dataset_info->nr_instances;
    else
      instance = instanceId;

    dataPoint =
        get_data_point(tree_node.attribute, instance,
                       params->dataset_info->nr_instances, params->dataset);
    if (attribute_type > max_nominal_ ||
        params->dataset_info->data_type == type_regression_) {
      if (dataPoint != -flt_max)
        tree_node =
            (dataPoint < tree_node.split_point)
                ? params->node_buffers_classify[tree_node.child_start]
                : params->node_buffers_classify[tree_node.child_start + 1];
      else
        tree_node = params->node_buffers_classify[tree_node.child_start];
    } else {
      if (dataPoint != -flt_max)
        tree_node =
            params
                ->node_buffers_classify[tree_node.child_start + int(dataPoint)];
      else
        tree_node = params->node_buffers_classify[tree_node.child_start];
    }
  }

  switch (params->dataset_info->data_type) {
    case type_classification_: {
      int classSelector = 0;
      T max_prob = 0;
      for (int i = 0; i < params->dataset_info->nr_target_values; ++i) {
        if (params->probability_tmp_buffer[tree_node.probability_start + i] >
            max_prob) {
          max_prob =
              params->probability_tmp_buffer[tree_node.probability_start + i];
          classSelector = i;
        }
      }

      if (params->target_data[instanceId] == classSelector)
        AtomicAdd(&params->oobCounts[randAttribute * 2], 1);
      else
        AtomicAdd(&params->oobCounts[randAttribute * 2 + 1], 1);
      break;
    }
    case type_regression_:
      AtomicAdd(&params->mse[randAttribute],
                params->probability_tmp_buffer[tree_node.probability_start]);
      break;
  }
}

template <typename T>
__device__ void GpuRf<T>::radix_sort_on_attribute(
    GpuDteAlgorithmShared::GpuParams<T>* params,
    GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T>& node,
    GpuDteAlgorithmShared::gpuDTE_TmpNodeValues<T>& tmp_node,
    unsigned int s_histograms[1024], unsigned int s_offsets[256]) {
  __shared__ unsigned int s_nrNegativeValues;
  __shared__ unsigned char s_thread_radix[64];

  if (threadIdx.x == 0) s_nrNegativeValues = 0;

  unsigned int* input =
      (unsigned int*)&params->dataset[tmp_node.tmp_attribute *
                                      params->dataset_info->nr_instances];
  unsigned int* indices =
      (unsigned int*)&params
          ->indices_buffer[params->iteration_info->tick_tock ? 0 : 1]
                          [node.node_index_start];
  unsigned int* indices2 =
      (unsigned int*)&params
          ->indices_buffer[params->iteration_info->tick_tock ? 1 : 0]
                          [node.node_index_start];

  for (int i = threadIdx.x; i < 1024; i += blockDim.x) s_histograms[i] = 0;

  __syncthreads();

  unsigned char* dataVal;
  for (int i = threadIdx.x; i < node.node_index_count; i += blockDim.x) {
    dataVal = (unsigned char*)&input[indices[i]];
    AtomicAdd(&s_histograms[*dataVal], 1);
    AtomicAdd(&s_histograms[256 + (*(dataVal + 1))], 1);
    AtomicAdd(&s_histograms[512 + (*(dataVal + 2))], 1);
    AtomicAdd(&s_histograms[768 + (*(dataVal + 3))], 1);
  }

  __syncthreads();

  for (int i = threadIdx.x + 128; i < 256; i += blockDim.x)
    AtomicAdd(&s_nrNegativeValues, s_histograms[768 + i]);

  // Radix sort, j is the pass number (0=LSB, 3=MSB)
  bool performPass;
  unsigned int* curCount;
  unsigned char uniqueVal;
  for (int j = 0; j < 4; j++) {
    __syncthreads();

    performPass = true;
    curCount = &s_histograms[j << 8];
    uniqueVal = *(((unsigned char*)&input[indices[0]]) + j);
    if (curCount[uniqueVal] == node.node_index_count) performPass = false;

    // Should we care about negative values?
    if (j != 3) {
      // Here we deal with positive values only
      if (performPass) {
        // Create offsets
        lib_gpu::CudaKernelHelpers::inplace_prefixsum(curCount, 256);

        // Perform Radix Sort
        bool skip;
        unsigned int id, spot;
        unsigned char radix;
        for (int i = 0; i < node.node_index_count; i += blockDim.x) {
          skip = threadIdx.x + i >= node.node_index_count;
          if (!skip) {
            id = indices[threadIdx.x + i];
            radix = *(((unsigned char*)&input[id]) + j);
            s_thread_radix[threadIdx.x] = radix;
            spot = curCount[radix];
          }

          __syncthreads();

          if (!skip) {
            AtomicAdd(&curCount[radix], 1);
            for (int ii = threadIdx.x; ii > 0; --ii)
              if (s_thread_radix[ii - 1] == radix) ++spot;
            indices2[spot] = id;
          }

          __syncthreads();
        }

        unsigned int* Tmp = indices;
        indices = indices2;
        indices2 = Tmp;
      }
    } else {
      // This is a special case to correctly handle negative values
      if (performPass) {
// Create biased offsets, in order for negative numbers to be sorted as well
#pragma unroll
        for (int i = threadIdx.x; i < 256; i += blockDim.x)
          s_offsets[i] = curCount[i];

        __syncthreads();

        lib_gpu::CudaKernelHelpers::inplace_prefixsum(s_offsets, 128);
        if (threadIdx.x == 0) s_offsets[0] = s_nrNegativeValues;

        lib_gpu::CudaKernelHelpers::inplace_reverse_prefixsum(&s_offsets[128],
                                                              128);

// Fixing the wrong place for negative values
#pragma unroll
        for (int i = threadIdx.x + 128; i < 256; i += blockDim.x)
          s_offsets[i] += curCount[i];

        __syncthreads();

        bool skip;
        int spot;
        unsigned int id;
        unsigned char radix;
        for (int i = 0; i < node.node_index_count; i += blockDim.x) {
          skip = threadIdx.x + i >= node.node_index_count;
          if (!skip) {
            id = indices[threadIdx.x + i];
            radix = input[id] >> 24;
            s_thread_radix[threadIdx.x] = radix;
            if (radix < 128)
              spot = s_offsets[radix];
            else
              spot = s_offsets[radix] - 1;
          }

          __syncthreads();

          if (!skip) {
            if (radix < 128)
              AtomicAdd((int*)&s_offsets[radix], 1);
            else
              AtomicAdd((int*)&s_offsets[radix], -1);

            for (int ii = threadIdx.x; ii > 0; --ii)
              if (s_thread_radix[ii - 1] == radix) spot += radix < 128 ? 1 : -1;

            indices2[spot] = id;
          }

          __syncthreads();
        }

        unsigned int* Tmp = indices;
        indices = indices2;
        indices2 = Tmp;
      } else {
        // The pass is useless, yet we still have to reverse the order of
        // current list if all values are negative.
        if (uniqueVal >= 128) {
          for (unsigned int i = threadIdx.x; i < node.node_index_count;
               i += blockDim.x)
            indices2[i] = indices[node.node_index_count - i - 1];

          // Swap pointers for next pass. Valid indices - the most recent ones -
          // are in mIndices after the swap.
          unsigned int* Tmp = indices;
          indices = indices2;
          indices2 = Tmp;
        }
      }
    }
  }

  __syncthreads();

  // Need to copy back to the correct indices buffer
  if (indices !=
      (unsigned int*)&params
          ->indices_buffer[params->iteration_info->tick_tock ? 0 : 1]
                          [node.node_index_start]) {
    for (int i = threadIdx.x; i < node.node_index_count; i += blockDim.x)
      indices2[i] = indices[i];
  }
}

template <typename T>
__device__ T GpuRf<T>::eval_numeric_attribute(
    GpuDteAlgorithmShared::GpuParams<T>* params,
    GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T>& node,
    GpuDteAlgorithmShared::gpuDTE_TmpNodeValues<T>& tmp_node, T* curr_dist,
    int att_type, unsigned int* s_histograms, unsigned int* s_offsets) {
  __shared__ T local_dist[40];
  int numInds = node.node_index_count;
  int nodeIndStart = node.node_index_start;
  int inst, bestI = 0;
  T val, preVal;
  T response, bestResponse = 0.0f;

  for (int i = 0; i < params->dataset_info->nr_target_values; ++i)
    local_dist[i + params->dataset_info->nr_target_values] = 0;
  for (int i = 0; i < params->dataset_info->nr_target_values; ++i)
    local_dist[i] =
        curr_dist[i] + curr_dist[i + params->dataset_info->nr_target_values];

  s_offsets[threadIdx.x] = 0;
  T prior = entropy_over_columns((T*)curr_dist, att_type,
                                 params->dataset_info->nr_target_values);

  __syncthreads();

  for (int i = threadIdx.x;
       i < params->dataset_info->nr_target_values * max_nominal_;
       i += blockDim.x)
    curr_dist[i] = local_dist[i];

  __syncthreads();

  // Find best split on attribute
  bool skip;
  for (int i = 0; i < numInds; i += blockDim.x) {
    skip = threadIdx.x + i >= node.node_index_count;

    if (!skip) {
      inst = params->indices_buffer[params->iteration_info->tick_tock ? 0 : 1]
                                   [nodeIndStart + i + threadIdx.x];
      val = get_data_point(tmp_node.tmp_attribute, inst,
                           params->dataset_info->nr_instances, params->dataset);
      s_offsets[threadIdx.x] = int(params->target_data[inst]);

      if (i + threadIdx.x != 0) {
        inst = params->indices_buffer[params->iteration_info->tick_tock ? 0 : 1]
                                     [nodeIndStart + i + threadIdx.x - 1];
        preVal =
            get_data_point(tmp_node.tmp_attribute, inst,
                           params->dataset_info->nr_instances, params->dataset);
      }

      AtomicAdd(&curr_dist[params->dataset_info->nr_target_values +
                           s_offsets[threadIdx.x]],
                T(1));
      AtomicAdd(&curr_dist[s_offsets[threadIdx.x]], T(-1));
    }

    __syncthreads();

    if (!skip) {
      for (int ii = threadIdx.x; ii >= 0; --ii) {
        ++local_dist[params->dataset_info->nr_target_values + s_offsets[ii]];
        --local_dist[s_offsets[ii]];
      }

      response = prior - entropy_conditioned_on_rows(
                             local_dist, att_type,
                             params->dataset_info->nr_target_values);
      if (bestResponse < response && (preVal < val || threadIdx.x + i == 0)) {
        bestResponse = response;
        bestI = i + threadIdx.x;
      }

      for (int ii = 0;
           ii < params->dataset_info->nr_target_values * max_nominal_; ++ii)
        local_dist[ii] = curr_dist[ii];
    }

    __syncthreads();
  }

  T* responses = (T*)s_offsets;
  responses[threadIdx.x] = bestResponse;
  s_offsets[threadIdx.x + blockDim.x] = bestI;

  for (int i = threadIdx.x;
       i < params->dataset_info->nr_target_values * max_nominal_;
       i += blockDim.x)
    curr_dist[i] = 0;

  __syncthreads();

  for (int i = blockDim.x >> 1; i > 0; i >>= 1) {
    if (threadIdx.x < i) {
      if (responses[i + threadIdx.x] > responses[threadIdx.x]) {
        responses[threadIdx.x] = responses[i + threadIdx.x];
        s_offsets[blockDim.x + threadIdx.x] =
            s_offsets[blockDim.x + threadIdx.x + i];
      }
    }

    __syncthreads();
  }

  if (threadIdx.x == 0) {
    bestI = s_offsets[blockDim.x];
    bestResponse = responses[threadIdx.x];

    if (bestI > 0) {
      T pointBeforeSplit = 0.0f, pointAfterSplit = 0.0f;
      int instJustBeforeSplit, instJustAfterSplit;

      instJustBeforeSplit =
          params->indices_buffer[params->iteration_info->tick_tock ? 0 : 1]
                                [nodeIndStart + bestI - 1];
      instJustAfterSplit =
          params->indices_buffer[params->iteration_info->tick_tock ? 0 : 1]
                                [nodeIndStart + bestI];
      pointBeforeSplit =
          get_data_point(tmp_node.tmp_attribute, instJustBeforeSplit,
                         params->dataset_info->nr_instances, params->dataset);
      pointAfterSplit =
          get_data_point(tmp_node.tmp_attribute, instJustAfterSplit,
                         params->dataset_info->nr_instances, params->dataset);

      tmp_node.tmp_split = (pointAfterSplit + pointBeforeSplit) / 2.0f;
    } else
      tmp_node.tmp_split = 0;
  }

  __syncthreads();

  // Assemble new distribution
  for (int i = threadIdx.x; i < numInds; i += blockDim.x) {
    inst = params->indices_buffer[params->iteration_info->tick_tock ? 0 : 1]
                                 [nodeIndStart + i];
    val = get_data_point(tmp_node.tmp_attribute, inst,
                         params->dataset_info->nr_instances, params->dataset);

    if (val != -flt_max)
      AtomicAdd(&curr_dist[params->dataset_info->nr_target_values *
                               ((val < tmp_node.tmp_split) ? 0 : 1) +
                           int(params->target_data[inst])],
                T(1));
    else
      AtomicAdd(&curr_dist[int(params->target_data[inst])], T(1));
  }

  return bestResponse;
}

template <typename T>
__device__ T GpuRf<T>::variance_calculation(
    GpuDteAlgorithmShared::GpuParams<T>* params,
    GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T>& node,
    GpuDteAlgorithmShared::gpuDTE_TmpNodeValues<T>& tmp_node, T* curr_dist,
    T* s_histograms) {
  int numInds = node.node_index_count;
  int nodeIndStart = node.node_index_start;
  int attribute = tmp_node.tmp_attribute;
  int inst, bestI = 0;
  T val, means[2], bestGain = 0.0f, preVal;
  means[0] = 0;

  if (threadIdx.x == 0) curr_dist[0] = 0.0f;

  // Calculate mean values from split
  for (int i = threadIdx.x; i < numInds; i += blockDim.x) {
    inst = params->indices_buffer[params->iteration_info->tick_tock ? 0 : 1]
                                 [nodeIndStart + i];
    val = get_data_point(attribute, inst, params->dataset_info->nr_instances,
                         params->dataset);

    if (val != -flt_max) AtomicAdd(&curr_dist[1], params->target_data[inst]);
  }

  __syncthreads();
  means[1] = curr_dist[1];

  T gain = 0;
  bool skip;
  for (int i = 0; i < numInds; i += blockDim.x) {
    skip = threadIdx.x + i >= numInds;
    if (!skip) {
      inst = params->indices_buffer[params->iteration_info->tick_tock ? 0 : 1]
                                   [nodeIndStart + i + threadIdx.x];
      val = get_data_point(attribute, inst, params->dataset_info->nr_instances,
                           params->dataset);
      s_histograms[threadIdx.x] = val;

      if (i + threadIdx.x != 0) {
        inst = params->indices_buffer[params->iteration_info->tick_tock ? 0 : 1]
                                     [nodeIndStart + i + threadIdx.x - 1];
        preVal =
            get_data_point(tmp_node.tmp_attribute, inst,
                           params->dataset_info->nr_instances, params->dataset);
      }

      AtomicAdd(&curr_dist[0], val);
      AtomicAdd(&curr_dist[1], -val);
    }

    __syncthreads();

    if (!skip) {
      for (int ii = threadIdx.x; ii >= 0; --ii) {
        means[0] += s_histograms[ii];
        means[1] -= s_histograms[ii];
      }

      T nLeft = i + threadIdx.x + 1;
      T nRight = numInds - nLeft;
      if (nRight < 1) nRight = 1;

      T diff = ((means[0] / nLeft) - (means[1] / nRight));
      gain = (nLeft * nRight * diff * diff / (nLeft + nRight));
      if (bestGain < gain && (preVal < val || threadIdx.x + i == 0)) {
        bestGain = gain;
        bestI = i + threadIdx.x;
      }

      means[0] = curr_dist[0];
      means[1] = curr_dist[1];
    }

    __syncthreads();
  }

  T* responses = (T*)s_histograms;
  responses[threadIdx.x] = bestGain;
  s_histograms[threadIdx.x + blockDim.x] = bestI;

  __syncthreads();

  for (int i = blockDim.x >> 1; i > 0; i >>= 1) {
    if (threadIdx.x < i) {
      if (responses[i + threadIdx.x] > responses[threadIdx.x]) {
        responses[threadIdx.x] = responses[i + threadIdx.x];
        s_histograms[blockDim.x + threadIdx.x] =
            s_histograms[blockDim.x + threadIdx.x + i];
      }
    }

    __syncthreads();
  }

  if (threadIdx.x == 0) {
    bestI = s_histograms[blockDim.x];
    bestGain = responses[threadIdx.x];

    if (bestI > 0) {
      T pointBeforeSplit = 0.0f, pointAfterSplit = 0.0f;
      int instJustBeforeSplit, instJustAfterSplit;

      instJustBeforeSplit =
          params->indices_buffer[params->iteration_info->tick_tock ? 0 : 1]
                                [nodeIndStart + bestI - 1];
      instJustAfterSplit =
          params->indices_buffer[params->iteration_info->tick_tock ? 0 : 1]
                                [nodeIndStart + bestI];
      pointBeforeSplit =
          get_data_point(tmp_node.tmp_attribute, instJustBeforeSplit,
                         params->dataset_info->nr_instances, params->dataset);
      pointAfterSplit =
          get_data_point(tmp_node.tmp_attribute, instJustAfterSplit,
                         params->dataset_info->nr_instances, params->dataset);

      tmp_node.tmp_split = (pointAfterSplit + pointBeforeSplit) / 2.0f;
    } else
      tmp_node.tmp_split = 0;
  }

  if (threadIdx.x < 4) curr_dist[threadIdx.x] = 0.0f;

  __syncthreads();

  for (int i = threadIdx.x; i < numInds; i += blockDim.x) {
    inst = params->indices_buffer[params->iteration_info->tick_tock ? 0 : 1]
                                 [nodeIndStart + i];
    val = get_data_point(attribute, inst, params->dataset_info->nr_instances,
                         params->dataset);

    if (val != -flt_max) {
      AtomicAdd(&curr_dist[(val < tmp_node.tmp_split) ? 0 : 1], T(1));
      AtomicAdd(&curr_dist[(val < tmp_node.tmp_split) ? 2 : 3], val);
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    curr_dist[2] = (curr_dist[0] != 0) ? curr_dist[2] / curr_dist[0] : 0;
    curr_dist[3] = (curr_dist[1] != 0) ? curr_dist[3] / curr_dist[1] : 0;
  }

  return bestGain;
}

template GpuRf<float>::GpuRf();
template GpuRf<double>::GpuRf();
}