#define DLLExport
#define TestExport

#include "gpurf.h"

#include "../../lib_algorithms/include/lib_algorithms.h"
#include "../../lib_algorithms/include/ml_algorithm_params.h"
#include "../../lib_core/include/lib_core.h"
#include "../../lib_data/include/lib_data.h"
#include "../../lib_gpu/include/cuda_kernel_helpers.hpp"
#include "../../lib_gpu/include/lib_gpu.h"
#include "../../lib_gpu/source/gpu_device_cuda.h"
#include "../../lib_models/include/lib_models.h"
#include "../../source_shared/include/lock_free_list.hpp"

namespace lib_algorithms {
template <typename T>
__global__ void host_kernel(GpuRf<T>* gpu_algo,
                            GpuRfStatic::GpuParams<T>* params,
                            GpuRfStatic::GpuRfKernelId type) {
  switch (type) {
    case GpuRfStatic::kSetupKernel:
      gpu_algo->gpurf_setup_kernel(params);
      break;
    case GpuRfStatic::kInitTreeBatch:
      gpu_algo->gpurf_initialize_tree_batch(params);
      break;
    case GpuRfStatic::kFindSplit:
      gpu_algo->gpurf_find_split(params);
      break;
    case GpuRfStatic::kPerformSplit:
      gpu_algo->gpurf_perform_split(params);
      break;
    case GpuRfStatic::kPredict:
      gpu_algo->gpurf_predict(params);
      break;
    case GpuRfStatic::kOobEstimate:
      gpu_algo->gpurf_oob_estimate(params);
      break;
    case GpuRfStatic::kFeatureImp:
      gpu_algo->gpurf_feature_importance(params);
      break;
    default:
      break;
  }
}

template <typename T>
GpuRf<T>::GpuRf() {}

template <typename T>
GpuRf<T>::~GpuRf() {}

template <typename T>
sp<lib_models::MlModel> GpuRf<T>::Fit(sp<lib_data::MlDataFrame<T>> data,
                                      sp<MlAlgorithmParams> params) {
  auto model = ModelsLib::GetInstance().CreateModel();
  auto gpu_params = CreateParams();
  auto nr_total_trees = params->Get<int>(AlgorithmsLib::kNrTrees);
  auto algo_type =
      params->Get<AlgorithmsLib::AlgorithmType>(AlgorithmsLib::kAlgoType);
  auto nr_samples = data->GetNrSamples();
  auto nr_features = data->GetNrFeatures();
  auto nr_targets = data->GetNrTargets();
  auto nr_nodes_per_batch = 1024;
  auto nr_threads_in_block = 64;
  int trees_built = 0;
  auto& device = GpuLib::GetInstance().GetGpuDevice();

  model->Add(ModelsLib::kNrTrees, nr_total_trees);
  model->Add(ModelsLib::kNrFeatures, nr_features);
  model->Add(ModelsLib::kNrTargets, nr_targets);
  model->Add(ModelsLib::kModelType, algo_type);

  GpuRfStatic::gpuDTE_NodeHeader_Classify<T> tmp_node;
  col_array<GpuRfStatic::gpuDTE_NodeHeader_Classify<T>> tree_nodes;
  col_array<T> tree_probabilities;

  auto barrier = CoreLib::GetInstance().CreateBarrier(2);
  bool run_rec_func = true;
  sutil::LockFreeList<std::pair<int, int>> job_list;
  auto data_rec_func = [&]() {
    int prob_id = 0;
    std::map<int, int> track_map;
    barrier->Wait();
    do {
      auto pair = job_list.pop_front();
      for (int i = 0; i < pair->first; ++i) {
        auto& gpu_node = gpu_params->node_buffers[pair->second][i];
        auto itr = track_map.find(gpu_node.parent_id);
        if (gpu_node.parent_id >= 0 && itr != track_map.end()) {
          tree_nodes[itr->second].child_start = int(tree_nodes.size());
          track_map.erase(itr->first);
        }

        tmp_node.child_count = gpu_node.attribute == -1 ? 0 : 2;
        tmp_node.attribute = gpu_node.attribute;
        tmp_node.split_point = gpu_node.split_point;

        tmp_node.probability_start = int(tree_probabilities.size());
        for (int ii = 0; ii < nr_targets * max_nominal_; ++ii)
          tree_probabilities.emplace_back(
              gpu_params
                  ->probability_buffers[prob_id]
                                       [i * nr_targets * max_nominal_ + ii]);

        track_map[gpu_node.trackinid] = int(tree_nodes.size());
        tree_nodes.emplace_back(tmp_node);
      }
      prob_id = prob_id == 0 ? 1 : 0;
      barrier->Wait();
    } while (run_rec_func);
  };
  sp<std::thread> data_rec_thread =
      std::make_shared<std::thread>(data_rec_func);

  AllocateFit(params, gpu_params, data);

  col_array<col_array<GpuRfStatic::gpuDTE_NodeHeader_Train<T>>> node_cache(
      2, col_array<GpuRfStatic::gpuDTE_NodeHeader_Train<T>>());
  while (nr_total_trees > 0) {
    int nodes_pulled;
    gpu_params->iteration_info->depth = 0;
    gpu_params->iteration_info->read_buffer_id = 0;
    gpu_params->iteration_info->write_buffer_id = 1;
    gpu_params->iteration_info->prob_buffer_id = 0;
    int stream_buffer = 2;
    gpu_params->iteration_info->tick_tock = true;

    int trees_launched = nr_total_trees > nr_nodes_per_batch
                             ? nr_nodes_per_batch
                             : nr_total_trees;
    nr_total_trees -= trees_launched;
    int nodes_left = trees_launched;
    int layer_id = 0;
    col_array<int> buffer_counts(3, 0);
    buffer_counts[gpu_params->iteration_info->read_buffer_id] = nodes_left;

    gpu_params->iteration_info->threads_launched = nr_nodes_per_batch;
    device.SynchronizeDevice(0);
    host_kernel<
        T><<<nr_nodes_per_batch / nr_threads_in_block, nr_threads_in_block>>>(
        this, gpu_params, GpuRfStatic::kSetupKernel);
    device.SynchronizeDevice(0);
    gpu_params->iteration_info->threads_launched = trees_launched;
    device.SynchronizeDevice(0);

    host_kernel<T><<<trees_launched, nr_threads_in_block>>>(
        this, gpu_params, GpuRfStatic::kInitTreeBatch);
    device.SynchronizeDevice(0);

    // Build trees
    do {
      bool swap_next = false;
      // Build node layer
      do {
        int nodes_launched = nodes_left > nr_nodes_per_batch / max_nominal_
                                 ? nr_nodes_per_batch / max_nominal_
                                 : nodes_left;
        nodes_left -= nodes_launched;
        gpu_params->iteration_info->threads_launched = nodes_launched;
        device.SynchronizeDevice(0);

        host_kernel<T><<<nodes_launched, nr_threads_in_block>>>(
            this, gpu_params, GpuRfStatic::kFindSplit);
        device.SynchronizeDevice(0);

        host_kernel<T><<<nodes_launched, nr_threads_in_block>>>(
            this, gpu_params, GpuRfStatic::kPerformSplit);
        device.SynchronizeDevice(0);

        gpu_params->iteration_info->node_offset += nodes_launched;
        buffer_counts[gpu_params->iteration_info->write_buffer_id] =
            gpu_params->node_cursors[new_nodes_];

        // Swap write buffer
        if (swap_next) {
          gpu_params->iteration_info->node_offset = 0;
          SwapBuffers(&gpu_params->iteration_info->read_buffer_id,
                      &stream_buffer);
          swap_next = false;

          // Stream partial layer results
          gpu_params->iteration_info->prob_buffer_id =
              gpu_params->iteration_info->prob_buffer_id == 0 ? 1 : 0;

          job_list.push_front(
              std::pair<int, int>(buffer_counts[stream_buffer], stream_buffer));
          buffer_counts[stream_buffer] = 0;
          barrier->Wait();
          nodes_left = nodes_pulled;
        } else if (!node_cache[layer_id].empty() &&
                   nodes_left - int(nr_nodes_per_batch / max_nominal_) <= 0) {
          nodes_pulled = nr_nodes_per_batch > node_cache[layer_id].size()
                             ? int(node_cache[layer_id].size())
                             : nr_nodes_per_batch;

          // Pre-stream next layer chunk for next iteration
          buffer_counts[stream_buffer] = nodes_pulled;
          StreamFromCache(stream_buffer, layer_id, node_cache, buffer_counts,
                          gpu_params->node_buffers[stream_buffer]);

          if (buffer_counts[gpu_params->iteration_info->write_buffer_id] > 0)
            StreamToCache(gpu_params->iteration_info->write_buffer_id, layer_id,
                          node_cache, buffer_counts,
                          gpu_params->node_buffers[gpu_params->iteration_info
                                                       ->write_buffer_id]);

          swap_next = true;
        }

        if (!swap_next) {
          // Stream nodes to the cache
          SwapBuffers(&gpu_params->iteration_info->write_buffer_id,
                      &stream_buffer);

          if (buffer_counts[stream_buffer] > 0)
            StreamToCache(stream_buffer, layer_id, node_cache, buffer_counts,
                          gpu_params->node_buffers[stream_buffer]);
        }

        // Update node counts on GPU
        gpu_params->node_cursors[work_cursor_] =
            gpu_params->node_cursors[new_nodes_] = 0;
      } while (nodes_left > 0);

      // Stream the last layer results
      gpu_params->iteration_info->prob_buffer_id =
          gpu_params->iteration_info->prob_buffer_id == 0 ? 1 : 0;

      job_list.push_front(std::pair<int, int>(
          buffer_counts[gpu_params->iteration_info->read_buffer_id],
          gpu_params->iteration_info->read_buffer_id));
      buffer_counts[gpu_params->iteration_info->read_buffer_id] = 0;
      barrier->Wait();

      // Prepare next layer
      layer_id = layer_id == 0 ? 1 : 0;
      if (!node_cache[layer_id].empty()) {
        nodes_left = nr_nodes_per_batch < node_cache[layer_id].size()
                         ? nr_nodes_per_batch
                         : int(node_cache[layer_id].size());
        buffer_counts[gpu_params->iteration_info->read_buffer_id] = nodes_left;
        StreamFromCache(
            gpu_params->iteration_info->read_buffer_id, layer_id, node_cache,
            buffer_counts,
            gpu_params
                ->node_buffers[gpu_params->iteration_info->read_buffer_id]);
      }

      ++gpu_params->iteration_info->depth;
      gpu_params->iteration_info->node_offset = 0;
      gpu_params->iteration_info->tick_tock =
          !gpu_params->iteration_info->tick_tock;
    } while (nodes_left > 0);

    trees_built += trees_launched;
  }

  run_rec_func = false;
  barrier->Wait();
  if (data_rec_thread->joinable()) data_rec_thread->join();
  FreeParams(&gpu_params);

  model->Add(ModelsLib::kNodeArray, tree_nodes);
  model->Add(ModelsLib::kProbArray, tree_probabilities);
  return model;
}

template <typename T>
sp<lib_data::MlResultData<T>> GpuRf<T>::Predict(
    sp<lib_data::MlDataFrame<T>> data, sp<lib_models::MlModel> model,
    sp<MlAlgorithmParams> params) {
  auto& device = GpuLib::GetInstance().GetGpuDevice();
  auto result_data = DataLib::GetInstance().CreateResultData<T>();
  auto gpu_params = CreateParams();
  int nr_samples = data->GetNrSamples();
  int block_size = block_size_;
  int max_blocks = max_blocks_;
  int nr_trees = model->Get<int>(ModelsLib::kNrTrees);
  int target_values = model->Get<int>(ModelsLib::kNrTargets);
  auto model_type =
      model->Get<AlgorithmsLib::AlgorithmType>(ModelsLib::kModelType);
  if (model_type == AlgorithmsLib::kRegression) target_values = 1;

  AllocatePredict(params, gpu_params, data, model);

  // Run prediction process
  {
    int launch_threads;
    int total_threads = nr_trees * nr_samples;

    while (total_threads > 0) {
      launch_threads = ceil(T(total_threads) / T(block_size)) > max_blocks
                           ? max_blocks * block_size
                           : total_threads;
      gpu_params->iteration_info->threads_launched = launch_threads;

      device.SynchronizeDevice(0);
      host_kernel<T><<<launch_threads / block_size, block_size>>>(
          this, gpu_params, GpuRfStatic::kPredict);
      device.SynchronizeDevice(0);

      gpu_params->iteration_info->tree_offset += launch_threads;
      total_threads -= launch_threads;
    }
  }

  // Fill out result buffers
  col_array<col_array<T>> predictions(nr_samples, col_array<T>());
  auto lambda_func = [&](int i) {
    if (model_type == AlgorithmsLib::kRegression)
      predictions[i].emplace_back(gpu_params->predictions[i]);
    else
      for (int ii = 0; ii < target_values; ++ii)
        predictions[i].emplace_back(
            gpu_params->predictions[i * target_values + ii]);
  };
  CoreLib::GetInstance().ParallelFor(0, nr_samples, lambda_func);

  FreeParams(&gpu_params);
  result_data->AddPredictions(predictions);
  return result_data;
}

template <typename T>
sp<lib_models::MlModel> GpuRf<T>::AggregateModels(
    col_array<sp<lib_models::MlModel>>) {
  return sp<lib_models::MlModel>();
}

template <typename T>
sp<lib_data::MlResultData<T>> GpuRf<T>::AggregateResults(
    col_array<sp<lib_data::MlResultData<T>>> results) {
  return sp<lib_data::MlResultData<T>>();
}

template <typename T>
col_array<sp<MlAlgorithmParams>> GpuRf<T>::SplitParameterPack(
    sp<MlAlgorithmParams> params, const int parts) {
  col_array<sp<MlAlgorithmParams>> part_vec;
  for (int i = 0; i < parts; ++i) {
    part_vec.emplace_back(
        AlgorithmsInterface::GetInstance().CreateGpuRfParamPack());
    part_vec.back()->Set(
        AlgorithmsInterface::kNrTrees,
        params->Get<int>(AlgorithmsInterface::kNrTrees) / parts);
  }

  return part_vec;
}

template <typename T>
void GpuRf<T>::AllocateFit(sp<MlAlgorithmParams> params,
                           GpuRfStatic::GpuParams<T>* gpu_params,
                           sp<lib_data::MlDataFrame<T>> data) {
  auto nr_samples = data->GetNrSamples();
  auto nr_features = data->GetNrFeatures();
  auto nr_targets = data->GetNrTargets();
  auto nr_nodes_per_batch = 1024;
  auto nr_total_trees = params->Get<int>(AlgorithmsLib::kNrTrees);
  auto nr_total_trees_per_iteration =
      nr_total_trees > 100 ? 100 : nr_total_trees;

  // Allocate training buffers
  auto& data_samples = data->GetSamples();
  auto& data_targets = data->GetTargets();
  lib_gpu::GpuDeviceCuda::AllocateManagedMemory(
      &gpu_params->dataset, sizeof(T) * data_samples.size());
  for (int i = 0; i < data_samples.size(); ++i)
    gpu_params->dataset[i] = data_samples[i];

  lib_gpu::GpuDeviceCuda::AllocateManagedMemory(
      &gpu_params->target_data, sizeof(T) * data_targets.size());
  for (int i = 0; i < data_targets.size(); ++i)
    gpu_params->target_data[i] = data_targets[i];

  lib_gpu::GpuDeviceCuda::AllocateManagedMemory(&gpu_params->attribute_type,
                                                sizeof(int) * nr_features);
  for (int i = 0; i < nr_features; ++i) gpu_params->attribute_type[i] = 2;

  for (int i = 0; i < 3; ++i)
    lib_gpu::GpuDeviceCuda::AllocateManagedMemory(
        &gpu_params->node_buffers[i],
        sizeof(GpuRfStatic::gpuDTE_NodeHeader_Train<T>) * nr_nodes_per_batch);

  // Tree header buffer
  lib_gpu::GpuDeviceCuda::AllocateManagedMemory(
      &gpu_params->node_tmp_buffer,
      sizeof(GpuRfStatic::gpuDTE_TmpNodeValues<T>) * nr_nodes_per_batch);
  lib_gpu::GpuDeviceCuda::AllocateManagedMemory(
      &gpu_params->probability_buffers[0],
      sizeof(T) * nr_nodes_per_batch * nr_targets * max_nominal_);
  lib_gpu::GpuDeviceCuda::AllocateManagedMemory(
      &gpu_params->probability_buffers[1],
      sizeof(T) * nr_nodes_per_batch * nr_targets * max_nominal_);

  lib_gpu::GpuDeviceCuda::AllocateManagedMemory(
      &gpu_params->probability_tmp_buffer,
      sizeof(T) * nr_nodes_per_batch * nr_targets * max_nominal_);

  lib_gpu::GpuDeviceCuda::AllocateManagedMemory(
      &gpu_params->indices_buffer[0],
      sizeof(int) * nr_samples * nr_total_trees_per_iteration);
  lib_gpu::GpuDeviceCuda::AllocateManagedMemory(
      &gpu_params->indices_buffer[1],
      sizeof(int) * nr_samples * nr_total_trees_per_iteration);

  lib_gpu::GpuDeviceCuda::AllocateManagedMemory(&gpu_params->target_starts,
                                                sizeof(int) * nr_targets);

  lib_gpu::GpuDeviceCuda::AllocateManagedMemory(
      &gpu_params->indices_inbag,
      sizeof(bool) * nr_samples * nr_total_trees_per_iteration);

  lib_gpu::GpuDeviceCuda::AllocateManagedMemory(
      &gpu_params->dataset_info, sizeof(GpuRfStatic::gpuDTE_DatasetInfo));
  lib_gpu::GpuDeviceCuda::AllocateManagedMemory(
      &gpu_params->static_info, sizeof(GpuRfStatic::gpuDTE_StaticInfo));
  lib_gpu::GpuDeviceCuda::AllocateManagedMemory(
      &gpu_params->iteration_info, sizeof(GpuRfStatic::gpuDTE_IterationInfo));

  gpu_params->dataset_info->nr_attributes = nr_features;
  gpu_params->dataset_info->nr_instances = nr_samples;
  gpu_params->dataset_info->nr_target_values = nr_targets;
  gpu_params->dataset_info->data_type = 0;

  gpu_params->static_info->loaded_trees = nr_total_trees_per_iteration;
  gpu_params->static_info->total_trees = nr_total_trees;
  gpu_params->static_info->max_node_size =
      params->Get<int>(AlgorithmsLib::kMinNodeSize);
  gpu_params->static_info->max_node_depth =
      params->Get<int>(AlgorithmsLib::kMaxDepth);
  gpu_params->static_info->node_buffer_size = 1024;

  auto k = params->Get<int>(AlgorithmsLib::kNrFeatures);
  gpu_params->static_info->nr_features =
      k > 0 ? k : int(std::round(log(nr_features))) + 1;
  gpu_params->static_info->max_class_count = nr_targets;
  gpu_params->static_info->balanced_sampling =
      params->Get<bool>(AlgorithmsLib::kEasyEnsemble);

  gpu_params->iteration_info->read_buffer_id =
      gpu_params->iteration_info->write_buffer_id = 0;
  gpu_params->iteration_info->tick_tock = true;

  lib_gpu::GpuDeviceCuda::AllocateManagedMemory(
      &gpu_params->random_states,
      sizeof(curandStateMRG32k3a) * nr_nodes_per_batch);
  lib_gpu::GpuDeviceCuda::AllocateManagedMemory(&gpu_params->node_cursors,
                                                sizeof(int) * 3);
}

template <typename T>
void GpuRf<T>::AllocatePredict(sp<MlAlgorithmParams> params,
                               GpuRfStatic::GpuParams<T>* gpu_params,
                               sp<lib_data::MlDataFrame<T>> data,
                               sp<lib_models::MlModel> model) {
  int nr_targets = model->Get<int>(ModelsLib::kNrTargets);
  int nr_samples = data->GetNrSamples();
  int nr_trees = model->Get<int>(ModelsLib::kNrTrees);
  int nr_features = model->Get<int>(ModelsLib::kNrFeatures);
  auto model_type =
      model->Get<AlgorithmsLib::AlgorithmType>(ModelsLib::kModelType);
  if (model_type == AlgorithmsLib::kRegression) nr_targets = 1;

  // Allocate prediction buffers
  col_array<T> pred_init(nr_samples * nr_targets, 0);
  auto& node_headers =
      model->Get<col_array<GpuRfStatic::gpuDTE_NodeHeader_Classify<T>>>(
          ModelsLib::kNodeArray);
  auto& prob_data = model->Get<col_array<T>>(ModelsLib::kProbArray);

  lib_gpu::GpuDeviceCuda::AllocateManagedMemory(
      &gpu_params->node_buffers_classify,
      sizeof(GpuRfStatic::gpuDTE_NodeHeader_Classify<T>) *
          int(node_headers.size()));
  for (int i = 0; i < node_headers.size(); ++i)
    gpu_params->node_buffers_classify[i] = node_headers[i];

  lib_gpu::GpuDeviceCuda::AllocateManagedMemory(
      &gpu_params->probability_tmp_buffer, sizeof(T) * prob_data.size());
  for (int i = 0; i < prob_data.size(); ++i)
    gpu_params->probability_tmp_buffer[i] = prob_data[i];

  auto& data_samples = data->GetSamples();
  lib_gpu::GpuDeviceCuda::AllocateManagedMemory(
      &gpu_params->dataset, sizeof(T) * data_samples.size());
  for (int i = 0; i < data_samples.size(); ++i)
    gpu_params->dataset[i] = data_samples[i];

  lib_gpu::GpuDeviceCuda::AllocateManagedMemory(&gpu_params->attribute_type,
                                                sizeof(int) * nr_features);
  for (int i = 0; i < nr_features; ++i) gpu_params->attribute_type[i] = 2;

  lib_gpu::GpuDeviceCuda::AllocateManagedMemory(&gpu_params->predictions,
                                                sizeof(T) * pred_init.size());
  for (int i = 0; i < pred_init.size(); ++i)
    gpu_params->predictions[i] = pred_init[i];

  gpu_params->dataset_info->nr_attributes = nr_features;
  gpu_params->dataset_info->nr_instances = nr_samples;
  gpu_params->dataset_info->nr_target_values = nr_targets;
  gpu_params->dataset_info->data_type = model_type;

  gpu_params->static_info->loaded_trees = nr_trees;
  gpu_params->static_info->total_trees = nr_trees;
  gpu_params->static_info->max_node_size =
      params->Get<int>(AlgorithmsLib::kMinNodeSize);
  gpu_params->static_info->max_node_depth =
      params->Get<int>(AlgorithmsLib::kMaxDepth);

  auto k = params->Get<int>(AlgorithmsLib::kNrFeatures);
  gpu_params->static_info->nr_features =
      k > 0 ? k : int(std::round(log(nr_features))) + 1;
  gpu_params->static_info->max_class_count = nr_targets;
  gpu_params->static_info->balanced_sampling =
      params->Get<bool>(AlgorithmsLib::kEasyEnsemble);

  gpu_params->iteration_info->read_buffer_id = 0;
  gpu_params->iteration_info->tree_offset = 0;
}

template <typename T>
void GpuRf<T>::FreeParams(GpuRfStatic::GpuParams<T>** dev_ptr) {
  if (*dev_ptr) {
    (*dev_ptr)->finalize();
    lib_gpu::GpuDeviceCuda::DeallocateMemory(dev_ptr);
  }
}

template <typename T>
GpuRfStatic::GpuParams<T>* GpuRf<T>::CreateParams() {
  GpuRfStatic::GpuParams<T>* dev_ptr = nullptr;
  lib_gpu::GpuDeviceCuda::AllocateManagedMemory(
      &dev_ptr, sizeof(GpuRfStatic::GpuParams<T>));
  dev_ptr->init();
  return dev_ptr;
}

template <typename T>
void GpuRf<T>::SwapBuffers(int* lhs, int* rhs) {
  int tmp = *rhs;
  *rhs = *lhs;
  *lhs = tmp;
}

template <typename T>
void GpuRf<T>::StreamToCache(
    int src_id, int layer_id,
    col_array<col_array<GpuRfStatic::gpuDTE_NodeHeader_Train<T>>>& node_cache,
    col_array<int>& buffer_counts,
    GpuRfStatic::gpuDTE_NodeHeader_Train<T>* node_headers) {
  int nr_nodes = buffer_counts[src_id];
  buffer_counts[src_id] = 0;
  if (nr_nodes <= 0) return;

  // Add to cache
  int cache_id = layer_id == 0 ? 1 : 0;
  for (int i = 0; i < nr_nodes; ++i)
    node_cache[cache_id].emplace_back(node_headers[i]);
}

template <typename T>
void GpuRf<T>::StreamFromCache(
    int dst_id, int layer_id,
    col_array<col_array<GpuRfStatic::gpuDTE_NodeHeader_Train<T>>>& node_cache,
    col_array<int>& buffer_counts,
    GpuRfStatic::gpuDTE_NodeHeader_Train<T>* node_headers) {
  int nr_nodes = buffer_counts[dst_id];

  // Pre-stream next layer chunk for next iteration
  for (int i = 0; i < nr_nodes; ++i) node_headers[i] = node_cache[layer_id][i];

  if (node_cache[layer_id].size() - nr_nodes > 0)
    for (int i = 0; i < node_cache[layer_id].size() - nr_nodes; ++i)
      node_cache[layer_id][i] = node_cache[layer_id][nr_nodes + i];

  for (int i = 0; i < nr_nodes; ++i) node_cache[layer_id].pop_back();
}

template <typename T>
__device__ void GpuRf<T>::gpurf_setup_kernel(
    GpuRfStatic::GpuParams<T>* params) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  // Each thread gets same seed, a different sequence number, no offset
  if (id < params->iteration_info->threads_launched)
    curand_init(324123, id, 0, &params->random_states[id]);
}

template <typename T>
__device__ void GpuRf<T>::gpurf_initialize_tree_batch(
    GpuRfStatic::GpuParams<T>* params) {
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
        localCursor = GpuRfStatic::AtomicAdd(&s_indexCursor, 1);
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
    GpuRfStatic::gpuDTE_NodeHeader_Train<T> root;
    root.parent_id = -2;
    root.attribute = -2;
    root.split_point = -2;
    root.trackinid = GpuRfStatic::AtomicAdd(&params->node_cursors[node_id_], 1);
    root.node_index_start = treeOffset;
    root.node_index_count = s_indexCursor;

    params->node_buffers[params->iteration_info->read_buffer_id][blockIdx.x] =
        root;
  }
}

template <typename T>
__device__ void GpuRf<T>::gpurf_find_split(GpuRfStatic::GpuParams<T>* params) {
  __shared__ T s_dynamic_shared[40];
  __shared__ unsigned int s_histograms[1024];
  __shared__ unsigned int s_offsets[256];
  __shared__ GpuRfStatic::gpuDTE_NodeHeader_Train<T> s_tree_node;
  __shared__ GpuRfStatic::gpuDTE_TmpNodeValues<T> s_tmp_node;
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
        GpuRfStatic::AtomicAdd(&s_dynamic_shared[int(params->target_data[inst])], T(1));
        break;
      case type_regression_:
        GpuRfStatic::AtomicAdd(&s_dynamic_shared[0], params->target_data[inst]);
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

    float response;
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
    GpuRfStatic::GpuParams<T>* params) {
  extern __shared__ int s_node_counts[];

  gpudte_perform_split(
      *params->static_info, *params->dataset_info, *params->iteration_info,
      params->probability_buffers, params->probability_tmp_buffer,
      params->dataset, params->attribute_type, s_node_counts,
      params->indices_buffer, params->node_cursors, params->node_buffers);
}

template <typename T>
__device__ void GpuRf<T>::gpurf_predict(GpuRfStatic::GpuParams<T>* params) {
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
    GpuRfStatic::GpuParams<T>* params) {
  if (threadIdx.x + blockIdx.x * blockDim.x >=
      params->iteration_info->threads_launched)
    return;

  int tid = params->iteration_info->tree_offset + threadIdx.x +
            blockIdx.x * blockDim.x;
  int instanceId = tid % params->dataset_info->nr_instances;
  int treeId = float(tid) / float(params->dataset_info->nr_instances);
  bool inBag =
      params->indices_inbag[params->dataset_info->nr_instances * treeId +
                            instanceId];
  if (inBag) return;

  float dataPoint;
  GpuRfStatic::gpuDTE_NodeHeader_Classify<T> tree_node =
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
      float max_prob = 0;
      for (int i = 0; i < params->dataset_info->nr_target_values; ++i) {
        if (params->probability_tmp_buffer[tree_node.probability_start + i] >
            max_prob) {
          max_prob =
              params->probability_tmp_buffer[tree_node.probability_start + i];
          classSelector = i;
        }
      }

      if (params->target_data[instanceId] == classSelector)
        GpuRfStatic::AtomicAdd(&params->oobCounts[0], 1);
      else
        GpuRfStatic::AtomicAdd(&params->oobCounts[1], 1);
      break;
    }
    case type_regression_:
      GpuRfStatic::AtomicAdd(&params->mse[0],
                params->probability_tmp_buffer[tree_node.probability_start]);
      break;
  }
}

template <typename T>
__device__ void GpuRf<T>::gpurf_feature_importance(
    GpuRfStatic::GpuParams<T>* params) {
  if (threadIdx.x + blockIdx.x * blockDim.x >=
      params->iteration_info->threads_launched)
    return;

  int randAttribute = params->iteration_info->depth;
  int tid = params->iteration_info->tree_offset + threadIdx.x +
            blockIdx.x * blockDim.x;
  int instanceId = tid % params->dataset_info->nr_instances;
  int treeId = float(tid) / float(params->dataset_info->nr_instances);
  bool inBag =
      params->indices_inbag[params->dataset_info->nr_instances * treeId +
                            instanceId];
  if (inBag) return;

  int instance;
  curandStateMRG32k3a localState = params->random_states[blockIdx.x];
  float dataPoint;
  GpuRfStatic::gpuDTE_NodeHeader_Classify<T> tree_node =
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
      float max_prob = 0;
      for (int i = 0; i < params->dataset_info->nr_target_values; ++i) {
        if (params->probability_tmp_buffer[tree_node.probability_start + i] >
            max_prob) {
          max_prob =
              params->probability_tmp_buffer[tree_node.probability_start + i];
          classSelector = i;
        }
      }

      if (params->target_data[instanceId] == classSelector)
        GpuRfStatic::AtomicAdd(&params->oobCounts[randAttribute * 2], 1);
      else
        GpuRfStatic::AtomicAdd(&params->oobCounts[randAttribute * 2 + 1], 1);
      break;
    }
    case type_regression_:
      GpuRfStatic::AtomicAdd(&params->mse[randAttribute],
                params->probability_tmp_buffer[tree_node.probability_start]);
      break;
  }
}

template <typename T>
__device__ void GpuRf<T>::radix_sort_on_attribute(
    GpuRfStatic::GpuParams<T>* params,
    GpuRfStatic::gpuDTE_NodeHeader_Train<T>& node,
    GpuRfStatic::gpuDTE_TmpNodeValues<T>& tmp_node, unsigned s_histograms[1024],
    unsigned s_offsets[256]) {
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
    GpuRfStatic::AtomicAdd(&s_histograms[*dataVal], unsigned int(1));
    GpuRfStatic::AtomicAdd(&s_histograms[256 + (*(dataVal + 1))], unsigned int(1));
    GpuRfStatic::AtomicAdd(&s_histograms[512 + (*(dataVal + 2))], unsigned int(1));
    GpuRfStatic::AtomicAdd(&s_histograms[768 + (*(dataVal + 3))], unsigned int(1));
  }

  __syncthreads();

  for (int i = threadIdx.x + 128; i < 256; i += blockDim.x)
    GpuRfStatic::AtomicAdd(&s_nrNegativeValues, s_histograms[768 + i]);

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
            GpuRfStatic::AtomicAdd(&curCount[radix], unsigned int(1));
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
              GpuRfStatic::AtomicAdd((int*)&s_offsets[radix], 1);
            else
              GpuRfStatic::AtomicAdd((int*)&s_offsets[radix], -1);

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
    GpuRfStatic::GpuParams<T>* params,
    GpuRfStatic::gpuDTE_NodeHeader_Train<T>& node,
    GpuRfStatic::gpuDTE_TmpNodeValues<T>& tmp_node, T* curr_dist, int att_type,
    unsigned* s_histograms, unsigned* s_offsets) {
  int numInds = node.node_index_count;
  int nodeIndStart = node.node_index_start;
  int inst, bestI = 0;
  T val, preVal;
  T response, bestResponse = 0.0f;
  T* local_dist;

  local_dist =
      (T*)&s_histograms[threadIdx.x * params->dataset_info->nr_target_values *
                        max_nominal_];
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

      GpuRfStatic::AtomicAdd(&curr_dist[params->dataset_info->nr_target_values +
                           s_offsets[threadIdx.x]],
                T(1));
      GpuRfStatic::AtomicAdd(&curr_dist[s_offsets[threadIdx.x]], T(-1));
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
      GpuRfStatic::AtomicAdd(&curr_dist[params->dataset_info->nr_target_values *
                               ((val < tmp_node.tmp_split) ? 0 : 1) +
                           int(params->target_data[inst])],
                T(1));
    else
      GpuRfStatic::AtomicAdd(&curr_dist[int(params->target_data[inst])], T(1));
  }

  return bestResponse;
}

template <typename T>
__device__ T
GpuRf<T>::variance_calculation(GpuRfStatic::GpuParams<T>* params,
                               GpuRfStatic::gpuDTE_NodeHeader_Train<T>& node,
                               GpuRfStatic::gpuDTE_TmpNodeValues<T>& tmp_node,
                               T* curr_dist, T* s_histograms) {
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

    if (val != -flt_max) GpuRfStatic::AtomicAdd(&curr_dist[1], params->target_data[inst]);
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

      GpuRfStatic::AtomicAdd(&curr_dist[0], val);
      GpuRfStatic::AtomicAdd(&curr_dist[1], -val);
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
      GpuRfStatic::AtomicAdd(&curr_dist[(val < tmp_node.tmp_split) ? 0 : 1], T(1));
      GpuRfStatic::AtomicAdd(&curr_dist[(val < tmp_node.tmp_split) ? 2 : 3], val);
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    curr_dist[2] = (curr_dist[0] != 0) ? curr_dist[2] / curr_dist[0] : 0;
    curr_dist[3] = (curr_dist[1] != 0) ? curr_dist[3] / curr_dist[1] : 0;
  }

  return bestGain;
}

template <typename T>
void GpuRfStatic::GpuParams<T>::init() {
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
void GpuRfStatic::GpuParams<T>::finalize() {
  lib_gpu::GpuDeviceCuda::DeallocateMemory(&node_cursors);
  lib_gpu::GpuDeviceCuda::DeallocateMemory(&probability_buffers[0]);
  lib_gpu::GpuDeviceCuda::DeallocateMemory(&probability_buffers[1]);
  lib_gpu::GpuDeviceCuda::DeallocateMemory(&node_buffers[0]);
  lib_gpu::GpuDeviceCuda::DeallocateMemory(&node_buffers[1]);
  lib_gpu::GpuDeviceCuda::DeallocateMemory(&node_buffers[2]);
  lib_gpu::GpuDeviceCuda::DeallocateMemory(&node_buffers_classify);
  lib_gpu::GpuDeviceCuda::DeallocateMemory(&node_tmp_buffer);
  lib_gpu::GpuDeviceCuda::DeallocateMemory(&probability_tmp_buffer);
  lib_gpu::GpuDeviceCuda::DeallocateMemory(&indices_buffer[0]);
  lib_gpu::GpuDeviceCuda::DeallocateMemory(&indices_buffer[1]);
  lib_gpu::GpuDeviceCuda::DeallocateMemory(&indices_inbag);
  lib_gpu::GpuDeviceCuda::DeallocateMemory(&target_starts);
  lib_gpu::GpuDeviceCuda::DeallocateMemory(&random_states);
  lib_gpu::GpuDeviceCuda::DeallocateMemory(&dataset);
  lib_gpu::GpuDeviceCuda::DeallocateMemory(&target_data);
  lib_gpu::GpuDeviceCuda::DeallocateMemory(&attribute_type);
  lib_gpu::GpuDeviceCuda::DeallocateMemory(&predictions);
  lib_gpu::GpuDeviceCuda::DeallocateMemory(&oobCounts);
  lib_gpu::GpuDeviceCuda::DeallocateMemory(&mse);
}

template <typename T>
__device__ T GpuRf<T>::get_data_point(int attribute, int instance,
                                      int nr_instances, T* dataset) {
  return dataset[attribute * nr_instances + instance];
}

template <typename T>
__device__ T GpuRf<T>::lnFunc(T num) {
  if (num <= 1e-6) return 0;
  return num * log(num);
}

template <typename T>
__device__ T GpuRf<T>::entropy_conditioned_on_rows(T* matrix, int att_type,
                                                   int nr_targets) {
  int x = (att_type > max_nominal_) ? 2 : 0;
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
__device__ T GpuRf<T>::entropy_over_columns(T* matrix, int att_type,
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
__device__ T GpuRf<T>::evaluate_nominal_attribute(
    GpuRfStatic::gpuDTE_NodeHeader_Train<T>& node,
    GpuRfStatic::gpuDTE_TmpNodeValues<T>& tmp_node, T* curr_dist, int att_type,
    int nr_targets, bool tick_tock, int** indices_buffer, T* targer_data,
    int nr_instances, T* dataset) {
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
      GpuRfStatic::AtomicAdd(&curr_dist[nr_targets * int(val) + int(targer_data[inst])],
                weight);
    else
      GpuRfStatic::AtomicAdd(&curr_dist[int(targer_data[inst])], weight);
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
__device__ void GpuRf<T>::gpudte_perform_split(
    GpuRfStatic::gpuDTE_StaticInfo& static_info,
    GpuRfStatic::gpuDTE_DatasetInfo& dataset_info,
    GpuRfStatic::gpuDTE_IterationInfo& iteration_info, T** probability_buffers,
    T* probability_tmp, T* dataset, int* attribute_types, int* node_counts,
    int** indices_buffers, int* node_cursors,
    GpuRfStatic::gpuDTE_NodeHeader_Train<T>** node_buffers) {
  __shared__ GpuRfStatic::gpuDTE_NodeHeader_Train<T> s_tree_node;
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
          ind_position = GpuRfStatic::AtomicAdd(&node_counts[0], 1);
        else
          ind_position = GpuRfStatic::AtomicAdd(&node_counts[1], 1);
        break;
      case type_regression_:
        if (data_point < s_tree_node.split_point)
          ind_position = GpuRfStatic::AtomicAdd(&node_counts[0], 1);
        else
          ind_position = GpuRfStatic::AtomicAdd(&node_counts[1], 1);
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

    GpuRfStatic::gpuDTE_NodeHeader_Train<T> child;
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
      int child_start = GpuRfStatic::AtomicAdd(&node_cursors[work_cursor_], child_count);
      GpuRfStatic::AtomicAdd(&node_cursors[new_nodes_], child_count);

      index_offset = 0;
      child.node_index_start = s_tree_node.node_index_start;
      for (int i = 0; i < child_count; ++i) {
        child.trackinid = GpuRfStatic::AtomicAdd(&node_cursors[node_id_], 1);
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
__device__ void GpuRf<T>::gpudte_predict(
    int tid, int nr_instances, int data_type, int nr_targets,
    GpuRfStatic::gpuDTE_NodeHeader_Classify<T>* node_buffer, T* dataset,
    T* probability_buffer, T* predictions, int* attribute_types) {
  // Access pattern
  // instances*trees: 5*5 = 25
  // tid:			0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
  // instances:		0 1 2 3 4 0 1 2 3 4 0  1  2  3  4  0  1  2  3
  // trees:			0 0 0 0 0 1 1 1 1 1 2  2  2  2  2  3  3  3  3

  int instanceId = tid % nr_instances;
  int treeId = T(tid) / T(nr_instances);
  T dataPoint;

  GpuRfStatic::gpuDTE_NodeHeader_Classify<T> tree_node = node_buffer[treeId];
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
      GpuRfStatic::AtomicAdd(&predictions[instanceId * nr_targets + classSelector], add);
      break;
    }
    case type_regression_:
      T mean = probability_buffer[tree_node.probability_start];
      GpuRfStatic::AtomicAdd(&predictions[instanceId], mean);
      break;
  }
}

__device__ float GpuRfStatic::AtomicAdd(float* address, float value) {
  return atomicAdd(address, value);
}

__device__ int GpuRfStatic::AtomicAdd(int* address, int value) {
  return atomicAdd(address, value);
}

__device__ unsigned int GpuRfStatic::AtomicAdd(unsigned int* address,
                                               unsigned int value) {
  return atomicAdd(address, value);
}

__device__ double GpuRfStatic::AtomicAdd(double* address, double value) {
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

template GpuRf<float>::GpuRf();
template GpuRf<double>::GpuRf();
}