#define DLLExport
#define TestExport

#include "gpurf.h"

#include "../../lib_algorithms/include/lib_algorithms.h"
#include "../../lib_algorithms/include/ml_algorithm_params.h"
#include "../../lib_core/include/lib_core.h"
#include "../../lib_data/include/lib_data.h"
#include "../../lib_ensembles/include/lib_ensembles.h"
#include "../../lib_gpu/include/cuda_kernel_helpers.hpp"
#include "../../lib_gpu/include/lib_gpu.h"
#include "../../lib_gpu/source/gpu_device_cuda.h"
#include "../../lib_models/include/lib_models.h"
#include "../../source_shared/include/lock_free_list.hpp"

namespace lib_ensembles {
__constant__ GpuDte::gpuDTE_StaticInfo static_info;
__constant__ GpuDte::gpuDTE_DatasetInfo dataset_info;
__constant__ GpuDte::gpuDTE_IterationInfo iteration_info;

__host__ void cpy_iteration_info(GpuDte::gpuDTE_IterationInfo* info) {
  cudaError_t error = cudaMemcpyToSymbol(lib_ensembles::iteration_info, info,
                                         sizeof(GpuDte::gpuDTE_IterationInfo),
                                         0, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) CoreLib::GetInstance().ThrowException("Cuda error");
}

__host__ void cpy_data_static_info(GpuDte::gpuDTE_DatasetInfo* data,
                                   GpuDte::gpuDTE_StaticInfo* info) {
  cudaError_t error = cudaMemcpyToSymbol(lib_ensembles::dataset_info, data,
                                         sizeof(GpuDte::gpuDTE_DatasetInfo), 0,
                                         cudaMemcpyHostToDevice);
  if (error != cudaSuccess) CoreLib::GetInstance().ThrowException("Cuda error");
  error = cudaMemcpyToSymbol(lib_ensembles::static_info, info,
                             sizeof(GpuDte::gpuDTE_StaticInfo), 0,
                             cudaMemcpyHostToDevice);
  if (error != cudaSuccess) CoreLib::GetInstance().ThrowException("Cuda error");
}

template <typename T>
__global__ void host_kernel(GpuRf<T>* gpu_algo, GpuDte::GpuParams<T> params,
                            GpuRfStatic::GpuRfKernelId type) {
  params.static_info = &static_info;
  params.dataset_info = &dataset_info;
  params.iteration_info = &iteration_info;
  switch (type) {
    case GpuRfStatic::kSetupKernel:
      gpu_algo->gpurf_setup_kernel(&params);
      break;
    case GpuRfStatic::kInitTreeBatch:
      gpu_algo->gpurf_initialize_tree_batch(&params);
      break;
    case GpuRfStatic::kFindSplit:
      gpu_algo->gpurf_find_split(&params);
      break;
    case GpuRfStatic::kPerformSplit:
      gpu_algo->gpurf_perform_split(&params);
      break;
    case GpuRfStatic::kPredict:
      gpu_algo->gpurf_predict(&params);
      break;
    case GpuRfStatic::kOobEstimate:
      gpu_algo->gpurf_oob_estimate(&params);
      break;
    case GpuRfStatic::kFeatureImp:
      gpu_algo->gpurf_feature_importance(&params);
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
sp<lib_models::MlModel> GpuRf<T>::Fit(
    sp<lib_data::MlDataFrame<T>> data,
    sp<lib_algorithms::MlAlgorithmParams> params) {
  GpuDte::gpuDTE_StaticInfo static_info;
  GpuDte::gpuDTE_DatasetInfo dataset_info;
  GpuDte::gpuDTE_IterationInfo iteration_info;
  auto device = GpuLib::GetInstance().CreateGpuDevice(
      params->Get<int>(EnsemblesLib::kDevId));
  auto gpu_params = GpuDte::GpuParams<T>();
  auto nr_total_trees = params->Get<int>(EnsemblesLib::kNrTrees);
  auto algo_type =
      params->Get<AlgorithmsLib::AlgorithmType>(EnsemblesLib::kAlgoType);
  auto nr_samples = data->GetNrSamples();
  auto nr_features = data->GetNrFeatures();
  auto nr_targets = data->GetNrTargets();
  int trees_built = 0;

  col_array<sp<lib_models::MlModel>> models;
  GpuDte::gpuDTE_NodeHeader_Classify<T> tmp_node;
  col_array<col_array<GpuDte::gpuDTE_NodeHeader_Classify<T>>> tree_nodes;
  col_array<col_array<T>> tree_probabilities;
  HostAllocFit host_alloc(device, nr_targets);

  // auto barrier = CoreLib::GetInstance().CreateBarrier(2);
  // bool run_rec_func = true;
  sutil::LockFreeList<std::pair<int, int>> job_list;
  col_map<int, int> track_map;
  int node_size = sizeof(GpuDte::gpuDTE_NodeHeader_Train<T>);
  int type_size = sizeof(T);
  int prob_id = 0;
  auto data_rec_func = [&]() {
    // device.SetDeviceForThread(params->Get<int>(EnsemblesLib::kDevId));
    // barrier->Wait();
    // do {
    auto pair = job_list.pop_front();
    device->CopyToHost(host_alloc.probability_cpy,
                       gpu_params.probability_buffers[prob_id],
                       type_size * pair->first * nr_targets,1);
    device->CopyToHost(host_alloc.node_cpy,
                       gpu_params.node_buffers[pair->second],
                       node_size * pair->first,1);
    device->SynchronizeDevice();
    for (int i = 0; i < pair->first; ++i) {
      auto& gpu_node = host_alloc.node_cpy[i];
      if (gpu_node.attribute != -1) {
        if (gpu_node.attribute >= nr_features || gpu_node.attribute < 0) {
          CoreLib::GetInstance().ThrowException("Faulty gpu node encountered.");
        }
      }

      auto itr = track_map.find(gpu_node.parent_id);
      if (gpu_node.parent_id >= 0 && itr != track_map.end()) {
        tree_nodes.back()[itr->second].child_start =
            int(tree_nodes.back().size());
        track_map.erase(itr->first);
      }

      tmp_node.child_count = gpu_node.attribute == -1 ? 0 : 2;
      tmp_node.attribute = gpu_node.attribute;
      tmp_node.split_point = gpu_node.split_point;

      if (tmp_node.child_count == 0) {
        tmp_node.probability_start = int(tree_probabilities.back().size());
        for (int ii = 0; ii < nr_targets; ++ii) {
          tree_probabilities.back().emplace_back(
              host_alloc.probability_cpy[i * nr_targets + ii]);
        }
      }

      if (tmp_node.child_count > 0)
        track_map[gpu_node.trackinid] = int(tree_nodes.back().size());
      tree_nodes.back().emplace_back(tmp_node);
    }
    prob_id = prob_id == 0 ? 1 : 0;
    // barrier->Wait();
    //} while (run_rec_func);
  };
  // sp<std::thread> data_rec_thread =
  //    std::make_shared<std::thread>(data_rec_func);

  AllocateFit(device, params, &gpu_params, data, static_info, dataset_info,
              iteration_info);

  col_array<col_array<GpuDte::gpuDTE_NodeHeader_Train<T>>> node_cache(
      2, col_array<GpuDte::gpuDTE_NodeHeader_Train<T>>());
  int nodes_pulled = 0;
  int stream_buffer = 2;
  int trees_left = nr_total_trees;
  while (trees_left > 0) {
    tree_probabilities.emplace_back(col_array<T>());
    tree_nodes.emplace_back(col_array<GpuDte::gpuDTE_NodeHeader_Classify<T>>());
    stream_buffer = 2;
    iteration_info.depth = 0;
    iteration_info.read_buffer_id = 0;
    iteration_info.write_buffer_id = 1;
    iteration_info.prob_buffer_id = 0;
    iteration_info.tick_tock = true;

    int trees_launched =
        trees_left > max_tree_batch_ ? max_tree_batch_ : trees_left;
    trees_left -= trees_launched;
    models.emplace_back(ModelsLib::GetInstance().CreateModel());
    models.back()->Add(ModelsLib::kNrTrees, trees_launched);
    models.back()->Add(ModelsLib::kNrFeatures, nr_features);
    models.back()->Add(ModelsLib::kNrTargets, nr_targets);
    models.back()->Add(ModelsLib::kModelType, algo_type);

    int nodes_left = trees_launched;
    int layer_id = 0;
    col_array<int> buffer_counts(3, 0);
    buffer_counts[iteration_info.read_buffer_id] = nodes_left;

    iteration_info.threads_launched = max_blocks_ * block_size_;
    cpy_iteration_info(&iteration_info);
    device->SynchronizeDevice();
    host_kernel<T><<<max_blocks_ / block_size_, block_size_>>>(
        this, gpu_params, GpuRfStatic::kSetupKernel);
    device->SynchronizeDevice();

    iteration_info.threads_launched = trees_launched;
    cpy_iteration_info(&iteration_info);

    host_kernel<T><<<trees_launched, block_size_>>>(
        this, gpu_params, GpuRfStatic::kInitTreeBatch);
    device->SynchronizeDevice();

    // Build trees
    do {
      bool swap_next = false;
      // Build node layer
      do {
        int nodes_launched = nodes_left > max_blocks_ / max_nominal_
                                 ? max_blocks_ / max_nominal_
                                 : nodes_left;

        nodes_left -= nodes_launched;
        iteration_info.threads_launched = nodes_launched;
        cpy_iteration_info(&iteration_info);

        host_kernel<T><<<nodes_launched, block_size_>>>(
            this, gpu_params, GpuRfStatic::kFindSplit);
        device->SynchronizeDevice();
        host_kernel<T><<<nodes_launched, block_size_>>>(
            this, gpu_params, GpuRfStatic::kPerformSplit);
        device->SynchronizeDevice();
        device->CopyToHost(host_alloc.node_cursor_cpy, gpu_params.node_cursors,
                           sizeof(int) * 3);
        device->SynchronizeDevice();

        iteration_info.node_offset += nodes_launched;
        buffer_counts[iteration_info.write_buffer_id] =
            host_alloc.node_cursor_cpy[new_nodes_];

        // Swap write buffer
        if (swap_next) {
          iteration_info.node_offset = 0;
          SwapBuffers(&iteration_info.read_buffer_id, &stream_buffer);
          swap_next = false;

          // Stream partial layer results
          iteration_info.prob_buffer_id =
              iteration_info.prob_buffer_id == 0 ? 1 : 0;

          job_list.push_front(
              std::pair<int, int>(buffer_counts[stream_buffer], stream_buffer));
          buffer_counts[stream_buffer] = 0;
          data_rec_func();
          // barrier->Wait();
          nodes_left = nodes_pulled;
        } else if (!node_cache[layer_id].empty() &&
                   nodes_left - int(max_blocks_ / max_nominal_) <= 0) {
          nodes_pulled = max_blocks_ > node_cache[layer_id].size()
                             ? int(node_cache[layer_id].size())
                             : max_blocks_;

          // Pre-stream next layer chunk for next iteration
          buffer_counts[stream_buffer] = nodes_pulled;
          StreamFromCache(device, host_alloc, stream_buffer, layer_id,
                          node_cache, buffer_counts,
                          gpu_params.node_buffers[stream_buffer]);

          if (buffer_counts[iteration_info.write_buffer_id] > 0)
            StreamToCache(
                device, host_alloc, iteration_info.write_buffer_id, layer_id,
                node_cache, buffer_counts,
                gpu_params.node_buffers[iteration_info.write_buffer_id]);

          swap_next = true;
        }

        if (!swap_next) {
          // Stream nodes to the cache
          SwapBuffers(&iteration_info.write_buffer_id, &stream_buffer);

          if (buffer_counts[stream_buffer] > 0)
            StreamToCache(device, host_alloc, stream_buffer, layer_id,
                          node_cache, buffer_counts,
                          gpu_params.node_buffers[stream_buffer]);
        }

        // Update node counts on GPU
        host_alloc.node_cursor_cpy[work_cursor_] =
            host_alloc.node_cursor_cpy[new_nodes_] = 0;
        device->CopyToDevice(host_alloc.node_cursor_cpy,
                             gpu_params.node_cursors, sizeof(int) * 3);
        device->SynchronizeDevice();
      } while (nodes_left > 0);

      // Stream the last layer results
      iteration_info.prob_buffer_id =
          iteration_info.prob_buffer_id == 0 ? 1 : 0;

      job_list.push_front(
          std::pair<int, int>(buffer_counts[iteration_info.read_buffer_id],
                              iteration_info.read_buffer_id));
      buffer_counts[iteration_info.read_buffer_id] = 0;
      data_rec_func();
      // barrier->Wait();

      // Prepare next layer
      layer_id = layer_id == 0 ? 1 : 0;
      if (!node_cache[layer_id].empty()) {
        nodes_left = max_blocks_ < node_cache[layer_id].size()
                         ? max_blocks_
                         : int(node_cache[layer_id].size());
        buffer_counts[iteration_info.read_buffer_id] = nodes_left;
        StreamFromCache(device, host_alloc, iteration_info.read_buffer_id,
                        layer_id, node_cache, buffer_counts,
                        gpu_params.node_buffers[iteration_info.read_buffer_id]);
      }

      ++iteration_info.depth;
      iteration_info.node_offset = 0;
      iteration_info.tick_tock = !iteration_info.tick_tock;
    } while (nodes_left > 0);

    trees_built += trees_launched;

    models.back()->Add(ModelsLib::kNodeArray, tree_nodes.back());
    models.back()->Add(ModelsLib::kProbArray, tree_probabilities.back());
  }

  // run_rec_func = false;
  // barrier->Wait();
  // if (data_rec_thread->joinable()) data_rec_thread->join();

  gpu_params.finalize(device);
  return AggregateModels(models);
}

template <typename T>
sp<lib_data::MlResultData<T>> GpuRf<T>::Predict(
    sp<lib_data::MlDataFrame<T>> data, sp<lib_models::MlModel> model,
    sp<lib_algorithms::MlAlgorithmParams> params) {
  GpuDte::gpuDTE_StaticInfo static_info;
  GpuDte::gpuDTE_DatasetInfo dataset_info;
  GpuDte::gpuDTE_IterationInfo iteration_info;
  auto device = GpuLib::GetInstance().CreateGpuDevice(
      params->Get<int>(EnsemblesLib::kDevId));
  auto result_data = DataLib::GetInstance().CreateResultData<T>();
  auto gpu_params = GpuDte::GpuParams<T>();
  int nr_samples = data->GetNrSamples();
  int nr_trees = model->Get<int>(ModelsLib::kNrTrees);
  int target_values = model->Get<int>(ModelsLib::kNrTargets);
  auto model_type =
      model->Get<AlgorithmsLib::AlgorithmType>(ModelsLib::kModelType);
  if (model_type == AlgorithmsLib::kRegression) target_values = 1;

  AllocatePredict(device, params, &gpu_params, data, model, static_info,
                  dataset_info, iteration_info);
  HostAllocPredict host_alloc(device, nr_samples, target_values);

  // Run prediction process
  {
    int launch_threads;
    int total_threads = nr_trees * nr_samples;

    while (total_threads > 0) {
      launch_threads = ceil(T(total_threads) / T(block_size_)) > max_blocks_
                           ? max_blocks_ * block_size_
                           : total_threads;
      iteration_info.threads_launched = launch_threads;
      cpy_iteration_info(&iteration_info);
      device->SynchronizeDevice();
      host_kernel<T><<<launch_threads / block_size_, block_size_>>>(
          this, gpu_params, GpuRfStatic::kPredict);
      device->SynchronizeDevice();

      iteration_info.tree_offset += launch_threads;
      total_threads -= launch_threads;
    }
  }

  // Fill out result buffers
  device->CopyToHost(host_alloc.predictions_cpy, gpu_params.predictions,
                     sizeof(T) * nr_samples * target_values);
  device->SynchronizeDevice();
  col_array<col_array<T>> predictions(nr_samples, col_array<T>());
  auto lambda_func = [&](int i) {
    if (model_type == AlgorithmsLib::kRegression)
      predictions[i].emplace_back(host_alloc.predictions_cpy[i]);
    else
      for (int ii = 0; ii < target_values; ++ii)
        predictions[i].emplace_back(
            host_alloc.predictions_cpy[i * target_values + ii]);
  };
  CoreLib::GetInstance().ParallelFor(0, nr_samples, lambda_func);

  gpu_params.finalize(device);
  result_data->AddPredictions(predictions);
  return result_data;
}

template <typename T>
sp<lib_models::MlModel> GpuRf<T>::AggregateModels(
    col_array<sp<lib_models::MlModel>> models) {
  if (models.size() < 2) return models[0];
  col_array<T> aggregate_prob;
  std::function<void(int)> rec_add_nodes;
  col_array<GpuDte::gpuDTE_NodeHeader_Classify<T>> aggregate_node;
  for (int i = 0; i < models.size(); ++i) {
    auto& node_headers =
        models[i]->Get<col_array<GpuDte::gpuDTE_NodeHeader_Classify<T>>>(
            ModelsLib::kNodeArray);
    auto& prob_data = models[i]->Get<col_array<T>>(ModelsLib::kProbArray);
    auto trees = models[i]->Get<int>(ModelsLib::kNrTrees);
    auto targets = models[i]->Get<int>(ModelsLib::kNrTargets);
    for (int ii = 0; ii < trees; ++ii) {
      aggregate_node.emplace_back(node_headers[ii]);
      if (aggregate_node.back().child_count <= 0) {
        for (int iii = 0; iii < targets; ++iii)
          aggregate_prob.emplace_back(
              prob_data[node_headers[ii].probability_start + iii]);
      }
    }
  }

  auto trees_agg = 0;
  for (int i = 0; i < models.size(); ++i) {
    auto& node_headers =
        models[i]->Get<col_array<GpuDte::gpuDTE_NodeHeader_Classify<T>>>(
            ModelsLib::kNodeArray);
    auto& prob_data = models[i]->Get<col_array<T>>(ModelsLib::kProbArray);
    auto trees = models[i]->Get<int>(ModelsLib::kNrTrees);
    auto targets = models[i]->Get<int>(ModelsLib::kNrTargets);
    rec_add_nodes = [&](int node_id) {
      if (aggregate_node[node_id].child_count > 0) {
        int child_start = int(aggregate_node.size());
        for (int i = 0; i < aggregate_node[node_id].child_count; ++i) {
          aggregate_node.emplace_back(
              node_headers[aggregate_node[node_id].child_start + i]);
        }
        aggregate_node[node_id].child_start = child_start;
        for (int i = 0; i < aggregate_node[node_id].child_count; ++i)
          rec_add_nodes(child_start + i);
      } else {
        int prob_start = int(aggregate_prob.size());
        for (int ii = 0; ii < targets; ++ii)
          aggregate_prob.emplace_back(
              prob_data[aggregate_node[node_id].probability_start + ii]);
        aggregate_node[node_id].probability_start = prob_start;
      }
    };

    for (int ii = 0; ii < trees; ++ii) rec_add_nodes(trees_agg + ii);
    trees_agg += trees;
  }

  models[0]->Add(ModelsLib::kNrTrees, trees_agg);
  models[0]->Add(ModelsLib::kNodeArray, aggregate_node);
  models[0]->Add(ModelsLib::kProbArray, aggregate_prob);
  return models[0];
}

template <typename T>
col_array<sp<lib_models::MlModel>> GpuRf<T>::SplitModel(
    sp<lib_models::MlModel> model, const int parts) {
  auto trees = model->Get<int>(ModelsLib::kNrTrees);
  auto targets = model->Get<int>(ModelsLib::kNrTargets);
  col_array<sp<lib_models::MlModel>> models(
      parts, ModelsLib::GetInstance().CreateModel());
  auto& node_headers =
      model->Get<col_array<GpuDte::gpuDTE_NodeHeader_Classify<T>>>(
          ModelsLib::kNodeArray);
  auto& prob_data = model->Get<col_array<T>>(ModelsLib::kProbArray);

  std::function<void(int)> rec_add_nodes;
  auto tree_split = trees / parts;
  for (int i = 0; i < parts; ++i) {
    col_array<T> prob_array;
    col_array<GpuDte::gpuDTE_NodeHeader_Classify<T>> node_array;

    rec_add_nodes = [&](int node_id) {
      if (node_array[node_id].child_count > 0) {
        int child_start = int(node_array.size());
        for (int i = 0; i < node_array[node_id].child_count; ++i) {
          node_array.emplace_back(
              node_headers[node_array[node_id].child_start + i]);
        }
        node_array[node_id].child_start = child_start;
        for (int i = 0; i < node_array[node_id].child_count; ++i)
          rec_add_nodes(child_start + i);
      } else {
        int prob_start = int(prob_array.size());
        for (int ii = 0; ii < targets; ++ii)
          prob_array.emplace_back(
              prob_data[node_array[node_id].probability_start + ii]);
        node_array[node_id].probability_start = prob_start;
      }
    };

    models[i]->Add(ModelsLib::kNrTrees, tree_split);
    models[i]->Add(ModelsLib::kNrTargets, targets);
    models[i]->Add(ModelsLib::kNrFeatures,
                   model->Get<int>(ModelsLib::kNrFeatures));
    models[i]->Add(
        ModelsLib::kModelType,
        model->Get<AlgorithmsLib::AlgorithmType>(ModelsLib::kModelType));

    auto tree_offset = tree_split * i;
    for (int ii = 0; ii < tree_split; ++ii) {
      node_array.emplace_back(node_headers[tree_offset + ii]);
      if (node_array.back().child_count <= 0) {
        int prob_start = int(prob_array.size());
        for (int iii = 0; iii < targets; ++iii)
          prob_array.emplace_back(
              prob_data[node_headers[tree_offset + ii].probability_start +
                        iii]);
        node_array.back().probability_start = prob_start;
      }
    }

    for (int ii = 0; ii < tree_split; ++ii) rec_add_nodes(ii);

    models[i]->Add(ModelsLib::kNodeArray, node_array);
    models[i]->Add(ModelsLib::kProbArray, prob_array);
  }
  return models;
}

template <typename T>
sp<lib_data::MlResultData<T>> GpuRf<T>::AggregateResults(
    col_array<sp<lib_data::MlResultData<T>>> results) {
  for (int i = 1; i < results.size(); ++i) *results[0] += *results[i];
  return results[0];
}

template <typename T>
col_array<sp<lib_algorithms::MlAlgorithmParams>> GpuRf<T>::SplitParameterPack(
    sp<lib_algorithms::MlAlgorithmParams> params, const int parts) {
  col_array<sp<lib_algorithms::MlAlgorithmParams>> part_vec;
  for (int i = 0; i < parts; ++i) {
    part_vec.emplace_back(EnsemblesLib::GetInstance().CreateGpuRfParamPack());
    part_vec.back()->Set(EnsemblesLib::kNrTrees,
                         params->Get<int>(EnsemblesLib::kNrTrees) / parts);
  }

  return part_vec;
}

template <typename T>
void GpuRf<T>::AllocateFit(sp<lib_gpu::GpuDevice> dev,
                           sp<lib_algorithms::MlAlgorithmParams> params,
                           GpuDte::GpuParams<T>* gpu_params,
                           sp<lib_data::MlDataFrame<T>> data,
                           GpuDte::gpuDTE_StaticInfo& static_info,
                           GpuDte::gpuDTE_DatasetInfo& dataset_info,
                           GpuDte::gpuDTE_IterationInfo& iteration_info) {
  auto nr_samples = data->GetNrSamples();
  auto nr_features = data->GetNrFeatures();
  auto nr_targets = data->GetNrTargets();
  auto nr_total_trees = params->Get<int>(EnsemblesLib::kNrTrees);

  // Allocate training buffers
  auto& data_samples = data->GetSamples();
  auto& data_targets = data->GetTargets();

  col_array<std::pair<void**, size_t>> mem_offsets;
  for (int i = 0; i < 3; ++i) {
    mem_offsets.emplace_back(std::pair<void**, size_t>(
        (void**)&gpu_params->node_buffers[i],
        sizeof(GpuDte::gpuDTE_NodeHeader_Train<T>) * max_blocks_));
  }
  for (int i = 0; i < 2; ++i) {
    mem_offsets.emplace_back(
        std::pair<void**, size_t>((void**)&gpu_params->indices_buffer[i],
                                  sizeof(int) * nr_samples * max_tree_batch_));
    mem_offsets.emplace_back(std::pair<void**, size_t>(
        (void**)&gpu_params->probability_buffers[i],
        sizeof(T) * max_blocks_ * nr_targets * max_nominal_));
  }
  mem_offsets.emplace_back(std::pair<void**, size_t>(
      (void**)&gpu_params->node_cursors, sizeof(int) * 3));
  mem_offsets.emplace_back(std::pair<void**, size_t>(
      (void**)&gpu_params->node_tmp_buffer,
      sizeof(GpuDte::gpuDTE_TmpNodeValues<T>) * max_blocks_));
  mem_offsets.emplace_back(std::pair<void**, size_t>(
      (void**)&gpu_params->probability_tmp_buffer,
      sizeof(T) * max_blocks_ * nr_targets * max_nominal_));
  mem_offsets.emplace_back(std::pair<void**, size_t>(
      (void**)&gpu_params->target_starts, sizeof(int) * nr_targets));
  mem_offsets.emplace_back(
      std::pair<void**, size_t>((void**)&gpu_params->indices_inbag,
                                sizeof(bool) * nr_samples * max_blocks_));
  mem_offsets.emplace_back(
      std::pair<void**, size_t>((void**)&gpu_params->random_states,
                                sizeof(curandStateMRG32k3a) * max_blocks_));
  mem_offsets.emplace_back(std::pair<void**, size_t>(
      (void**)&gpu_params->attribute_type, sizeof(int) * nr_features));
  mem_offsets.emplace_back(std::pair<void**, size_t>(
      (void**)&gpu_params->target_data, sizeof(T) * data_targets.size()));
  mem_offsets.emplace_back(std::pair<void**, size_t>(
      (void**)&gpu_params->dataset, sizeof(T) * data_samples.size()));

  for (int i = 0; i < mem_offsets.size(); ++i)
    dev->AllocateMemory(mem_offsets[i].first, mem_offsets[i].second);

  T* dataset_cpy;
  T* target_cpy;
  int* attribute_cpy;
  int* node_cursor_cpy;
  dev->AllocateHostMemory((void**)&node_cursor_cpy, sizeof(int) * 3);
  dev->AllocateHostMemory((void**)&dataset_cpy,
                          sizeof(T) * data_samples.size());
  dev->AllocateHostMemory((void**)&target_cpy, sizeof(T) * data_targets.size());
  dev->AllocateHostMemory((void**)&attribute_cpy,
                          sizeof(T) * data_samples.size());
  memset(attribute_cpy, 2, sizeof(int) * nr_features);
  memset(node_cursor_cpy, 0, sizeof(int) * 3);
  memcpy(dataset_cpy, data_samples.data(), sizeof(T) * data_samples.size());
  memcpy(target_cpy, data_targets.data(), sizeof(T) * data_targets.size());
  dev->CopyToDevice(dataset_cpy, gpu_params->dataset,
                    sizeof(T) * data_samples.size());
  dev->CopyToDevice(target_cpy, gpu_params->target_data,
                    sizeof(T) * data_targets.size());
  dev->CopyToDevice(attribute_cpy, gpu_params->attribute_type,
                    sizeof(int) * nr_features);
  dev->CopyToDevice(node_cursor_cpy, gpu_params->node_cursors, sizeof(int) * 3);

  memset(&dataset_info, 0, sizeof(dataset_info));
  dataset_info.nr_attributes = nr_features;
  dataset_info.nr_instances = nr_samples;
  dataset_info.nr_target_values = nr_targets;
  dataset_info.data_type = params->Get<int>(EnsemblesLib::kAlgoType);

  memset(&static_info, 0, sizeof(static_info));
  static_info.loaded_trees = max_tree_batch_;
  static_info.total_trees = nr_total_trees;
  static_info.max_node_size = params->Get<int>(EnsemblesLib::kMinNodeSize);
  static_info.max_node_depth = params->Get<int>(EnsemblesLib::kMaxDepth);
  static_info.node_buffer_size = 1024;

  auto k = params->Get<int>(EnsemblesLib::kNrFeatures);
  static_info.nr_features = k > 0 ? k : int(std::round(log(nr_features))) + 1;
  static_info.max_class_count = nr_targets;
  static_info.balanced_sampling =
      params->Get<bool>(EnsemblesLib::kEasyEnsemble);

  memset(&iteration_info, 0, sizeof(iteration_info));
  iteration_info.read_buffer_id = iteration_info.write_buffer_id = 0;
  iteration_info.tick_tock = true;

  cpy_data_static_info(&dataset_info, &static_info);
  cpy_iteration_info(&iteration_info);
  dev->SynchronizeDevice();
  dev->DeallocateHostMemory(dataset_cpy);
  dev->DeallocateHostMemory(target_cpy);
  dev->DeallocateHostMemory(attribute_cpy);
  dev->DeallocateHostMemory(node_cursor_cpy);
}

template <typename T>
void GpuRf<T>::AllocatePredict(sp<lib_gpu::GpuDevice> dev,
                               sp<lib_algorithms::MlAlgorithmParams> params,
                               GpuDte::GpuParams<T>* gpu_params,
                               sp<lib_data::MlDataFrame<T>> data,
                               sp<lib_models::MlModel> model,
                               GpuDte::gpuDTE_StaticInfo& static_info,
                               GpuDte::gpuDTE_DatasetInfo& dataset_info,
                               GpuDte::gpuDTE_IterationInfo& iteration_info) {
  int nr_targets = model->Get<int>(ModelsLib::kNrTargets);
  int nr_samples = data->GetNrSamples();
  int nr_trees = model->Get<int>(ModelsLib::kNrTrees);
  int nr_features = model->Get<int>(ModelsLib::kNrFeatures);
  auto model_type =
      model->Get<AlgorithmsLib::AlgorithmType>(ModelsLib::kModelType);
  if (model_type == AlgorithmsLib::kRegression) nr_targets = 1;

  // Allocate prediction buffers
  auto& node_headers =
      model->Get<col_array<GpuDte::gpuDTE_NodeHeader_Classify<T>>>(
          ModelsLib::kNodeArray);
  auto& prob_data = model->Get<col_array<T>>(ModelsLib::kProbArray);
  auto& data_samples = data->GetSamples();

  col_array<std::pair<void**, size_t>> mem_offsets;
  mem_offsets.emplace_back(std::pair<void**, size_t>(
      (void**)&gpu_params->predictions, sizeof(T) * nr_samples * nr_targets));
  mem_offsets.emplace_back(
      std::pair<void**, size_t>((void**)&gpu_params->node_buffers_classify,
                                sizeof(GpuDte::gpuDTE_NodeHeader_Classify<T>) *
                                    int(node_headers.size())));
  mem_offsets.emplace_back(
      std::pair<void**, size_t>((void**)&gpu_params->probability_tmp_buffer,
                                sizeof(T) * prob_data.size()));
  mem_offsets.emplace_back(std::pair<void**, size_t>(
      (void**)&gpu_params->dataset, sizeof(T) * data_samples.size()));
  mem_offsets.emplace_back(std::pair<void**, size_t>(
      (void**)&gpu_params->attribute_type, sizeof(int) * nr_features));

  for (int i = 0; i < mem_offsets.size(); ++i)
    dev->AllocateMemory(mem_offsets[i].first, mem_offsets[i].second);

  GpuDte::gpuDTE_NodeHeader_Classify<T>* node_head_cpy;
  T* prob_data_cpy;
  T* data_samples_cpy;
  T* pred_init_cpy;
  int* attribute_cpy;
  dev->AllocateHostMemory((void**)&pred_init_cpy,
                          sizeof(T) * nr_samples * nr_targets);
  dev->AllocateHostMemory(
      (void**)&node_head_cpy,
      sizeof(GpuDte::gpuDTE_NodeHeader_Classify<T>) * node_headers.size());
  dev->AllocateHostMemory((void**)&prob_data_cpy, sizeof(T) * prob_data.size());
  dev->AllocateHostMemory((void**)&data_samples_cpy,
                          sizeof(T) * data_samples.size());
  dev->AllocateHostMemory((void**)&attribute_cpy, sizeof(int) * nr_features);
  memset(attribute_cpy, 2, sizeof(int) * nr_features);
  memset(pred_init_cpy, 0, sizeof(T) * nr_samples * nr_targets);
  memcpy(node_head_cpy, node_headers.data(),
         sizeof(GpuDte::gpuDTE_NodeHeader_Classify<T>) * node_headers.size());
  memcpy(prob_data_cpy, prob_data.data(), sizeof(T) * prob_data.size());
  memcpy(data_samples_cpy, data_samples.data(),
         sizeof(T) * data_samples.size());
  dev->CopyToDevice(
      node_head_cpy, gpu_params->node_buffers_classify,
      sizeof(GpuDte::gpuDTE_NodeHeader_Classify<T>) * node_headers.size());
  dev->CopyToDevice(prob_data_cpy, gpu_params->probability_tmp_buffer,
                    sizeof(T) * prob_data.size());
  dev->CopyToDevice(data_samples_cpy, gpu_params->dataset,
                    sizeof(T) * data_samples.size());
  dev->CopyToDevice(attribute_cpy, gpu_params->attribute_type,
                    sizeof(int) * nr_features);
  dev->CopyToDevice(pred_init_cpy, gpu_params->predictions,
                    sizeof(T) * nr_samples * nr_targets);

  memset(&dataset_info, 0, sizeof(dataset_info));
  dataset_info.nr_attributes = nr_features;
  dataset_info.nr_instances = nr_samples;
  dataset_info.nr_target_values = nr_targets;
  dataset_info.data_type = model_type;

  memset(&static_info, 0, sizeof(static_info));
  static_info.loaded_trees = nr_trees;
  static_info.total_trees = nr_trees;
  static_info.max_node_size = params->Get<int>(EnsemblesLib::kMinNodeSize);
  static_info.max_node_depth = params->Get<int>(EnsemblesLib::kMaxDepth);

  auto k = params->Get<int>(EnsemblesLib::kNrFeatures);
  static_info.nr_features = k > 0 ? k : int(std::round(log(nr_features))) + 1;
  static_info.max_class_count = nr_targets;
  static_info.balanced_sampling =
      params->Get<bool>(EnsemblesLib::kEasyEnsemble);

  memset(&iteration_info, 0, sizeof(iteration_info));
  iteration_info.read_buffer_id = 0;
  iteration_info.tree_offset = 0;

  cpy_data_static_info(&dataset_info, &static_info);
  cpy_iteration_info(&iteration_info);
  dev->SynchronizeDevice();
  dev->DeallocateHostMemory(node_head_cpy);
  dev->DeallocateHostMemory(prob_data_cpy);
  dev->DeallocateHostMemory(data_samples_cpy);
  dev->DeallocateHostMemory(attribute_cpy);
  dev->DeallocateHostMemory(pred_init_cpy);
}

template <typename T>
void GpuRf<T>::SwapBuffers(int* lhs, int* rhs) {
  int tmp = *rhs;
  *rhs = *lhs;
  *lhs = tmp;
}

template <typename T>
void GpuRf<T>::StreamToCache(
    sp<lib_gpu::GpuDevice> dev, HostAllocFit& host_alloc, int src_id,
    int layer_id,
    col_array<col_array<GpuDte::gpuDTE_NodeHeader_Train<T>>>& node_cache,
    col_array<int>& buffer_counts,
    GpuDte::gpuDTE_NodeHeader_Train<T>* node_headers) {
  int nr_nodes = buffer_counts[src_id];
  buffer_counts[src_id] = 0;
  if (nr_nodes <= 0) return;

  dev->CopyToHost(host_alloc.node_cpy, node_headers,
                  nr_nodes * sizeof(GpuDte::gpuDTE_NodeHeader_Train<T>));
  dev->SynchronizeDevice();

  // Add to cache
  int cache_id = layer_id == 0 ? 1 : 0;
  for (int i = 0; i < nr_nodes; ++i)
    node_cache[cache_id].emplace_back(host_alloc.node_cpy[i]);
}

template <typename T>
void GpuRf<T>::StreamFromCache(
    sp<lib_gpu::GpuDevice> dev, HostAllocFit& host_alloc, int dst_id,
    int layer_id,
    col_array<col_array<GpuDte::gpuDTE_NodeHeader_Train<T>>>& node_cache,
    col_array<int>& buffer_counts,
    GpuDte::gpuDTE_NodeHeader_Train<T>* node_headers) {
  int nr_nodes = buffer_counts[dst_id];

  // Pre-stream next layer chunk for next iteration
  memcpy(host_alloc.node_cpy, node_cache[layer_id].data(),
         sizeof(GpuDte::gpuDTE_NodeHeader_Train<T>) * nr_nodes);
  dev->CopyToDevice(host_alloc.node_cpy, node_headers,
                    sizeof(GpuDte::gpuDTE_NodeHeader_Train<T>) * nr_nodes);

  if (node_cache[layer_id].size() - nr_nodes > 0)
    for (int i = 0; i < node_cache[layer_id].size() - nr_nodes; ++i)
      node_cache[layer_id][i] = node_cache[layer_id][nr_nodes + i];

  for (int i = 0; i < nr_nodes; ++i) node_cache[layer_id].pop_back();
  dev->SynchronizeDevice();
}

template <typename T>
__device__ void GpuRf<T>::gpurf_setup_kernel(GpuDte::GpuParams<T>* params) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  // Each thread gets same seed, a different sequence number, no offset
  if (id < params->iteration_info->threads_launched)
    curand_init(324123, id, 0, &params->random_states[id]);
}

template <typename T>
__device__ void GpuRf<T>::gpurf_initialize_tree_batch(
    GpuDte::GpuParams<T>* params) {
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
        localCursor = GpuDte::AtomicAdd(&s_indexCursor, 1);
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
    GpuDte::gpuDTE_NodeHeader_Train<T> root;
    root.parent_id = -2;
    root.attribute = -2;
    root.split_point = -2;
    root.trackinid = GpuDte::AtomicAdd(&params->node_cursors[node_id_], 1);
    root.node_index_start = treeOffset;
    root.node_index_count = s_indexCursor;

    params->node_buffers[params->iteration_info->read_buffer_id][blockIdx.x] =
        root;
  }
}

template <typename T>
__device__ void GpuRf<T>::gpurf_find_split(GpuDte::GpuParams<T>* params) {
  __shared__ T s_dynamic_shared[40];
  __shared__ unsigned int s_histograms[1024];
  __shared__ unsigned int s_offsets[256];
  __shared__ GpuDte::gpuDTE_NodeHeader_Train<T> s_tree_node;
  __shared__ GpuDte::gpuDTE_TmpNodeValues<T> s_tmp_node;
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
        GpuDte::AtomicAdd(&s_dynamic_shared[int(params->target_data[inst])],
                          T(1));
        break;
      case type_regression_:
        GpuDte::AtomicAdd(&s_dynamic_shared[0], params->target_data[inst]);
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
__device__ void GpuRf<T>::gpurf_perform_split(GpuDte::GpuParams<T>* params) {
  extern __shared__ int s_node_counts[];

  GpuDte::gpudte_perform_split(
      *(params->static_info), *(params->dataset_info),
      *(params->iteration_info), params->probability_buffers,
      params->probability_tmp_buffer, params->dataset, params->attribute_type,
      s_node_counts, params->indices_buffer, params->node_cursors,
      params->node_buffers);
}

template <typename T>
__device__ void GpuRf<T>::gpurf_predict(GpuDte::GpuParams<T>* params) {
  if (threadIdx.x + blockIdx.x * blockDim.x >=
      params->iteration_info->threads_launched)
    return;

  int tid = params->iteration_info->tree_offset + threadIdx.x +
            blockIdx.x * blockDim.x;
  GpuDte::gpudte_predict(
      tid, params->dataset_info->nr_instances, params->dataset_info->data_type,
      params->dataset_info->nr_target_values, params->node_buffers_classify,
      params->dataset, params->probability_tmp_buffer, params->predictions,
      params->attribute_type);
}

template <typename T>
__device__ void GpuRf<T>::gpurf_oob_estimate(GpuDte::GpuParams<T>* params) {
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
  GpuDte::gpuDTE_NodeHeader_Classify<T> tree_node =
      params->node_buffers_classify[treeId];
  while (tree_node.child_count != 0) {
    int attribute_type = params->attribute_type[tree_node.attribute];

    dataPoint = GpuDte::get_data_point(tree_node.attribute, instanceId,
                                       params->dataset_info->nr_instances,
                                       params->dataset);
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
        GpuDte::AtomicAdd(&params->oobCounts[0], 1);
      else
        GpuDte::AtomicAdd(&params->oobCounts[1], 1);
      break;
    }
    case type_regression_:
      GpuDte::AtomicAdd(
          &params->mse[0],
          params->probability_tmp_buffer[tree_node.probability_start]);
      break;
  }
}

template <typename T>
__device__ void GpuRf<T>::gpurf_feature_importance(
    GpuDte::GpuParams<T>* params) {
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
  GpuDte::gpuDTE_NodeHeader_Classify<T> tree_node =
      params->node_buffers_classify[treeId];
  while (tree_node.child_count != 0) {
    int attribute_type = params->attribute_type[tree_node.attribute];

    if (randAttribute == tree_node.attribute)
      instance = curand(&localState) % params->dataset_info->nr_instances;
    else
      instance = instanceId;

    dataPoint = GpuDte::get_data_point(tree_node.attribute, instance,
                                       params->dataset_info->nr_instances,
                                       params->dataset);
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
        GpuDte::AtomicAdd(&params->oobCounts[randAttribute * 2], 1);
      else
        GpuDte::AtomicAdd(&params->oobCounts[randAttribute * 2 + 1], 1);
      break;
    }
    case type_regression_:
      GpuDte::AtomicAdd(
          &params->mse[randAttribute],
          params->probability_tmp_buffer[tree_node.probability_start]);
      break;
  }
}

template <typename T>
__device__ void GpuRf<T>::radix_sort_on_attribute(
    GpuDte::GpuParams<T>* params, GpuDte::gpuDTE_NodeHeader_Train<T>& node,
    GpuDte::gpuDTE_TmpNodeValues<T>& tmp_node, unsigned int s_histograms[1024],
    unsigned int s_offsets[256]) {
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
    GpuDte::AtomicAdd(&s_histograms[*dataVal], 1);
    GpuDte::AtomicAdd(&s_histograms[256 + (*(dataVal + 1))], 1);
    GpuDte::AtomicAdd(&s_histograms[512 + (*(dataVal + 2))], 1);
    GpuDte::AtomicAdd(&s_histograms[768 + (*(dataVal + 3))], 1);
  }

  __syncthreads();

  for (int i = threadIdx.x + 128; i < 256; i += blockDim.x)
    GpuDte::AtomicAdd(&s_nrNegativeValues, s_histograms[768 + i]);

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
            GpuDte::AtomicAdd(&curCount[radix], 1);
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
              GpuDte::AtomicAdd((int*)&s_offsets[radix], 1);
            else
              GpuDte::AtomicAdd((int*)&s_offsets[radix], -1);

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
    GpuDte::GpuParams<T>* params, GpuDte::gpuDTE_NodeHeader_Train<T>& node,
    GpuDte::gpuDTE_TmpNodeValues<T>& tmp_node, T* curr_dist, int att_type,
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
  T prior = GpuDte::entropy_over_columns(
      (T*)curr_dist, att_type, params->dataset_info->nr_target_values);

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
      val = GpuDte::get_data_point(tmp_node.tmp_attribute, inst,
                                   params->dataset_info->nr_instances,
                                   params->dataset);
      s_offsets[threadIdx.x] = int(params->target_data[inst]);

      if (i + threadIdx.x != 0) {
        inst = params->indices_buffer[params->iteration_info->tick_tock ? 0 : 1]
                                     [nodeIndStart + i + threadIdx.x - 1];
        preVal = GpuDte::get_data_point(tmp_node.tmp_attribute, inst,
                                        params->dataset_info->nr_instances,
                                        params->dataset);
      }

      GpuDte::AtomicAdd(&curr_dist[params->dataset_info->nr_target_values +
                                   s_offsets[threadIdx.x]],
                        T(1));
      GpuDte::AtomicAdd(&curr_dist[s_offsets[threadIdx.x]], T(-1));
    }

    __syncthreads();

    if (!skip) {
      for (int ii = threadIdx.x; ii >= 0; --ii) {
        ++local_dist[params->dataset_info->nr_target_values + s_offsets[ii]];
        --local_dist[s_offsets[ii]];
      }

      response = prior - GpuDte::entropy_conditioned_on_rows(
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
      pointBeforeSplit = GpuDte::get_data_point(
          tmp_node.tmp_attribute, instJustBeforeSplit,
          params->dataset_info->nr_instances, params->dataset);
      pointAfterSplit = GpuDte::get_data_point(
          tmp_node.tmp_attribute, instJustAfterSplit,
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
    val = GpuDte::get_data_point(tmp_node.tmp_attribute, inst,
                                 params->dataset_info->nr_instances,
                                 params->dataset);

    if (val != -flt_max)
      GpuDte::AtomicAdd(&curr_dist[params->dataset_info->nr_target_values *
                                       ((val < tmp_node.tmp_split) ? 0 : 1) +
                                   int(params->target_data[inst])],
                        T(1));
    else
      GpuDte::AtomicAdd(&curr_dist[int(params->target_data[inst])], T(1));
  }

  return bestResponse;
}

template <typename T>
__device__ T GpuRf<T>::variance_calculation(
    GpuDte::GpuParams<T>* params, GpuDte::gpuDTE_NodeHeader_Train<T>& node,
    GpuDte::gpuDTE_TmpNodeValues<T>& tmp_node, T* curr_dist, T* s_histograms) {
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
    val = GpuDte::get_data_point(
        attribute, inst, params->dataset_info->nr_instances, params->dataset);

    if (val != -flt_max)
      GpuDte::AtomicAdd(&curr_dist[1], params->target_data[inst]);
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
      val = GpuDte::get_data_point(
          attribute, inst, params->dataset_info->nr_instances, params->dataset);
      s_histograms[threadIdx.x] = val;

      if (i + threadIdx.x != 0) {
        inst = params->indices_buffer[params->iteration_info->tick_tock ? 0 : 1]
                                     [nodeIndStart + i + threadIdx.x - 1];
        preVal = GpuDte::get_data_point(tmp_node.tmp_attribute, inst,
                                        params->dataset_info->nr_instances,
                                        params->dataset);
      }

      GpuDte::AtomicAdd(&curr_dist[0], val);
      GpuDte::AtomicAdd(&curr_dist[1], -val);
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
      pointBeforeSplit = GpuDte::get_data_point(
          tmp_node.tmp_attribute, instJustBeforeSplit,
          params->dataset_info->nr_instances, params->dataset);
      pointAfterSplit = GpuDte::get_data_point(
          tmp_node.tmp_attribute, instJustAfterSplit,
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
    val = GpuDte::get_data_point(
        attribute, inst, params->dataset_info->nr_instances, params->dataset);

    if (val != -flt_max) {
      GpuDte::AtomicAdd(&curr_dist[(val < tmp_node.tmp_split) ? 0 : 1], T(1));
      GpuDte::AtomicAdd(&curr_dist[(val < tmp_node.tmp_split) ? 2 : 3], val);
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
GpuRf<T>::HostAllocFit::HostAllocFit(sp<lib_gpu::GpuDevice> dev, size_t targets)
    : dev_(dev) {
  dev_->AllocateHostMemory((void**)&probability_cpy,
                           sizeof(T) * max_blocks_ * targets * max_nominal_);
  dev_->AllocateHostMemory(
      (void**)&node_cpy,
      sizeof(GpuDte::gpuDTE_NodeHeader_Train<T>) * max_blocks_);
  dev_->AllocateHostMemory((void**)&node_cursor_cpy, sizeof(int) * 3);
}
template <typename T>
GpuRf<T>::HostAllocFit::~HostAllocFit() {
  dev_->DeallocateHostMemory(probability_cpy);
  dev_->DeallocateHostMemory(node_cpy);
  dev_->DeallocateHostMemory(node_cursor_cpy);
}

template <typename T>
GpuRf<T>::HostAllocPredict::HostAllocPredict(sp<lib_gpu::GpuDevice> dev,
                                             size_t samples, size_t targets)
    : dev_(dev) {
  dev_->AllocateHostMemory((void**)&predictions_cpy,
                           samples * targets * sizeof(T));
}
template <typename T>
GpuRf<T>::HostAllocPredict::~HostAllocPredict() {
  dev_->DeallocateHostMemory(predictions_cpy);
}

template GpuRf<float>::GpuRf();
template GpuRf<double>::GpuRf();
}