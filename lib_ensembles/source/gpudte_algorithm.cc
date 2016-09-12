#include "precomp.h"

#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <cassert>
#include "gpudte_algorithm.h"
#include "lib_core.h"
#include "lib_ensembles.h"
#include "lib_gpu.h"

#include "gpudte.h"

namespace lib_ensembles {
template <typename T>
sp<lib_models::MlModel> GpuDteAlgorithm<T>::Fit(
    sp<lib_data::MlDataFrame<T>> data,
    sp<lib_algorithms::MlAlgorithmParams> params) {
  GpuDteAlgorithmShared::gpuDTE_StaticInfo static_info;
  GpuDteAlgorithmShared::gpuDTE_DatasetInfo dataset_info;
  GpuDteAlgorithmShared::gpuDTE_IterationInfo iteration_info;
  auto device = GpuLib::GetInstance().CreateGpuDevice(
      params->Get<int>(EnsemblesLib::kDevId));
  auto gpu_params = GpuDteAlgorithmShared::GpuParams<T>();
  auto nr_total_trees = params->Get<int>(EnsemblesLib::kNrTrees);
  auto algo_type =
      params->Get<AlgorithmsLib::AlgorithmType>(EnsemblesLib::kAlgoType);
  auto nr_samples = data->GetNrSamples();
  auto nr_features = data->GetNrFeatures();
  auto nr_targets = data->GetNrTargets();
  int trees_built = 0;

  col_array<sp<lib_models::MlModel>> models;
  GpuDteAlgorithmShared::gpuDTE_NodeHeader_Classify<T> tmp_node;
  col_array<col_array<GpuDteAlgorithmShared::gpuDTE_NodeHeader_Classify<T>>>
      tree_nodes;
  col_array<col_array<T>> tree_probabilities;
  HostAllocFit host_alloc(device, nr_targets);

  // auto barrier = CoreLib::GetInstance().CreateBarrier(2);
  // bool run_rec_func = true;
  sutil::LockFreeList<std::pair<int, int>> job_list;
  col_map<int, int> track_map;
  int node_size = sizeof(GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T>);
  int type_size = sizeof(T);
  int prob_id = 0;
  auto data_rec_func = [&]() {
    // device.SetDeviceForThread(params->Get<int>(EnsemblesLib::kDevId));
    // barrier->Wait();
    // do {
    auto pair = job_list.pop_front();
    device->CopyToHost(host_alloc.probability_cpy,
                       gpu_params.probability_buffers[prob_id],
                       type_size * pair->first * nr_targets, 1);
    device->CopyToHost(host_alloc.node_cpy,
                       gpu_params.node_buffers[pair->second],
                       node_size * pair->first, 1);
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
        track_map[gpu_node.tracking_id] = int(tree_nodes.back().size());
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

  col_array<col_array<GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T>>>
      node_cache(
          2, col_array<GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T>>());
  int nodes_pulled = 0;
  int stream_buffer = 2;
  int trees_left = nr_total_trees;
  while (trees_left > 0) {
    tree_probabilities.emplace_back(col_array<T>());
    tree_nodes.emplace_back(
        col_array<GpuDteAlgorithmShared::gpuDTE_NodeHeader_Classify<T>>());
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
    gpu_functions_->CopyIterationInfo(iteration_info);
    device->SynchronizeDevice();
    gpu_functions_->CallCudaKernel(max_blocks_ / block_size_, block_size_,
                                   gpu_params,
                                   GpuDteAlgorithmShared::kSetupKernel);
    device->SynchronizeDevice();

    iteration_info.threads_launched = trees_launched;
    gpu_functions_->CopyIterationInfo(iteration_info);
    gpu_functions_->CallCudaKernel(trees_launched, block_size_, gpu_params,
                                   GpuDteAlgorithmShared::kInitTreeBatch);
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
        gpu_functions_->CopyIterationInfo(iteration_info);
        // cpy_iteration_info(&iteration_info);

        gpu_functions_->CallCudaKernel(nodes_launched, block_size_, gpu_params,
                                       GpuDteAlgorithmShared::kFindSplit);
        device->SynchronizeDevice();
        gpu_functions_->CallCudaKernel(nodes_launched, block_size_, gpu_params,
                                       GpuDteAlgorithmShared::kPerformSplit);
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
sp<lib_data::MlResultData<T>> GpuDteAlgorithm<T>::Predict(
    sp<lib_data::MlDataFrame<T>> data, sp<lib_models::MlModel> model,
    sp<lib_algorithms::MlAlgorithmParams> params) {
  GpuDteAlgorithmShared::gpuDTE_StaticInfo static_info;
  GpuDteAlgorithmShared::gpuDTE_DatasetInfo dataset_info;
  GpuDteAlgorithmShared::gpuDTE_IterationInfo iteration_info;
  auto device = GpuLib::GetInstance().CreateGpuDevice(
      params->Get<int>(EnsemblesLib::kDevId));
  auto result_data = DataLib::GetInstance().CreateResultData<T>();
  auto gpu_params = GpuDteAlgorithmShared::GpuParams<T>();
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
      gpu_functions_->CopyIterationInfo(iteration_info);
      device->SynchronizeDevice();
      gpu_functions_->CallCudaKernel(launch_threads / block_size_, block_size_,
                                     gpu_params,
                                     GpuDteAlgorithmShared::kPredict);
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
sp<lib_models::MlModel> GpuDteAlgorithm<T>::AggregateModels(
    col_array<sp<lib_models::MlModel>> models) {
  if (models.size() < 2) return models[0];
  col_array<T> aggregate_prob;
  std::function<void(int)> rec_add_nodes;
  col_array<GpuDteAlgorithmShared::gpuDTE_NodeHeader_Classify<T>>
      aggregate_node;
  for (int i = 0; i < models.size(); ++i) {
    auto& node_headers =
        models[i]
            ->Get<col_array<
                GpuDteAlgorithmShared::gpuDTE_NodeHeader_Classify<T>>>(
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
        models[i]
            ->Get<col_array<
                GpuDteAlgorithmShared::gpuDTE_NodeHeader_Classify<T>>>(
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
col_array<sp<lib_models::MlModel>> GpuDteAlgorithm<T>::SplitModel(
    sp<lib_models::MlModel> model, const int parts) {
  auto trees = model->Get<int>(ModelsLib::kNrTrees);
  auto targets = model->Get<int>(ModelsLib::kNrTargets);
  col_array<sp<lib_models::MlModel>> models(
      parts, ModelsLib::GetInstance().CreateModel());
  auto& node_headers = model->Get<
      col_array<GpuDteAlgorithmShared::gpuDTE_NodeHeader_Classify<T>>>(
      ModelsLib::kNodeArray);
  auto& prob_data = model->Get<col_array<T>>(ModelsLib::kProbArray);

  std::function<void(int)> rec_add_nodes;
  auto tree_split = trees / parts;
  for (int i = 0; i < parts; ++i) {
    col_array<T> prob_array;
    col_array<GpuDteAlgorithmShared::gpuDTE_NodeHeader_Classify<T>> node_array;

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
sp<lib_data::MlResultData<T>> GpuDteAlgorithm<T>::AggregateResults(
    col_array<sp<lib_data::MlResultData<T>>> results) {
  for (int i = 1; i < results.size(); ++i) *results[0] += *results[i];
  return results[0];
}

template <typename T>
col_array<sp<lib_algorithms::MlAlgorithmParams>>
GpuDteAlgorithm<T>::SplitParameterPack(
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
void GpuDteAlgorithm<T>::AllocateFit(
    sp<lib_gpu::GpuDevice> dev, sp<lib_algorithms::MlAlgorithmParams> params,
    GpuDteAlgorithmShared::GpuParams<T>* gpu_params,
    sp<lib_data::MlDataFrame<T>> data,
    GpuDteAlgorithmShared::gpuDTE_StaticInfo& static_info,
    GpuDteAlgorithmShared::gpuDTE_DatasetInfo& dataset_info,
    GpuDteAlgorithmShared::gpuDTE_IterationInfo& iteration_info) {
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
        sizeof(GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T>) *
            max_blocks_));
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
      sizeof(GpuDteAlgorithmShared::gpuDTE_TmpNodeValues<T>) * max_blocks_));
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

  gpu_functions_->CopyDataStaticInfo(dataset_info, static_info);
  gpu_functions_->CopyIterationInfo(iteration_info);
  dev->SynchronizeDevice();
  dev->DeallocateHostMemory(dataset_cpy);
  dev->DeallocateHostMemory(target_cpy);
  dev->DeallocateHostMemory(attribute_cpy);
  dev->DeallocateHostMemory(node_cursor_cpy);
}

template <typename T>
void GpuDteAlgorithm<T>::AllocatePredict(
    sp<lib_gpu::GpuDevice> dev, sp<lib_algorithms::MlAlgorithmParams> params,
    GpuDteAlgorithmShared::GpuParams<T>* gpu_params,
    sp<lib_data::MlDataFrame<T>> data, sp<lib_models::MlModel> model,
    GpuDteAlgorithmShared::gpuDTE_StaticInfo& static_info,
    GpuDteAlgorithmShared::gpuDTE_DatasetInfo& dataset_info,
    GpuDteAlgorithmShared::gpuDTE_IterationInfo& iteration_info) {
  int nr_targets = model->Get<int>(ModelsLib::kNrTargets);
  int nr_samples = data->GetNrSamples();
  int nr_trees = model->Get<int>(ModelsLib::kNrTrees);
  int nr_features = model->Get<int>(ModelsLib::kNrFeatures);
  auto model_type =
      model->Get<AlgorithmsLib::AlgorithmType>(ModelsLib::kModelType);
  if (model_type == AlgorithmsLib::kRegression) nr_targets = 1;

  // Allocate prediction buffers
  auto& node_headers = model->Get<
      col_array<GpuDteAlgorithmShared::gpuDTE_NodeHeader_Classify<T>>>(
      ModelsLib::kNodeArray);
  auto& prob_data = model->Get<col_array<T>>(ModelsLib::kProbArray);
  auto& data_samples = data->GetSamples();

  col_array<std::pair<void**, size_t>> mem_offsets;
  mem_offsets.emplace_back(std::pair<void**, size_t>(
      (void**)&gpu_params->predictions, sizeof(T) * nr_samples * nr_targets));
  mem_offsets.emplace_back(std::pair<void**, size_t>(
      (void**)&gpu_params->node_buffers_classify,
      sizeof(GpuDteAlgorithmShared::gpuDTE_NodeHeader_Classify<T>) *
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

  GpuDteAlgorithmShared::gpuDTE_NodeHeader_Classify<T>* node_head_cpy;
  T* prob_data_cpy;
  T* data_samples_cpy;
  T* pred_init_cpy;
  int* attribute_cpy;
  dev->AllocateHostMemory((void**)&pred_init_cpy,
                          sizeof(T) * nr_samples * nr_targets);
  dev->AllocateHostMemory(
      (void**)&node_head_cpy,
      sizeof(GpuDteAlgorithmShared::gpuDTE_NodeHeader_Classify<T>) *
          node_headers.size());
  dev->AllocateHostMemory((void**)&prob_data_cpy, sizeof(T) * prob_data.size());
  dev->AllocateHostMemory((void**)&data_samples_cpy,
                          sizeof(T) * data_samples.size());
  dev->AllocateHostMemory((void**)&attribute_cpy, sizeof(int) * nr_features);
  memset(attribute_cpy, 2, sizeof(int) * nr_features);
  memset(pred_init_cpy, 0, sizeof(T) * nr_samples * nr_targets);
  memcpy(node_head_cpy, node_headers.data(),
         sizeof(GpuDteAlgorithmShared::gpuDTE_NodeHeader_Classify<T>) *
             node_headers.size());
  memcpy(prob_data_cpy, prob_data.data(), sizeof(T) * prob_data.size());
  memcpy(data_samples_cpy, data_samples.data(),
         sizeof(T) * data_samples.size());
  dev->CopyToDevice(
      node_head_cpy, gpu_params->node_buffers_classify,
      sizeof(GpuDteAlgorithmShared::gpuDTE_NodeHeader_Classify<T>) *
          node_headers.size());
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

  gpu_functions_->CopyDataStaticInfo(dataset_info, static_info);
  gpu_functions_->CopyIterationInfo(iteration_info);
  dev->SynchronizeDevice();
  dev->DeallocateHostMemory(node_head_cpy);
  dev->DeallocateHostMemory(prob_data_cpy);
  dev->DeallocateHostMemory(data_samples_cpy);
  dev->DeallocateHostMemory(attribute_cpy);
  dev->DeallocateHostMemory(pred_init_cpy);
}

template <typename T>
void GpuDteAlgorithm<T>::SwapBuffers(int* lhs, int* rhs) {
  int tmp = *rhs;
  *rhs = *lhs;
  *lhs = tmp;
}

template <typename T>
void GpuDteAlgorithm<T>::StreamToCache(
    sp<lib_gpu::GpuDevice> dev, HostAllocFit& host_alloc, int src_id,
    int layer_id,
    col_array<col_array<GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T>>>&
        node_cache,
    col_array<int>& buffer_counts,
    GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T>* node_headers) {
  int nr_nodes = buffer_counts[src_id];
  buffer_counts[src_id] = 0;
  if (nr_nodes <= 0) return;

  dev->CopyToHost(
      host_alloc.node_cpy, node_headers,
      nr_nodes * sizeof(GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T>));
  dev->SynchronizeDevice();

  // Add to cache
  int cache_id = layer_id == 0 ? 1 : 0;
  for (int i = 0; i < nr_nodes; ++i)
    node_cache[cache_id].emplace_back(host_alloc.node_cpy[i]);
}

template <typename T>
void GpuDteAlgorithm<T>::StreamFromCache(
    sp<lib_gpu::GpuDevice> dev, HostAllocFit& host_alloc, int dst_id,
    int layer_id,
    col_array<col_array<GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T>>>&
        node_cache,
    col_array<int>& buffer_counts,
    GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T>* node_headers) {
  int nr_nodes = buffer_counts[dst_id];

  // Pre-stream next layer chunk for next iteration
  memcpy(host_alloc.node_cpy, node_cache[layer_id].data(),
         sizeof(GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T>) * nr_nodes);
  dev->CopyToDevice(
      host_alloc.node_cpy, node_headers,
      sizeof(GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T>) * nr_nodes);

  if (node_cache[layer_id].size() - nr_nodes > 0)
    for (int i = 0; i < node_cache[layer_id].size() - nr_nodes; ++i)
      node_cache[layer_id][i] = node_cache[layer_id][nr_nodes + i];

  for (int i = 0; i < nr_nodes; ++i) node_cache[layer_id].pop_back();
  dev->SynchronizeDevice();
}

template <typename T>
GpuDteAlgorithm<T>::HostAllocFit::HostAllocFit(sp<lib_gpu::GpuDevice> dev,
                                               size_t targets)
    : dev_(dev) {
  dev_->AllocateHostMemory((void**)&probability_cpy,
                           sizeof(T) * max_blocks_ * targets * max_nominal_);
  dev_->AllocateHostMemory(
      (void**)&node_cpy,
      sizeof(GpuDteAlgorithmShared::gpuDTE_NodeHeader_Train<T>) * max_blocks_);
  dev_->AllocateHostMemory((void**)&node_cursor_cpy, sizeof(int) * 3);
}
template <typename T>
GpuDteAlgorithm<T>::HostAllocFit::~HostAllocFit() {
  dev_->DeallocateHostMemory(probability_cpy);
  dev_->DeallocateHostMemory(node_cpy);
  dev_->DeallocateHostMemory(node_cursor_cpy);
}

template <typename T>
GpuDteAlgorithm<T>::HostAllocPredict::HostAllocPredict(
    sp<lib_gpu::GpuDevice> dev, size_t samples, size_t targets)
    : dev_(dev) {
  dev_->AllocateHostMemory((void**)&predictions_cpy,
                           samples * targets * sizeof(T));
}
template <typename T>
GpuDteAlgorithm<T>::HostAllocPredict::~HostAllocPredict() {
  dev_->DeallocateHostMemory(predictions_cpy);
}

template GpuDteAlgorithm<float>::GpuDteAlgorithm(sp<GpuDte<float>> func);
template GpuDteAlgorithm<double>::GpuDteAlgorithm(sp<GpuDte<double>> func);
}