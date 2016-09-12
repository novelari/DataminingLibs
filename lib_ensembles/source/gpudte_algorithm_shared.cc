#include "precomp.h"

#include "gpudte_algorithm_shared.h"

namespace lib_ensembles {
template <typename T>
GpuDteAlgorithmShared::GpuParams<T>::GpuParams() {
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
void GpuDteAlgorithmShared::GpuParams<T>::finalize(sp<lib_gpu::GpuDevice> dev) {
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

template void GpuDteAlgorithmShared::GpuParams<float>::finalize(
    sp<lib_gpu::GpuDevice> dev);
template void GpuDteAlgorithmShared::GpuParams<double>::finalize(
    sp<lib_gpu::GpuDevice> dev);
template GpuDteAlgorithmShared::GpuParams<float>::GpuParams();
template GpuDteAlgorithmShared::GpuParams<double>::GpuParams();
}