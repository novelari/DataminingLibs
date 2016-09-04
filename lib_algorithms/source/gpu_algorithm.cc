#include "precomp.h"

#include "gpu_algorithm.h"
#include "lib_gpu.h"
#include "ml_algorithm_params.h"
#include "algorithms_interface.h"
#include "lib_ensembles.h"

namespace lib_algorithms {
template <typename T>
inline GpuAlgorithm<T>::GpuAlgorithm(sp<MlAlgorithm<T>> gpu_alg)
    : device_algorithm_(gpu_alg) {}
template <typename T>
GpuAlgorithm<T>::~GpuAlgorithm() {}

template <typename T>
sp<lib_models::MlModel> GpuAlgorithm<T>::Fit(sp<lib_data::MlDataFrame<T>> data,
                                             sp<MlAlgorithmParams> params) {
  auto &device = GpuLib::GetInstance().GetGpuDevice();
  device.SetDeviceForThread(0);
  params->Set(EnsemblesLib::kDevId, 0);
  return device_algorithm_->Fit(data, params);
  /*auto param_vec = SplitParameterPack(params, device.GetDeviceCount());
  col_array<sp<lib_models::MlModel>> models;
  for (int i = 0; i < device.GetDeviceCount(); ++i) {
    device.SetDeviceForThread(i);
    param_vec[i]->Set(AlgorithmsInterface::kDevId, i);
    models.push_back(device_algorithm_->Fit(data, param_vec[i]));
  }
  return AggregateModels(models);*/
}
template <typename T>
sp<lib_data::MlResultData<T>> GpuAlgorithm<T>::Predict(
    sp<lib_data::MlDataFrame<T>> data, sp<lib_models::MlModel> model,
    sp<MlAlgorithmParams> params) {
  auto &device = GpuLib::GetInstance().GetGpuDevice();
  device.SetDeviceForThread(0);
  params->Set(EnsemblesLib::kDevId, 0);
  return device_algorithm_->Predict(data, model, params);
  /*auto param_vec = SplitParameterPack(params, device.GetDeviceCount());
  col_array<sp<lib_data::MlResultData<T>>> results;
  for (int i = 0; i < device.GetDeviceCount(); ++i) {
    device.SetDeviceForThread(i);
    param_vec[i]->Set(AlgorithmsInterface::kDevId, i);
    results.push_back(device_algorithm_->Predict(data, model, param_vec[i]));
  }
  return AggregateResults(results);*/
}

template <typename T>
sp<lib_models::MlModel> GpuAlgorithm<T>::AggregateModels(
    col_array<sp<lib_models::MlModel>> models) {
  return device_algorithm_->AggregateModels(models);
}

template <typename T>
sp<lib_data::MlResultData<T>> GpuAlgorithm<T>::AggregateResults(
    col_array<sp<lib_data::MlResultData<T>>> results) {
  return device_algorithm_->AggregateResults(results);
}

template <typename T>
col_array<sp<MlAlgorithmParams>> GpuAlgorithm<T>::SplitParameterPack(
    sp<MlAlgorithmParams> params, const int parts) {
  return device_algorithm_->SplitParameterPack(params, parts);
}

template GpuAlgorithm<float>::GpuAlgorithm(sp<MlAlgorithm<float>> gpu_alg);
template GpuAlgorithm<double>::GpuAlgorithm(sp<MlAlgorithm<double>> gpu_alg);
}