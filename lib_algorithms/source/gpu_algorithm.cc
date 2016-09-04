#include "precomp.h"

#include "algorithms_interface.h"
#include "gpu_algorithm.h"
#include "lib_core.h"
#include "lib_ensembles.h"
#include "lib_gpu.h"
#include "ml_algorithm_params.h"

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

  auto dev_count = device.GetDeviceCount();
  auto param_vec = SplitParameterPack(params, dev_count);
  col_array<sp<lib_models::MlModel>> models(dev_count,
                                            sp<lib_models::MlModel>());
  auto run_func = [&](int i) {
    device.SetDeviceForThread(i);
    param_vec[i]->Set(EnsemblesLib::kDevId, i);
    models[i] = device_algorithm_->Fit(data, param_vec[i]);
  };
  CoreLib::GetInstance().ParallelFor(0, dev_count, run_func);

  return AggregateModels(models);
}
template <typename T>
sp<lib_data::MlResultData<T>> GpuAlgorithm<T>::Predict(
    sp<lib_data::MlDataFrame<T>> data, sp<lib_models::MlModel> model,
    sp<MlAlgorithmParams> params) {
  auto &device = GpuLib::GetInstance().GetGpuDevice();
  device.SetDeviceForThread(0);
  params->Set(EnsemblesLib::kDevId, 0);
  return device_algorithm_->Predict(data, model, params);

  auto dev_count = device.GetDeviceCount();
  auto param_vec = SplitParameterPack(params, dev_count);
  auto model_vec = SplitModel(model, dev_count);
  col_array<sp<lib_data::MlResultData<T>>> results(
      dev_count, sp<lib_data::MlResultData<T>>());
  auto run_func = [&](int i) {
    device.SetDeviceForThread(i);
    param_vec[i]->Set(EnsemblesLib::kDevId, i);
    results[i] = device_algorithm_->Predict(data, model_vec[i], param_vec[i]);
  };
  CoreLib::GetInstance().ParallelFor(0, dev_count, run_func);

  return AggregateResults(results);
}

template <typename T>
sp<lib_models::MlModel> GpuAlgorithm<T>::AggregateModels(
    col_array<sp<lib_models::MlModel>> models) {
  return device_algorithm_->AggregateModels(models);
}

template <typename T>
col_array<sp<lib_models::MlModel>> GpuAlgorithm<T>::SplitModel(
    sp<lib_models::MlModel> model, const int parts) {
  return device_algorithm_->SplitModel(model, parts);
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