#include "precomp.h"

#include "algorithms_interface.h"
#include "gpu_algorithm.h"
#include "ml_algorithm_params_impl.h"

namespace lib_algorithms {
AlgorithmsInterface& AlgorithmsInterface::GetInstance() {
  static AlgorithmsInterface instance;
  return instance;
}

sp<MlAlgorithmParams> AlgorithmsInterface::CreateAlgorithmParams(int size) {
  return std::make_shared<MlAlgorithmParamsImpl>(size);
}

template <typename T>
sp<MlAlgorithm<T>> AlgorithmsInterface::CreateGpuAlgorithm(
    sp<MlAlgorithm<T>> algo) {
  return std::make_shared<GpuAlgorithm<T>>(algo);
}

template DLLExport sp<MlAlgorithm<float>>
AlgorithmsInterface::CreateGpuAlgorithm(sp<MlAlgorithm<float>> algo);
template DLLExport sp<MlAlgorithm<double>>
AlgorithmsInterface::CreateGpuAlgorithm(sp<MlAlgorithm<double>> algo);
}