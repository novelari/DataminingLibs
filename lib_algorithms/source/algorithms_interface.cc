#include "precomp.h"

#include "algorithms_interface.h"
#include "gpu_algorithm.h"
#include "gpurf.h"
#include "ml_algorithm_params_impl.h"

namespace lib_algorithms {
AlgorithmsInterface& AlgorithmsInterface::GetInstance() {
  static AlgorithmsInterface instance;
  return instance;
}

template <typename T>
sp<MlAlgorithm<T>> AlgorithmsInterface::CreateGpuRfAlgorithm() {
  return std::make_shared<GpuAlgorithm<T>>(std::make_shared<GpuRf<T>>());
}

sp<MlAlgorithmParams> AlgorithmsInterface::CreateGpuRfParamPack() {
  auto params =
      std::make_shared<MlAlgorithmParamsImpl>(kDteEndMarker);

  // Set default parameters for random forest
  params->Set(kNrTrees, 100);
  params->Set(kNrFeatures, 0);
  params->Set(kMaxDepth, 1000);
  params->Set(kMinNodeSize, 5);
  params->Set(kAlgoType, kClassification);
  params->Set(kBagging, true);
  params->Set(kEasyEnsemble, false);
  return params;
}

AlgorithmsInterface::AlgorithmsInterface() {}
AlgorithmsInterface::~AlgorithmsInterface() {}

template DLLExport sp<MlAlgorithm<float>>
AlgorithmsInterface::CreateGpuRfAlgorithm();
template DLLExport sp<MlAlgorithm<double>>
AlgorithmsInterface::CreateGpuRfAlgorithm();
}