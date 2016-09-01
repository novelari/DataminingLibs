#include "precomp.h"

#include "ensembles_interface.h"
#include "gpurf.h"

namespace lib_ensembles {
EnsemblesInterface& EnsemblesInterface::GetInstance() {
  static EnsemblesInterface instance;
  return instance;
}

template <typename T>
sp<lib_algorithms::MlAlgorithm<T>> EnsemblesInterface::CreateGpuRfAlgorithm() {
  auto algo = std::make_shared<GpuRf<T>>();
  return AlgorithmsLib::GetInstance().CreateGpuAlgorithm<T>(algo);
}

sp<lib_algorithms::MlAlgorithmParams>
EnsemblesInterface::CreateGpuRfParamPack() {
  auto params =
      AlgorithmsLib::GetInstance().CreateAlgorithmParams(kDteEndMarker);

  // Set default parameters for random forest
  params->Set(kNrTrees, 100);
  params->Set(kNrFeatures, 0);
  params->Set(kMaxDepth, 1000);
  params->Set(kMinNodeSize, 5);
  params->Set(kAlgoType, AlgorithmsLib::kClassification);
  params->Set(kBagging, true);
  params->Set(kEasyEnsemble, false);
  return params;
}

template DLLExport sp<lib_algorithms::MlAlgorithm<float>>
EnsemblesInterface::CreateGpuRfAlgorithm();
template DLLExport sp<lib_algorithms::MlAlgorithm<double>>
EnsemblesInterface::CreateGpuRfAlgorithm();
}