#pragma once
#include "../../lib_algorithms/include/lib_algorithms.h"

namespace lib_ensembles {
class DLLExport EnsemblesInterface {
 public:
  static EnsemblesInterface& GetInstance();

  /**
  * \brief
  *	Create a GpuRf algorithm handle
  *
  * \return
  * ::sp<MlAlgorithm>
  */
  template <typename T>
  sp<lib_algorithms::MlAlgorithm<T>> CreateGpuRfAlgorithm();
  template<typename T>
  sp<lib_algorithms::MlAlgorithm<T>> CreateGpuErtAlgorithm();

  sp<lib_algorithms::MlAlgorithmParams> CreateGpuRfParamPack();
  sp<lib_algorithms::MlAlgorithmParams> CreateGpuErtParamPack();
  enum CommonParams {
    kDevId = 0,

    kCommonEndMarker
  };
  enum DteParams {
    kNrTrees = kCommonEndMarker,
    kAlgoType,
    kNrFeatures,
    kMaxDepth,
    kMinNodeSize,
    kEasyEnsemble,
    kBagging,

    kDteEndMarker
  };

 private:
  EnsemblesInterface() = default;
  ~EnsemblesInterface() = default;
};
}