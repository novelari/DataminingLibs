#pragma once

namespace lib_algorithms {
template <typename T>
class MlAlgorithm;
class MlAlgorithmParams;
class DLLExport AlgorithmsInterface {
 public:
  static AlgorithmsInterface& GetInstance();

  enum AlgorithmType { kClassification = 0, kRegression };

  /**
  * \brief
  *	Create a GpuRf algorithm handle
  *
  * \return
  * ::sp<MlAlgorithm>
  */
  template <typename T>
  sp<MlAlgorithm<T>> CreateGpuRfAlgorithm();
  sp<MlAlgorithmParams> CreateGpuRfParamPack();
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
  AlgorithmsInterface();
  ~AlgorithmsInterface();
};
}