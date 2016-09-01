#pragma once

namespace lib_algorithms {
template <typename T>
class MlAlgorithm;
class MlAlgorithmParams;
class DLLExport AlgorithmsInterface {
 public:
  static AlgorithmsInterface& GetInstance();

  enum AlgorithmType { kClassification = 0, kRegression };

  sp<MlAlgorithmParams> CreateAlgorithmParams(int size);

  template <typename T>
  sp<MlAlgorithm<T>> CreateGpuAlgorithm(sp<MlAlgorithm<T>> algo);

 private:
  AlgorithmsInterface() {}
  ~AlgorithmsInterface() {}
};
}