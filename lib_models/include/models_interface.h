#pragma once

namespace lib_models {
class DLLExport ModelsInterface {
 public:
  static ModelsInterface& GetInstance();

  sp<MlModel> CreateModel();
  enum GpuDteModel {
    kNrTrees,
	kNrTargets,
	kNrFeatures,
    kModelType,

    kNodeArray,
	kProbArray,

    kGpuDteModelEndMarker
  };

 private:
  ModelsInterface() = default;
  ~ModelsInterface() = default;
};
}