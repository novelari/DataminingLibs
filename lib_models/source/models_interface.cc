#include "precomp.h"

#include "ml_model_impl.h"
#include "models_interface.h"

namespace lib_models {
ModelsInterface& ModelsInterface::GetInstance() {
  static ModelsInterface instance;
  return instance;
}

sp<MlModel> ModelsInterface::CreateModel() {
  return std::make_shared<MlModelImpl>();
}
}