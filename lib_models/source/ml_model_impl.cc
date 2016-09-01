#include "precomp.h"

#include "ml_model_impl.h"

namespace lib_models {
void MlModelImpl::AddData(const int id, const sutil::any_type data) {
  while (id >= data_.size()) data_.emplace_back(sutil::any_type());
  data_[id] = std::move(data);
}
sutil::any_type& MlModelImpl::GetData(const int id) { return data_[id]; }
}