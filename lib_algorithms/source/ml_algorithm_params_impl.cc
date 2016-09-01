#include "precomp.h"

#include "ml_algorithm_params_impl.h"

namespace lib_algorithms {
MlAlgorithmParamsImpl::MlAlgorithmParamsImpl(int nr_params) {
  params_ = col_array<sutil::any_type>(nr_params, sutil::any_type());
}
MlAlgorithmParamsImpl::~MlAlgorithmParamsImpl() {}
void MlAlgorithmParamsImpl::SetParam(const int param,
                                     const sutil::any_type value) {
  params_[param] = std::move(value);
}
sutil::any_type& MlAlgorithmParamsImpl::GetParam(const int param) {
  return params_[param];
}
}