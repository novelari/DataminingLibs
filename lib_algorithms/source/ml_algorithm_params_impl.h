#pragma once
#include "ml_algorithm_params.h"

namespace lib_algorithms {
class MlAlgorithmParamsImpl : public MlAlgorithmParams {
 public:
  MlAlgorithmParamsImpl(int nr_params);
  virtual ~MlAlgorithmParamsImpl();

 private:
  void SetParam(const int param, const sutil::any_type value) override;
  sutil::any_type& GetParam(const int param) override;

  col_array<sutil::any_type> params_;
};
}