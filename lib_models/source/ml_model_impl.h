#pragma once
#include "ml_model.h"

namespace lib_models {
class MlModelImpl : public MlModel {
 public:
  MlModelImpl() = default;

 private:
  void AddData(const int id, const sutil::any_type data) override;
  sutil::any_type& GetData(const int id) override;

  col_array<sutil::any_type> data_;
};
}