#pragma once
#include "lib_models.h"

namespace lib_models {
auto &model_interface = ModelsInterface::GetInstance();
TEST(lib_models, ml_model_impl) {
  auto model = model_interface.CreateModel();
  col_array<int> t_vec = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
  model->Add(0, t_vec);
  auto check_vec = model->Get<col_array<int>>(0);
  for (int i = 0; i < t_vec.size(); ++i) ASSERT_EQ(t_vec[i], check_vec[i]);
}
}
