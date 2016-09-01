#pragma once
#include "lib_data.h"

namespace lib_data {
auto& data_interface = DataInterface::GetInstance();
TEST(lib_data, create_float_dataframe) {
  auto data = data_interface.CreateDataFrame<float>();
  ASSERT_EQ(data->GetNrFeatures(), 0);
  ASSERT_EQ(data->GetNrSamples(), 0);
}
TEST(lib_data, create_double_dataframe) {
  auto data = data_interface.CreateDataFrame<double>();
  ASSERT_EQ(data->GetNrFeatures(), 0);
  ASSERT_EQ(data->GetNrSamples(), 0);
}
}