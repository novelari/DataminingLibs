#pragma once
#include "ml_dataframe.h"

namespace lib_parsing {
template <typename T>
class DLLExport MlParser {
 public:
  MlParser();
  virtual ~MlParser();

  virtual bool Parse(sp<lib_data::MlDataFrame<T>> data,
                     string filepath) = 0;
  virtual bool Parse(sp<lib_data::MlDataFrame<T>> data,
                     col_array<char> &raw_data) = 0;
};
}