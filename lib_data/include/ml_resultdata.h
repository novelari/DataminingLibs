#pragma once

namespace lib_data {
template <typename T>
class DLLExport MlResultData {
 public:
  virtual ~MlResultData() {}

  virtual void AddSingleValue(string name, T value) = 0;
  virtual void AddMultipleValue(string name, col_array<T> vals) = 0;
  virtual void AddPredictions(col_array<col_array<T>> predictions) = 0;

  virtual T GetSingleValue(string name) = 0;
  virtual col_array<T>& GetMultipleValue(string name) = 0;

  virtual string ToString() = 0;
};
}