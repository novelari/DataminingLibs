#pragma once

namespace lib_data {
template <typename T>
class DLLExport MlDataFrame {
 public:
  virtual ~MlDataFrame() {}

  virtual void AddTargetData(col_array<T> targets) = 0;
  virtual void AddSampleData(col_array<T> samples) = 0;
  virtual void AddFeatureMap(col_array<col_map<string, T>> col_map) = 0;
  virtual void AddTargetMap(col_map<string, T> col_map) = 0;

  virtual int GetNrSamples() = 0;
  virtual int GetNrFeatures() = 0;
  virtual int GetNrTargets() = 0;
  virtual const col_array<T>& GetTargets() = 0;
  virtual const col_array<T>& GetSamples() = 0;
};
}