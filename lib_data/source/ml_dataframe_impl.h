#pragma once
#include "ml_dataframe.h"

namespace lib_data {
template <typename T>
class MlDataFrameImpl : public MlDataFrame<T> {
 public:
  MlDataFrameImpl();
  virtual ~MlDataFrameImpl();

  void AddTargetData(col_array<T> targets) override;
  void AddSampleData(col_array<T> samples) override;
  void AddFeatureMap(col_array<col_map<string, T>> col_map) override;
  void AddTargetMap(col_map<string, T> col_map) override;

  int GetNrSamples() override;
  int GetNrFeatures() override;
  int GetNrTargets() override;
  const col_array<T>& GetTargets() override;
  const col_array<T>& GetSamples() override;

 private:
  col_array<T> targets_;
  col_array<T> samples_;
  col_map<string, T> target_col_map_;
  col_array<col_map<string, T>> feature_col_maps_;
};
}