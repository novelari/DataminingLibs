#include "precomp.h"

#include "ml_dataframe_impl.h"

namespace lib_data {
template <typename T>
MlDataFrameImpl<T>::MlDataFrameImpl() {}

template <typename T>
MlDataFrameImpl<T>::~MlDataFrameImpl() {}

template <typename T>
inline void MlDataFrameImpl<T>::AddTargetData(col_array<T> targets) {
  targets_ = std::move(targets);
}

template <typename T>
void MlDataFrameImpl<T>::AddSampleData(col_array<T> samples) {
  samples_ = std::move(samples);
}

template <typename T>
void MlDataFrameImpl<T>::AddFeatureMap(col_array<col_map<string, T>> col_map) {
  feature_col_maps_ = std::move(col_map);
}

template <typename T>
void MlDataFrameImpl<T>::AddTargetMap(col_map<string, T> col_map) {
  target_col_map_ = std::move(col_map);
}

template <typename T>
int MlDataFrameImpl<T>::GetNrSamples() {
  return int(targets_.size());
}

template <typename T>
int MlDataFrameImpl<T>::GetNrFeatures() {
  return int(feature_col_maps_.size());
}

template <typename T>
int MlDataFrameImpl<T>::GetNrTargets() {
  return int(target_col_map_.size());
}

template <typename T>
const col_array<T>& MlDataFrameImpl<T>::GetTargets() {
  return targets_;
}

template <typename T>
const col_array<T>& MlDataFrameImpl<T>::GetSamples() {
  return samples_;
}

template MlDataFrameImpl<float>::MlDataFrameImpl();
template MlDataFrameImpl<double>::MlDataFrameImpl();
}