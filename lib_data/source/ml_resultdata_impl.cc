#include "precomp.h"

#include "ml_resultdata_impl.h"

namespace lib_data {
template <typename T>
MlResultDataImpl<T>::MlResultDataImpl() {}
template <typename T>
inline void MlResultDataImpl<T>::AddSingleValue(string name, T value) {
  single_val_[name] = value;
}
template <typename T>
inline void MlResultDataImpl<T>::AddMultipleValue(string name,
                                                  col_array<T> vals) {
  multiple_val_[name] = std::move(vals);
}
template <typename T>
void MlResultDataImpl<T>::AddPredictions(col_array<col_array<T>> predictions) {
  if (predictions_.empty())
    predictions_ = std::move(predictions);
}

template <typename T>
T MlResultDataImpl<T>::GetAccuracy(const col_array<T> &targets) {
  T acc = 0;
  for (int i = 0; i < targets.size(); ++i) {
    T high = 0;
    T target = 0;
    for (int ii = 0; ii < predictions_[i].size(); ++ii) {
      if (predictions_[i][ii] > high) {
        high = predictions_[i][ii];
        target = T(ii);
      }
    }
    if (std::abs(target - targets[i]) < 0.001) acc += 1;
  }
  return acc / targets.size();
}

template <typename T>
inline T MlResultDataImpl<T>::GetSingleValue(string name) {
  return single_val_[name];
}
template <typename T>
inline col_array<T> &MlResultDataImpl<T>::GetMultipleValue(string name) {
  return multiple_val_[name];
}

template <typename T>
string MlResultDataImpl<T>::ToString() {
  string result = "";
  for (auto &pair : single_val_)
    result += pair.first + std::to_string(pair.second) + "\r\n";
  for (auto &pair : multiple_val_) {
    result += pair.first + "[";
    for (auto &val : pair.second) result += std::to_string(val) + ",";
    result += "]\r\n";
  }
  return result;
}

template <typename T>
MlResultData<T> &MlResultDataImpl<T>::operator+=(const MlResultData<T> &rhs) {
  auto rhs_ref = static_cast<const MlResultDataImpl<T> *>(&rhs);
  for (int i = 0; i < predictions_.size(); ++i) {
    for (int ii = 0; ii < predictions_[i].size(); ++ii) {
      predictions_[i][ii] += rhs_ref->predictions_[i][ii];
    }
  }
  return *this;
}

template MlResultDataImpl<float>::MlResultDataImpl();
template MlResultDataImpl<double>::MlResultDataImpl();
}