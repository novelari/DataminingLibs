#pragma once

namespace lib_algorithms {
class DLLExport MlAlgorithmParams {
 public:
  virtual ~MlAlgorithmParams() {}

  template <typename T>
  void Set(int param, T value) {
    SetParam(param, sutil::any_type(value));
  }

  template <typename T>
  const T& Get(const int param) {
    return GetParam(param).get_value<T>();
  }

 private:
  virtual void SetParam(const int param, const sutil::any_type value) = 0;
  virtual sutil::any_type& GetParam(const int param) = 0;
};
}