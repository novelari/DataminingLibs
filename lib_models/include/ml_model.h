#pragma once
namespace sutil {
struct any_type;
}

namespace lib_models {
class DLLExport MlModel {
 public:
  virtual ~MlModel() = default;

  template <typename T>
  void Add(const int id, const T& data) {
    AddData(id, sutil::any_type(data));
  }

  template <typename T>
  T& Get(const int id) {
    return GetData(id).get_value<T>();
  }

 private:
  virtual void AddData(const int id, const sutil::any_type data) = 0;
  virtual sutil::any_type& GetData(const int id) = 0;
};
}