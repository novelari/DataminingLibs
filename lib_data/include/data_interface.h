#pragma once
namespace lib_data {
template <typename T>
class MlDataFrame;
template <typename T>
class MlResultData;
class DLLExport DataInterface {
 public:
  static DataInterface& GetInstance();

  template<typename T>
  sp<MlDataFrame<T>> CreateDataFrame();
  template<typename T>
  sp<MlResultData<T>> CreateResultData();

 private:
  DataInterface();
  ~DataInterface();
};
}