#include "precomp.h"

#include "data_interface.h"
#include "ml_dataframe_impl.h"
#include "ml_resultdata_impl.h"

namespace lib_data {
DataInterface& DataInterface::GetInstance() {
  static DataInterface instance;
  return instance;
}

template <typename T>
sp<MlDataFrame<T>> DataInterface::CreateDataFrame() {
  return std::make_shared<MlDataFrameImpl<T>>();
}

template <typename T>
inline sp<MlResultData<T>> DataInterface::CreateResultData() {
  return std::make_shared<MlResultDataImpl<T>>();
}

DataInterface::DataInterface() {}
DataInterface::~DataInterface() {}

template DLLExport sp<MlDataFrame<float>> DataInterface::CreateDataFrame();
template DLLExport sp<MlDataFrame<double>> DataInterface::CreateDataFrame();

template DLLExport sp<MlResultData<float>> DataInterface::CreateResultData();
template DLLExport sp<MlResultData<double>> DataInterface::CreateResultData();
}