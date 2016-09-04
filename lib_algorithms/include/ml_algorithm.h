#pragma once

namespace lib_models {
class MlModel;
}
namespace lib_data {
template <typename T>
class MlDataFrame;
template <typename T>
class MlResultData;
}

namespace lib_algorithms {
class MlAlgorithmParams;
template <typename T>
class DLLExport MlAlgorithm {
 public:
  MlAlgorithm();
  virtual ~MlAlgorithm();

  /**
  *	\brief
  *	Fit a model to a data set.
  *	\param
  *	data - Data to be fited into model.
  *	\returns
  *	A shared pointer to the fited model.
  */
  virtual sp<lib_models::MlModel> Fit(sp<lib_data::MlDataFrame<T>> data,
                                      sp<MlAlgorithmParams> params) = 0;
  /**
  *	\brief
  *	Predict a data frame using a model.
  * \param
  * data - Data to be predicted by a model
  * model - The model used to perform the predicted of the data.
  * \returns
  * A shared pointer to the resulting predictions
  */
  virtual sp<lib_data::MlResultData<T>> Predict(
      sp<lib_data::MlDataFrame<T>> data, sp<lib_models::MlModel> model,
      sp<MlAlgorithmParams> params) = 0;

  virtual sp<lib_models::MlModel> AggregateModels(
      col_array<sp<lib_models::MlModel>> models) = 0;
  virtual col_array<sp<lib_models::MlModel>> SplitModel(
      sp<lib_models::MlModel> model, const int parts) = 0;
  virtual sp<lib_data::MlResultData<T>> AggregateResults(
      col_array<sp<lib_data::MlResultData<T>>> results) = 0;
  virtual col_array<sp<MlAlgorithmParams>> SplitParameterPack(
      sp<MlAlgorithmParams> params, const int parts) = 0;
};
}