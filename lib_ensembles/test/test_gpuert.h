#pragma once
#include "test_resources.h"

namespace test_gpuert {
auto &algorithms_face = AlgorithmsLib::GetInstance();
auto &ensembles_face = EnsemblesLib::GetInstance();

sp<lib_models::MlModel> gpuert_model_flt;
sp<lib_models::MlModel> gpuert_model_dbl;
auto gpuert_flt = ensembles_face.CreateGpuErtAlgorithm<float>();
auto gpuert_dbl = ensembles_face.CreateGpuErtAlgorithm<double>();

TEST(lib_ensembles_gpuert, fit_double_rawdata) {
  auto params = ensembles_face.CreateGpuErtParamPack();
  params->Set(EnsemblesLib::kNrTrees, 200);
  params->Set(EnsemblesLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(gpuert_model_dbl = gpuert_dbl->Fit(
                      lib_ensembles::data_fit_raw_dbl, params););
}

TEST(lib_ensembles_gpuert, predict_double_rawdata) {
  auto params = ensembles_face.CreateGpuErtParamPack();
  double acc = 0;
  ASSERT_NO_THROW(auto results =
                      gpuert_dbl->Predict(lib_ensembles::data_predict_raw_dbl,
                                          gpuert_model_dbl, params);
                  acc = results->GetAccuracy(
                      lib_ensembles::data_predict_raw_dbl->GetTargets()););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}

TEST(lib_ensembles_gpuert, fit_float_rawdata) {
  auto params = ensembles_face.CreateGpuErtParamPack();
  params->Set(EnsemblesLib::kNrTrees, 100);
  params->Set(EnsemblesLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(gpuert_model_flt = gpuert_flt->Fit(
                      lib_ensembles::data_fit_raw_flt, params););
}

TEST(lib_ensembles_gpuert, predict_float_rawdata) {
  auto params = ensembles_face.CreateGpuErtParamPack();
  float acc = 0;
  ASSERT_NO_THROW(auto results =
                      gpuert_flt->Predict(lib_ensembles::data_predict_raw_flt,
                                          gpuert_model_flt, params);
                  acc = results->GetAccuracy(
                      lib_ensembles::data_predict_raw_flt->GetTargets()););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}

TEST(lib_ensembles_gpuert, fit_float_csvdata) {
  auto params = ensembles_face.CreateGpuErtParamPack();
  params->Set(EnsemblesLib::kNrTrees, 400);
  params->Set(EnsemblesLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(gpuert_model_flt =
                      gpuert_flt->Fit(lib_ensembles::data_csv_flt, params););
}

TEST(lib_ensembles_gpuert, predict_float_csvdata) {
  auto params = ensembles_face.CreateGpuErtParamPack();
  float acc = 0;
  ASSERT_NO_THROW(
      auto results = gpuert_flt->Predict(lib_ensembles::data_csv_flt,
                                         gpuert_model_flt, params);
      acc = results->GetAccuracy(lib_ensembles::data_csv_flt->GetTargets()););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}

TEST(lib_ensembles_gpuert, fit_double_csvdata) {
  auto params = ensembles_face.CreateGpuErtParamPack();
  params->Set(EnsemblesLib::kNrTrees, 100);
  params->Set(EnsemblesLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(gpuert_model_dbl =
                      gpuert_dbl->Fit(lib_ensembles::data_csv_dbl, params););
}

TEST(lib_ensembles_gpuert, predict_double_csvdata) {
  auto params = ensembles_face.CreateGpuErtParamPack();
  double acc = 0;
  ASSERT_NO_THROW(
      auto results = gpuert_dbl->Predict(lib_ensembles::data_csv_dbl,
                                         gpuert_model_dbl, params);
      acc = results->GetAccuracy(lib_ensembles::data_csv_dbl->GetTargets()););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}
}