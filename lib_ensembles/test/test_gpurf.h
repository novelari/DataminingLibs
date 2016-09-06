#pragma once
#include "lib_algorithms.h"
#include "lib_data.h"
#include "lib_ensembles.h"
#include "lib_models.h"
#include "lib_parsing.h"

namespace lib_algorithms {
char csv_data_fit[] = {
    "a1,a2,a3,a4,a5,a6,a7,a8,a9,a10\na,1,2,3,4,5,6,7,8,0\nb,1,2,3,4,5,6,7,8,"
    "1\nc,1,2,3,4,5,6,7,8,0\ne,1,2,3,4,5,6,7,8,1\nf,1,2,3,4,5,6,7,8,0\nd,1,2,3,"
    "4,5,6,7,8,1\ng,1,2,3,4,5,6,7,8,0\nh,1,2,3,4,5,6,7,8,1\ni,1,2,3,4,5,6,7,8,"
    "0\na,1,2,3,4,5,6,7,8,0\nb,1,2,3,4,5,6,7,8,1\nc,1,2,3,4,5,6,7,8,0\ne,1,2,3,"
    "4,5,6,7,8,1\nf,1,2,3,4,5,6,7,8,0\nd,1,2,3,4,5,6,7,8,1\ng,1,2,3,4,5,6,7,8,"
    "0\nh,1,2,3,4,5,6,7,8,1\ni,1,2,3,4,5,6,7,8,0\na,1,2,3,4,5,6,7,8,0\nb,1,2,3,"
    "4,5,6,7,8,1\nc,1,2,3,4,5,6,7,8,0\ne,1,2,3,4,5,6,7,8,1\nf,1,2,3,4,5,6,7,8,"
    "0\nd,1,2,3,4,5,6,7,8,1\ng,1,2,3,4,5,6,7,8,0\nh,1,2,3,4,5,6,7,8,1\ni,1,2,3,"
    "4,5,6,7,8,0\na,1,2,3,4,5,6,7,8,0\nb,1,2,3,4,5,6,7,8,1\nc,1,2,3,4,5,6,7,8,"
    "0\ne,1,2,3,4,5,6,7,8,1\nf,1,2,3,4,5,6,7,8,0\nd,1,2,3,4,5,6,7,8,1\ng,1,2,3,"
    "4,5,6,7,8,0\nh,1,2,3,4,5,6,7,8,1\ni,1,2,3,4,5,6,7,8,0"};
char csv_data_predict[] = {
    "a1,a2,a3,a4,a5,a6,a7,a8,a9,a10\na,1,2,3,4,5,6,7,8,0\nb,1,2,3,4,5,6,7,8,"
    "1\nc,1,2,3,4,5,6,7,8,0\ne,1,2,3,4,5,6,7,8,1\nf,1,2,3,4,5,6,7,8,0\nd,1,2,3,"
    "4,5,6,7,8,1\ng,1,2,3,4,5,6,7,8,0\nh,1,2,3,4,5,6,7,8,1\ni,1,2,3,4,5,6,7,8,"
    "0"};

col_array<char> raw_data_fit(csv_data_fit,
                             csv_data_fit +
                                 sizeof(csv_data_fit) / sizeof(*csv_data_fit));
col_array<char> raw_data_predict(csv_data_predict,
                                 csv_data_predict +
                                     sizeof(csv_data_predict) /
                                         sizeof(*csv_data_predict));

auto &algorithms_face = AlgorithmsLib::GetInstance();
auto &parser_face = ParsingLib::GetInstance();
auto &ensembles_face = EnsemblesLib::GetInstance();

sp<lib_models::MlModel> model_flt;
sp<lib_models::MlModel> model_dbl;
auto gpurf_flt = ensembles_face.CreateGpuRfAlgorithm<float>();
auto gpurf_dbl = ensembles_face.CreateGpuRfAlgorithm<double>();

auto data_fit_raw_dbl = parser_face.ParseData<double>(
    lib_parsing::ParsingInterface::kCsv, raw_data_fit);
auto data_predict_raw_dbl = parser_face.ParseData<double>(
    lib_parsing::ParsingInterface::kCsv, raw_data_predict);

auto data_fit_raw_flt = parser_face.ParseData<float>(
    lib_parsing::ParsingInterface::kCsv, raw_data_fit);
auto data_predict_raw_flt = parser_face.ParseData<float>(
    lib_parsing::ParsingInterface::kCsv, raw_data_predict);

auto data_csv_dbl = parser_face.ParseFile<double>(
    lib_parsing::ParsingInterface::kCsv, "../../runnable_test/spambase.csv");
auto data_csv_flt = parser_face.ParseFile<float>(
    lib_parsing::ParsingInterface::kCsv, "../../runnable_test/spambase.csv");

TEST(lib_ensembles, gpurf_fit_double_rawdata) {
  auto params = ensembles_face.CreateGpuRfParamPack();
  params->Set(EnsemblesLib::kNrTrees, 200);
  params->Set(EnsemblesLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(model_dbl = gpurf_dbl->Fit(data_fit_raw_dbl, params););
}

TEST(lib_ensembles, gpurf_predict_double_rawdata) {
  auto params = ensembles_face.CreateGpuRfParamPack();
  double acc = 0;
  ASSERT_NO_THROW(
      auto results =
          gpurf_dbl->Predict(data_predict_raw_dbl, model_dbl, params);
      acc = results->GetAccuracy(data_predict_raw_dbl->GetTargets()););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}

TEST(lib_ensembles, gpurf_fit_float_rawdata) {
  auto params = ensembles_face.CreateGpuRfParamPack();
  params->Set(EnsemblesLib::kNrTrees, 100);
  params->Set(EnsemblesLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(model_flt = gpurf_flt->Fit(data_fit_raw_flt, params););
}

TEST(lib_ensembles, gpurf_predict_float_rawdata) {
  auto params = ensembles_face.CreateGpuRfParamPack();
  float acc = 0;
  ASSERT_NO_THROW(
      auto results =
          gpurf_flt->Predict(data_predict_raw_flt, model_flt, params);
      acc = results->GetAccuracy(data_predict_raw_flt->GetTargets()););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}

TEST(lib_ensembles, gpurf_fit_float_csvdata) {
  auto params = ensembles_face.CreateGpuRfParamPack();
  params->Set(EnsemblesLib::kNrTrees, 400);
  params->Set(EnsemblesLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(model_flt = gpurf_flt->Fit(data_csv_flt, params););
}

TEST(lib_ensembles, gpurf_predict_float_csvdata) {
  auto params = ensembles_face.CreateGpuRfParamPack();
  float acc = 0;
  ASSERT_NO_THROW(auto results =
                      gpurf_flt->Predict(data_csv_flt, model_flt, params);
                  acc = results->GetAccuracy(data_csv_flt->GetTargets()););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}

TEST(lib_ensembles, gpurf_fit_double_csvdata) {
  auto params = ensembles_face.CreateGpuRfParamPack();
  params->Set(EnsemblesLib::kNrTrees, 100);
  params->Set(EnsemblesLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(model_dbl = gpurf_dbl->Fit(data_csv_dbl, params););
}

TEST(lib_ensembles, gpurf_predict_double_csvdata) {
  auto params = ensembles_face.CreateGpuRfParamPack();
  double acc = 0;
  ASSERT_NO_THROW(auto results =
                      gpurf_dbl->Predict(data_csv_dbl, model_dbl, params);
                  acc = results->GetAccuracy(data_csv_dbl->GetTargets()););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}
}