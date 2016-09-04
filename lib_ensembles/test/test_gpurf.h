#pragma once
#include "lib_algorithms.h"
#include "lib_data.h"
#include "lib_models.h"
#include "lib_parsing.h"
#include "lib_ensembles.h"

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

sp<lib_models::MlModel> model;
auto gpurf = ensembles_face.CreateGpuRfAlgorithm<float>();

TEST(lib_ensembles, gpurf_fit) {
  //auto data = parser_face.ParseData<float>(lib_parsing::ParsingInterface::kCsv,
    //                                       raw_data_fit);

  auto data = parser_face.ParseFile<float>(lib_parsing::ParsingInterface::kCsv,
	  "../../runnable_test/spambase.csv");
  auto params = ensembles_face.CreateGpuRfParamPack();
  params->Set(EnsemblesLib::kNrTrees, 100);
  params->Set(EnsemblesLib::kAlgoType, AlgorithmsLib::kClassification);
  EXPECT_NO_THROW(model = gpurf->Fit(data, params););
}

TEST(lib_ensembles, gpurf_predict) {
  //auto data = parser_face.ParseData<float>(lib_parsing::ParsingInterface::kCsv,
    //                                       raw_data_predict);
  auto data = parser_face.ParseFile<float>(lib_parsing::ParsingInterface::kCsv,
                                           "../../runnable_test/spambase.csv");
  auto params = ensembles_face.CreateGpuRfParamPack();
  params->Set(EnsemblesLib::kNrTrees, 100);
  float acc = 0;
  EXPECT_NO_THROW(auto results = gpurf->Predict(data, model, params);
                  acc = results->GetAccuracy(data->GetTargets()););
}
}