#pragma once
#include "test_resources.h"


bool fitData (){
	auto &algorithms_face = AlgorithmsLib::GetInstance();
	auto &ensembles_face = EnsemblesLib::GetInstance();

	sp<lib_models::MlModel> model_flt;
	sp<lib_models::MlModel> model_dbl;
	auto gpurf_flt = ensembles_face.CreateGpuRfAlgorithm<float>();
	auto gpurf_dbl = ensembles_face.CreateGpuRfAlgorithm<double>();

	auto params = ensembles_face.CreateGpuRfParamPack();
	params->Set(EnsemblesLib::kNrTrees, 100);
	params->Set(EnsemblesLib::kAlgoType, AlgorithmsLib::kClassification);
	model_flt =	gpurf_flt->Fit(lib_ensembles::data_fit_raw_flt, params);

	return true;

}
