#define DLLExport
#define TestExport

#include <iostream>
#include "test_resources.h"
#include "lib_parsing.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

auto &algorithms_face = AlgorithmsLib::GetInstance();
auto &ensembles_face = EnsemblesLib::GetInstance();
auto &parser_face2 = ParsingLib::GetInstance();

sp<lib_models::MlModel> model_flt;
sp<lib_models::MlModel> model_dbl;
auto gpurf_flt = ensembles_face.CreateGpuRfAlgorithm<float>();
auto gpurf_dbl = ensembles_face.CreateGpuRfAlgorithm<double>();

string stest2 (string a){
	for (int i = 0; i < a.size(); ++i) {
		std::cout<<(char) a[i];
	}
	std::cout<<"\n";

	return a;
}
char* stest3 (char a[]){
	for (int i = 0; i < 20; ++i) {
		std::cout<<(char) a[i];
	}
	string s(a);
	return a;
}
char* stest(char* r){
	for (int i = 0; i < 20; ++i) {
		std::cout<<(char) r[i];
	}
	string str(r);
	return r;
}

void fitData(string data_path) {

	auto params = ensembles_face.CreateGpuRfParamPack();
	params->Set(EnsemblesLib::kNrTrees, 100);
	params->Set(EnsemblesLib::kAlgoType, AlgorithmsLib::kClassification);

	auto data = parser_face2.ParseFile<float>(
			lib_parsing::ParsingInterface::kCsv, data_path);

	model_flt = gpurf_flt->Fit(data, params);

}

float predictData(string data_path) {

	auto params = ensembles_face.CreateGpuRfParamPack();
	float acc = 0;
	auto data = parser_face2.ParseFile<float>(
			lib_parsing::ParsingInterface::kCsv, data_path);
	auto results = gpurf_flt->Predict(data, model_flt, params);
	acc = results->GetAccuracy(data->GetTargets());

	return acc;
}

///////////////////////
void fitData2(char* csv_data_fit) {


	std::cout<<"_____________"<<csv_data_fit;

	auto params = ensembles_face.CreateGpuRfParamPack();
	params->Set(EnsemblesLib::kNrTrees, 100);
	params->Set(EnsemblesLib::kAlgoType, AlgorithmsLib::kClassification);

	col_array<char> raw_data_fit(csv_data_fit,
	                             csv_data_fit +
	                                 sizeof(csv_data_fit) / sizeof(*csv_data_fit));
	auto data_fit_raw_flt = parser_face2.ParseData<float>(
	    lib_parsing::ParsingInterface::kCsv, raw_data_fit);

	model_flt = gpurf_flt->Fit(data_fit_raw_flt, params);

}

float predictData2(char* csv_data_predict) {

	auto params = ensembles_face.CreateGpuRfParamPack();
	float acc = 0;

	col_array<char> raw_data_predict(csv_data_predict,
	                                 csv_data_predict +
	                                     sizeof(csv_data_predict) /
	                                         sizeof(*csv_data_predict));

	auto data_predict_raw_flt = parser_face2.ParseData<float>(
	    lib_parsing::ParsingInterface::kCsv, raw_data_predict);

	auto results = gpurf_flt->Predict(data_predict_raw_flt, model_flt, params);
	acc = results->GetAccuracy(data_predict_raw_flt->GetTargets());

	return acc;
}

float executeRF (char* data){
	fitData2(data);
	return predictData2(data);
}


extern "C" float cRF(char*, int); // one way
float cRF (char* d, int l){
  printf("%s\n", d );
  return executeRF(d);
}
////////////////////
float executeRF_File(int* data_path, int l) {
	  std::cout<<l<<"\n";
	  char* t = new char[l];

	  for (int i = 0; i < l; i++){
	    t[i] = (char) data_path[i];
	  }

	  for (int i = 0; i < l; i++){
	    std::cout<<t[i];
	  }

	std::string path1(t);
	std::string path = "";
	for (int i = 0; i < path1.size() -1; i++){
		    path += path1[i];
		  }
	std::cout<<"The Path :"<<path;

	fitData(path);
	return predictData(path);
}

int main (){

	return 0;
}
