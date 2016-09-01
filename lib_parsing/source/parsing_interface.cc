#include "precomp.h"

#include "csv_parser.h"
#include "lib_data.h"
#include "lib_core.h"
#include "parsing_interface.h"

namespace lib_parsing {
ParsingInterface &ParsingInterface::GetInstance() {
  static ParsingInterface instance;
  return instance;
}

template <typename T>
sp<lib_data::MlDataFrame<T>> ParsingInterface::ParseData(
    ParserType type, col_array<char> &raw_data) {
  auto parser = GetParser<T>(type);
  auto data = DataLib::GetInstance().CreateDataFrame<T>();
  if (!parser->Parse(data, raw_data))
    CoreLib::GetInstance().ThrowException("Parsing failed.");
  return data;
}

template <typename T>
sp<lib_data::MlDataFrame<T>> ParsingInterface::ParseFile(ParserType type,
                                                         string filepath) {
  auto parser = GetParser<T>(type);
  auto data = DataLib::GetInstance().CreateDataFrame<T>();
  if (!parser->Parse(data, filepath))
    CoreLib::GetInstance().ThrowException("Parsing failed.");
  return data;
}

template <typename T>
inline sp<MlParser<T>> ParsingInterface::GetParser(ParserType type) {
  sp<MlParser<T>> parser;
  switch (type) {
    case kCsv:
      parser = std::make_shared<CsvParser<T>>();
      break;
  }
  return parser;
}

ParsingInterface::ParsingInterface() {}
ParsingInterface::~ParsingInterface() {}

template DLLExport sp<lib_data::MlDataFrame<float>> ParsingInterface::ParseData(
    ParserType type, col_array<char> &raw_data);
template DLLExport sp<lib_data::MlDataFrame<double>>
ParsingInterface::ParseData(ParserType type, col_array<char> &raw_data);
template DLLExport sp<lib_data::MlDataFrame<float>> ParsingInterface::ParseFile(
    ParserType type, string filepath);
template DLLExport sp<lib_data::MlDataFrame<double>>
ParsingInterface::ParseFile(ParserType type, string filepath);
}