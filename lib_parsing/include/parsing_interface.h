#pragma once
#include "ml_dataframe.h"
#include "ml_parser.h"

namespace lib_parsing {
class DLLExport ParsingInterface {
 public:
  static ParsingInterface &GetInstance();

  enum ParserType { kCsv, kArff, kRds };

  template <typename T>
  sp<lib_data::MlDataFrame<T>> ParseData(ParserType type,
                                         col_array<char> &raw_data);
  template <typename T>
  sp<lib_data::MlDataFrame<T>> ParseFile(ParserType type, string filepath);

 private:
  template <typename T>
  sp<MlParser<T>> GetParser(ParserType type);

  ParsingInterface();
  ~ParsingInterface();
};
}