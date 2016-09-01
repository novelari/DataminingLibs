#pragma once
#include "ml_parser.h"

namespace lib_parsing {
template <typename T>
class CsvParser : public MlParser<T> {
 public:
  CsvParser();
  virtual ~CsvParser();

  bool Parse(sp<lib_data::MlDataFrame<T>> data, string filepath) override;
  bool Parse(sp<lib_data::MlDataFrame<T>> data,
             col_array<char>& raw_data) override;

 private:
  template <typename Ts>
  bool Parse(sp<lib_data::MlDataFrame<T>> data, Ts& stream);

  class csv_ctype : public std::ctype<char> {
    mask my_table[table_size];

   public:
    csv_ctype(size_t refs = 0) : std::ctype<char>(&my_table[0], false, refs) {
      std::copy_n(classic_table(), table_size, my_table);
      my_table[','] = (mask)space;
      my_table['	'] = (mask)space;
    }
  };
};
}