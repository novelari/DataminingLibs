#pragma once
#include <fstream>
#include "lib_data.h"
#include "lib_parsing.h"

namespace lib_parsing {
char csv_data[] = {
    "a1,a2,a3,a4,a5,a6,a7,a8,a9,a10\na,1,2,3,4,5,6,7,8,0\nb,1,2,3,4,5,6,7,8,"
    "1\nc,1,2,3,4,5,6,7,8,0\ne,1,2,3,4,5,6,7,8,1\nf,1,2,3,4,5,6,7,8,0\nd,1,2,3,"
    "4,5,6,7,8,1\ng,1,2,3,4,5,6,7,8,0\nh,1,2,3,4,5,6,7,8,1\ni,1,2,3,4,5,6,7,8,"
    "0"};
char csv_data_bad[] = {"a1,2,1,2\n2,122,1"};

auto &parsininterface = ParsingInterface::GetInstance();
TEST(lib_parsing, csv_parsinfile) {
  {
    std::ofstream open("./test_data.csv");
    open << csv_data;
    open.close();
  }

  {
    std::ofstream open("./test_data_fail.csv");
    open << csv_data_bad;
    open.close();
  }

  ASSERT_NO_THROW({
    auto data = parsininterface.ParseFile<float>(ParsingInterface::kCsv,
                                                 "./test_data.csv");
  });
  ASSERT_NO_THROW({
    auto data2 = parsininterface.ParseFile<double>(ParsingInterface::kCsv,
                                                   "./test_data.csv");
  });

  ASSERT_ANY_THROW({
    auto data = parsininterface.ParseFile<float>(ParsingInterface::kCsv,
                                                 "./test_data_fail.csv");
  });
  ASSERT_ANY_THROW({
    auto data2 = parsininterface.ParseFile<double>(ParsingInterface::kCsv,
                                                   "./test_data_fail.csv");
  });
}

TEST(lib_parsing, csv_parsindata) {
  col_array<char> raw_data(csv_data,
                           csv_data + sizeof(csv_data) / sizeof(*csv_data));
  col_array<char> raw_data_fail(
      csv_data_bad,
      csv_data_bad + sizeof(csv_data_bad) / sizeof(*csv_data_bad));

  ASSERT_NO_THROW({
    auto data =
        parsininterface.ParseData<float>(ParsingInterface::kCsv, raw_data);
  });
  ASSERT_NO_THROW({
    auto data2 =
        parsininterface.ParseData<double>(ParsingInterface::kCsv, raw_data);
  });

  ASSERT_ANY_THROW({
    auto data =
        parsininterface.ParseData<float>(ParsingInterface::kCsv, raw_data_fail);
  });
  ASSERT_ANY_THROW({
    auto data2 = parsininterface.ParseData<double>(ParsingInterface::kCsv,
                                                   raw_data_fail);
  });
}
}