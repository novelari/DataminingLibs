#define DLLExport
#define TestExport

#include "gtest/gtest.h"

#include "test_gpurf.h"
#include "test_gpuert.h"
#include "test_ml_model.h"
#include "test_gpu_device.h"
#include "test_dataframe.h"
#include "test_csv_parsing.h"

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}