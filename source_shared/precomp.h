#ifdef WindowsBuild
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#define _WIN32_WINDOWS 0x0601
#define DLLExport __declspec(dllexport)
#define TestExport __declspec(dllexport)
#include <windows.h>
#endif

#ifdef UnixBuild
#define DLLExport __attribute__((visibility("default")))
#define TestExport __attribute__((visibility("default")))
#endif

#include <malloc.h>
#include <memory>
#include <stdlib.h>

#include <cassert>
#include <algorithm>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef WindowsBuild
#include <io.h>
#endif
#ifdef UnixBuild
#include <sys/io.h>
#endif
#include <math.h>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <random>
#include <thread>

#include "global_defines.h"

#include "any_type.hpp"
#include "lock_free_list.hpp"
