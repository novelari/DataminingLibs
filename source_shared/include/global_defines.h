#pragma once
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <mutex>
#include <condition_variable>

template <typename T>
using sp = std::shared_ptr<T>;
template <typename T>
using wp = std::weak_ptr<T>;
template <typename T>
using up = std::unique_ptr<T>;
template <typename T>
using col_array = std::vector<T>;
template <typename T1, typename T2>
using col_map = std::map<T1, T2>;
using string = std::string;
using mutex = std::mutex;
using mutex_lock = std::unique_lock<mutex>;
using condition_var = std::condition_variable;