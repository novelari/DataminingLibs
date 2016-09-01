#include "precomp.h"

#include "ml_algorithm.h"

namespace lib_algorithms {
template <typename T>
MlAlgorithm<T>::MlAlgorithm() {}
template <typename T>
MlAlgorithm<T>::~MlAlgorithm() {}

template MlAlgorithm<float>::MlAlgorithm();
template MlAlgorithm<double>::MlAlgorithm();
}