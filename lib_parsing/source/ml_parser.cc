#include "precomp.h"

#include "ml_parser.h"

namespace lib_parsing {
template <typename T>
inline MlParser<T>::MlParser() {}

template <typename T>
inline MlParser<T>::~MlParser() {}

template MlParser<float>::MlParser();
template MlParser<double>::MlParser();
template MlParser<float>::~MlParser();
template MlParser<double>::~MlParser();
}