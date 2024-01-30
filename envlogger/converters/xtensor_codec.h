// Copyright 2024 DeepMind Technologies Limited..
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Functions for manipulating storage.proto:{Datum, Data} objects.
//
// It can produce Datum objects from C++ types using the Encode() method, and
// conversely it can parse Datum objects into C++ types using the Decode()
// method. Besides "scalar" types, this library also supports handling
// xt::xarray objects, which are multidimensional arrays analogous to Python's
// numpy ndarrays.
//
// Please see xtensor_codec_test.cc for usage examples.
//
// Consumers of Decode() may also want to use a library for writing visitors to
// reduce the boilerplate when dealing with the output. Please see
// make_visitor.h in this same directory.

#ifndef THIRD_PARTY_PY_ENVLOGGER_CONVERTERS_XTENSOR_CODEC_H_
#define THIRD_PARTY_PY_ENVLOGGER_CONVERTERS_XTENSOR_CODEC_H_

#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/strings/cord.h"
#include <gmpxx.h>
#include "envlogger/proto/storage.pb.h"
#include "xtensor/xarray.hpp"

namespace envlogger {

// BasicType is the set union of all types that are supported in each Datum.

using BasicType =
    std::variant<float, double, int32_t, int64_t, uint32_t, uint64_t, bool,
                 std::string, absl::Cord, mpz_class, int8_t, int16_t, uint8_t,
                 uint16_t, xt::xarray<float>, xt::xarray<double>,
                 xt::xarray<int32_t>, xt::xarray<int64_t>, xt::xarray<uint32_t>,
                 xt::xarray<uint64_t>, xt::xarray<bool>,
                 xt::xarray<std::string>, xt::xarray<absl::Cord>,
                 xt::xarray<mpz_class>, xt::xarray<int8_t>, xt::xarray<int16_t>,
                 xt::xarray<uint8_t>, xt::xarray<uint16_t>>;

// Encode() transforms a C++ object into a Datum that can be serialized to disk.

// Scalars.
Datum Encode(const float value);
Datum Encode(const double value);
Datum Encode(const int32_t value);
Datum Encode(const int64_t value);
Datum Encode(const uint32_t value);
Datum Encode(const uint64_t value);
Datum Encode(const bool value);
Datum Encode(const std::string& value);
// Define an explicit `const char*` overload to avoid using the `bool` overload.
// Please see: https://godbolt.org/z/Pntp3S
Datum Encode(const char* value);
// We use Cord to represent bytes.
Datum Encode(absl::Cord value);
// GNU gmp's mpz_class is used to represent arbitrary precision ints.
Datum Encode(mpz_class value);
Datum Encode(const int8_t value);
Datum Encode(const int16_t value);
Datum Encode(const uint8_t value);
Datum Encode(const uint16_t value);
// xt::xarrays.
// as_bytes indicates that `value` will be written as a big chunk of big-endian
// floats. This is much faster than individually writing float entries in the
// proto field, but the output is (obviously) not human-readable. This is the
// native format for Numpy's `frombuffer` method.
Datum Encode(const xt::xarray<float>& value, bool as_bytes = true);
Datum Encode(const xt::xarray<double>& value);
Datum Encode(const xt::xarray<int32_t>& value);
Datum Encode(const xt::xarray<int64_t>& value);
Datum Encode(const xt::xarray<uint32_t>& value);
Datum Encode(const xt::xarray<uint64_t>& value);
Datum Encode(const xt::xarray<bool>& value);
Datum Encode(const xt::xarray<std::string>& value);
Datum Encode(const xt::xarray<absl::Cord>& value);
Datum Encode(const xt::xarray<mpz_class>& value);
Datum Encode(const xt::xarray<int8_t>& value);
Datum Encode(const xt::xarray<int16_t>& value);
Datum Encode(const xt::xarray<uint8_t>& value);
Datum Encode(const xt::xarray<uint16_t>& value);

// Decode() parses a Datum proto and maybe returns a BasicType.
std::optional<BasicType> Decode(const Datum& datum);

// A non-owning view of envlogger::Data.
class DataView {
 public:
  using value_type = Data;

  // An iterator for arrays and tuples.
  class const_iterator {
   public:
    const_iterator(const DataView* view, int index)
        : view_(view), index_(index) {}

    using iterator_category = std::input_iterator_tag;
    using value_type = Data;
    using difference_type = ptrdiff_t;
    using pointer = const value_type*;
    using reference = const value_type&;

    bool operator==(const const_iterator& rhs) const;
    bool operator!=(const const_iterator& rhs) const;
    const const_iterator& operator++();
    const Data& operator*() const;

   private:
    const DataView* view_ = nullptr;
    // Current index for iteration.
    int index_ = 0;
  };

  // `data` MUST outlive this DataView instance.
  explicit DataView(const Data* data);

  // Returns the number of elements that this view has.
  // Returns 0 if data is nullptr or Type() == Data::kDatum.
  size_t size() const;

  // Returns whether this view is empty.
  bool empty() const { return size() == 0; }

  const Data* data() const { return data_; }
  const Data& operator*() const { return *data(); }
  const Data* operator->() const { return data(); }

  // Returns a DataView of value[index] if the value of the data is an array
  // or tuple.
  //
  // Dies if out of bound or data_ is not subscriptable.
  DataView operator[](int index) const;

  // Returns a DataView of value[key] if the value of this data is a dict.
  //
  // Dies if the key is not in the dict or data_ is not a dict.
  //
  // Notice that we take a const string ref instead of a string view to avoid a
  // costly conversion in order to lookup the internal proto map.
  DataView operator[](const std::string& key) const;

  const_iterator begin() const;
  const_iterator end() const;

  // Accessor for dictionary contents.
  //
  // Also supports iteration like:
  // for (const auto [key, value] : my_view.items()) {
  //   ...
  // }
  const google::protobuf::Map<std::string, Data>& items() const;

 private:
  const Data* data_ = nullptr;
};

}  // namespace envlogger

#endif  // THIRD_PARTY_PY_ENVLOGGER_CONVERTERS_XTENSOR_CODEC_H_
