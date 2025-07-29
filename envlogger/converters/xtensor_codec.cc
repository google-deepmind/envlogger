// Copyright 2025 DeepMind Technologies Limited..
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

#include "envlogger/converters/xtensor_codec.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "glog/logging.h"
#include "google/protobuf/repeated_field.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include <gmpxx.h>
#include "riegeli/bytes/string_writer.h"
#include "riegeli/endian/endian_reading.h"
#include "riegeli/endian/endian_writing.h"
#include "xtensor/xview.hpp"

namespace envlogger {
namespace {

// Dimension size reserved for scalars. Please see proto definition.
constexpr int SCALAR_DIM_SIZE = -438;

std::vector<int> ShapeVector(const envlogger::Datum::Shape& shape) {
  std::vector<int> v;
  for (const auto& d : shape.dim()) {
    v.push_back(d.size());
  }
  return v;
}

bool IsScalar(const std::vector<int>& shape) {
  return shape.size() == 1 && shape[0] == SCALAR_DIM_SIZE;
}

// Converts `value` arbitrary precision integer to a binary string.
std::string FromMpzClass(mpz_class value) {
  auto z = value.get_mpz_t();
  const int num_bits = mpz_sizeinbase(z, 2);
  const int num_bytes = num_bits / 8 + 1;
  const bool is_negative = mpz_sgn(z) == -1;
  if (is_negative) {
    // NOTE: We cannot reinterpret cast z from a signed to an unsigned value so
    // we convert it by hand using the formula:
    // z_unsigned = 2^num_bytes(z) + z
    mpz_t max_int;
    mpz_init(max_int);
    mpz_pow_ui(max_int, /*base=*/mpz_class(2).get_mpz_t(), num_bytes * 8);
    mpz_add(z, max_int, z);
    mpz_clear(max_int);
  }
  std::string output(num_bytes, '\0');
  const int should_pad_first_byte = num_bits % 8 ? 0 : 1;
  mpz_export(/*rop=*/output.data() + should_pad_first_byte,
             /*countp=*/nullptr, /*order=*/1,
             /*size=*/1, /*endian=*/1, /*nails=*/0, z);
  return output;
}

// Converts the binary representation `bytes` to an arbitrary precision integer.
mpz_class ToMpzClass(const absl::string_view bytes) {
  const bool is_padded = bytes[0] == '\0';
  mpz_t z;
  mpz_init(z);
  mpz_import(/*rop=*/z, /*count=*/bytes.size() + (is_padded ? -1 : 0),
             /*order=*/1,
             /*size=*/1, /*endian=*/1, /*nails=*/0,
             bytes.data() + (is_padded ? 1 : 0));
  const bool is_negative = static_cast<uint8_t>(bytes[0]) > 127;
  if (is_negative) {
    // NOTE: We cannot reinterpret cast from an unsigned to a signed integer so
    // we convert it by hand using the formula:
    // z_signed = z - 2^num_bytes(z)
    mpz_t max_int;
    mpz_init(max_int);
    mpz_pow_ui(max_int, /*base=*/mpz_class(2).get_mpz_t(), bytes.size() * 8);
    mpz_sub(z, z, max_int);
    mpz_clear(max_int);
  }
  mpz_class output(z);
  mpz_clear(z);  // Free up memory taken by `z`.
  return output;
}

template <typename T>
xt::xarray<T> FillXarrayValues(const google::protobuf::RepeatedField<T>& values,
                               const std::vector<int>& shape) {
  xt::xarray<T> output;
  output.resize({static_cast<uint64_t>(values.size())});
  for (int i = 0; i < values.size(); ++i) {
    output(i) = values[i];
  }
  output.reshape(shape);
  return output;
}

template <typename T>
xt::xarray<T> FillXarrayValues(const google::protobuf::RepeatedPtrField<T>& values,
                               const std::vector<int>& shape) {
  xt::xarray<T> output;
  output.resize({static_cast<uint64_t>(values.size())});
  for (int i = 0; i < values.size(); ++i) {
    output(i) = values[i];
  }
  output.reshape(shape);
  return output;
}

template <typename T>
inline T LoadScalar(const char* p) {
  constexpr int kSize = sizeof(T);
  static_assert(std::is_integral<T>::value &&
                    (kSize == 1 || kSize == 2 || kSize == 4 || kSize == 8),
                "T needs to be an integral type with byte size 1, 2, 4, or 8.");
  return riegeli::ReadBigEndian<T>(p);
}

// Specialization of the above for floats.
template <>
inline float LoadScalar(const char* p) {
  return riegeli::ReadBigEndianFloat(p);
}

template <typename T>
inline void StoreInteger(T value, char* p) {
  constexpr int kSize = sizeof(T);
  static_assert(std::is_integral<T>::value &&
                    (kSize == 1 || kSize == 2 || kSize == 4 || kSize == 8),
                "T needs to be an integral type with byte size 1, 2, 4, or 8.");
  riegeli::WriteBigEndian<T>(value, p);
}

template <typename T>
xt::xarray<T> FillXarrayValuesBigEndian(const absl::string_view values,
                                        const std::vector<int>& shape) {
  const uint64_t num_elements = values.size() / sizeof(T);
  xt::xarray<T> output;
  output.resize({num_elements});
  for (uint64_t i = 0; i < num_elements * sizeof(T); i += sizeof(T)) {
    output(i / sizeof(T)) = LoadScalar<T>(values.substr(i, sizeof(T)).data());
  }
  output.reshape(shape);
  return output;
}

xt::xarray<absl::Cord> FillXarrayValuesCord(
    const google::protobuf::RepeatedPtrField<std::string>& values,
    const std::vector<int>& shape) {
  xt::xarray<absl::Cord> output;
  for (int i = 0; i < values.size(); ++i) {
    output(i) = absl::Cord(values[i]);
  }
  output.reshape(shape);
  return output;
}

xt::xarray<mpz_class> FillXarrayValuesMpzClass(
    const google::protobuf::RepeatedPtrField<std::string>& values,
    const std::vector<int>& shape) {
  xt::xarray<mpz_class> output;
  for (int i = 0; i < values.size(); ++i) {
    output(i) = ToMpzClass(values[i]);
  }
  output.reshape(shape);
  return output;
}

template <typename T>
std::optional<BasicType> DecodeValues(const google::protobuf::RepeatedField<T>& values,
                                      const std::vector<int>& shape,
                                      bool is_scalar) {
  if (values.empty()) return std::nullopt;

  if (is_scalar) {
    return values[0];
  } else {
    return FillXarrayValues(values, shape);
  }
}

template <typename T>
std::optional<BasicType> DecodeValues(const google::protobuf::RepeatedPtrField<T>& values,
                                      const std::vector<int>& shape,
                                      bool is_scalar) {
  if (values.empty()) return std::nullopt;

  if (is_scalar) {
    return values[0];
  } else {
    return FillXarrayValues(values, shape);
  }
}

template <typename T>
std::optional<BasicType> DecodeValuesBigEndian(const std::string& values,
                                               const std::vector<int>& shape,
                                               bool is_scalar) {
  if (values.empty()) return std::nullopt;

  if (is_scalar) {
    return LoadScalar<T>(values.data());
  } else {
    return FillXarrayValuesBigEndian<T>(values, shape);
  }
}

template <typename T>
std::optional<BasicType> DecodeValues(const std::string& values,
                                      const std::vector<int>& shape,
                                      bool is_scalar) {
  return DecodeValuesBigEndian<T>(values, shape, is_scalar);
}

}  // namespace

////////////////////////////////////////////////////////////////////////////////
// Encode()
////////////////////////////////////////////////////////////////////////////////

// Scalars

Datum Encode(const float value) {
  Datum datum;
  datum.mutable_shape()->add_dim()->set_size(SCALAR_DIM_SIZE);
  datum.mutable_values()->add_float_values(value);
  return datum;
}

Datum Encode(const double value) {
  Datum datum;
  datum.mutable_shape()->add_dim()->set_size(SCALAR_DIM_SIZE);
  datum.mutable_values()->add_double_values(value);
  return datum;
}

Datum Encode(const int32_t value) {
  Datum datum;
  datum.mutable_shape()->add_dim()->set_size(SCALAR_DIM_SIZE);
  datum.mutable_values()->add_int32_values(value);
  return datum;
}

Datum Encode(const int64_t value) {
  Datum datum;
  datum.mutable_shape()->add_dim()->set_size(SCALAR_DIM_SIZE);
  datum.mutable_values()->add_int64_values(value);
  return datum;
}

Datum Encode(const uint32_t value) {
  Datum datum;
  datum.mutable_shape()->add_dim()->set_size(SCALAR_DIM_SIZE);
  datum.mutable_values()->add_uint32_values(value);
  return datum;
}

Datum Encode(const uint64_t value) {
  Datum datum;
  datum.mutable_shape()->add_dim()->set_size(SCALAR_DIM_SIZE);
  datum.mutable_values()->add_uint64_values(value);
  return datum;
}

Datum Encode(const bool value) {
  Datum datum;
  datum.mutable_shape()->add_dim()->set_size(SCALAR_DIM_SIZE);
  datum.mutable_values()->add_bool_values(value);
  return datum;
}

Datum Encode(const std::string& value) {
  Datum datum;
  datum.mutable_shape()->add_dim()->set_size(SCALAR_DIM_SIZE);
  datum.mutable_values()->add_string_values(value);
  return datum;
}

Datum Encode(const char* value) {
  Datum datum;
  datum.mutable_shape()->add_dim()->set_size(SCALAR_DIM_SIZE);
  datum.mutable_values()->add_string_values(value);
  return datum;
}

Datum Encode(absl::Cord value) {
  Datum datum;
  datum.mutable_shape()->add_dim()->set_size(SCALAR_DIM_SIZE);
  datum.mutable_values()->add_bytes_values(std::string(value));
  return datum;
}

Datum Encode(mpz_class value) {
  Datum datum;
  datum.mutable_shape()->add_dim()->set_size(SCALAR_DIM_SIZE);
  datum.mutable_values()->add_bigint_values(FromMpzClass(value));
  return datum;
}

Datum Encode(const int8_t value) {
  Datum datum;
  datum.mutable_shape()->add_dim()->set_size(SCALAR_DIM_SIZE);
  std::string buffer(sizeof(int8_t), '\0');
  StoreInteger(value, buffer.data());
  datum.mutable_values()->set_int8_values(buffer);
  return datum;
}

Datum Encode(const int16_t value) {
  Datum datum;
  datum.mutable_shape()->add_dim()->set_size(SCALAR_DIM_SIZE);
  std::string buffer(sizeof(int16_t), '\0');
  StoreInteger(value, buffer.data());
  datum.mutable_values()->set_int16_values(buffer);
  return datum;
}

Datum Encode(const uint8_t value) {
  Datum datum;
  datum.mutable_shape()->add_dim()->set_size(SCALAR_DIM_SIZE);
  std::string buffer(sizeof(uint8_t), '\0');
  StoreInteger(value, buffer.data());
  datum.mutable_values()->set_uint8_values(buffer);
  return datum;
}

Datum Encode(const uint16_t value) {
  Datum datum;
  datum.mutable_shape()->add_dim()->set_size(SCALAR_DIM_SIZE);
  std::string buffer(sizeof(uint16_t), '\0');
  StoreInteger(value, buffer.data());
  datum.mutable_values()->set_uint16_values(buffer);
  return datum;
}

// xt::xarrays.

Datum Encode(const xt::xarray<float>& value, bool as_bytes) {
  Datum datum;
  auto* shape = datum.mutable_shape();
  for (const auto& dim : value.shape()) {
    shape->add_dim()->set_size(dim);
  }
  if (as_bytes) {
    riegeli::StringWriter writer(
        datum.mutable_values()->mutable_float_values_buffer());
    writer.SetWriteSizeHint(value.size() * sizeof(float));
    for (const float f : xt::ravel<xt::layout_type::row_major>(value)) {
      riegeli::WriteBigEndianFloat(f, writer);
    }
    writer.Close();
  } else {
    for (const auto x : xt::ravel<xt::layout_type::row_major>(value)) {
      datum.mutable_values()->add_float_values(x);
    }
  }
  return datum;
}

Datum Encode(const xt::xarray<double>& value) {
  Datum datum;
  auto* shape = datum.mutable_shape();
  for (const auto& dim : value.shape()) {
    shape->add_dim()->set_size(dim);
  }
  for (const auto x : xt::ravel<xt::layout_type::row_major>(value)) {
    datum.mutable_values()->add_double_values(x);
  }
  return datum;
}

Datum Encode(const xt::xarray<int32_t>& value) {
  Datum datum;
  auto* shape = datum.mutable_shape();
  for (const auto& dim : value.shape()) {
    shape->add_dim()->set_size(dim);
  }
  for (const auto x : xt::ravel<xt::layout_type::row_major>(value)) {
    datum.mutable_values()->add_int32_values(x);
  }
  return datum;
}

Datum Encode(const xt::xarray<int64_t>& value) {
  Datum datum;
  auto* shape = datum.mutable_shape();
  for (const auto& dim : value.shape()) {
    shape->add_dim()->set_size(dim);
  }
  for (const auto x : xt::ravel<xt::layout_type::row_major>(value)) {
    datum.mutable_values()->add_int64_values(x);
  }
  return datum;
}

Datum Encode(const xt::xarray<uint32_t>& value) {
  Datum datum;
  auto* shape = datum.mutable_shape();
  for (const auto& dim : value.shape()) {
    shape->add_dim()->set_size(dim);
  }
  for (const auto x : xt::ravel<xt::layout_type::row_major>(value)) {
    datum.mutable_values()->add_uint32_values(x);
  }
  return datum;
}

Datum Encode(const xt::xarray<uint64_t>& value) {
  Datum datum;
  auto* shape = datum.mutable_shape();
  for (const auto& dim : value.shape()) {
    shape->add_dim()->set_size(dim);
  }
  for (const auto x : xt::ravel<xt::layout_type::row_major>(value)) {
    datum.mutable_values()->add_uint64_values(x);
  }
  return datum;
}

Datum Encode(const xt::xarray<bool>& value) {
  Datum datum;
  auto* shape = datum.mutable_shape();
  for (const auto& dim : value.shape()) {
    shape->add_dim()->set_size(dim);
  }
  for (const auto x : xt::ravel<xt::layout_type::row_major>(value)) {
    datum.mutable_values()->add_bool_values(x);
  }
  return datum;
}

Datum Encode(const xt::xarray<std::string>& value) {
  Datum datum;
  auto* shape = datum.mutable_shape();
  for (const auto& dim : value.shape()) {
    shape->add_dim()->set_size(dim);
  }
  for (const auto& x : xt::ravel<xt::layout_type::row_major>(value)) {
    datum.mutable_values()->add_string_values(x);
  }
  return datum;
}

Datum Encode(const xt::xarray<absl::Cord>& value) {
  Datum datum;
  auto* shape = datum.mutable_shape();
  for (const auto& dim : value.shape()) {
    shape->add_dim()->set_size(dim);
  }
  for (const auto& x : xt::ravel<xt::layout_type::row_major>(value)) {
    datum.mutable_values()->add_bytes_values(std::string(x));
  }
  return datum;
}

Datum Encode(const xt::xarray<mpz_class>& value) {
  Datum datum;
  auto* shape = datum.mutable_shape();
  for (const auto& dim : value.shape()) {
    shape->add_dim()->set_size(dim);
  }
  for (const auto& x : xt::ravel<xt::layout_type::row_major>(value)) {
    datum.mutable_values()->add_bigint_values(FromMpzClass(x));
  }
  return datum;
}

// Implementation of a generic packer for integer types T.
// Values in `value` are encoded with a big-endian byte order and added one
// after the other in `output`.
template <typename T>
Datum::Shape EncodeBigEndian(const xt::xarray<T>& value, std::string* output) {
  Datum::Shape shape;
  int num_elements = 1;
  for (const auto& dim : value.shape()) {
    shape.add_dim()->set_size(dim);
    num_elements *= dim;
  }

  // Allocate space in the string for all elements.
  *output = std::string(sizeof(T) * num_elements, '\0');
  int i = 0;
  for (const auto x : xt::ravel<xt::layout_type::row_major>(value)) {
    StoreInteger<T>(x, &(*output)[i]);
    i += sizeof(T);
  }
  return shape;
}

Datum Encode(const xt::xarray<int8_t>& value) {
  Datum datum;
  *datum.mutable_shape() =
      EncodeBigEndian(value, datum.mutable_values()->mutable_int8_values());
  return datum;
}

Datum Encode(const xt::xarray<int16_t>& value) {
  Datum datum;
  *datum.mutable_shape() =
      EncodeBigEndian(value, datum.mutable_values()->mutable_int16_values());
  return datum;
}

Datum Encode(const xt::xarray<uint8_t>& value) {
  Datum datum;
  *datum.mutable_shape() =
      EncodeBigEndian(value, datum.mutable_values()->mutable_uint8_values());
  return datum;
}

Datum Encode(const xt::xarray<uint16_t>& value) {
  Datum datum;
  *datum.mutable_shape() =
      EncodeBigEndian(value, datum.mutable_values()->mutable_uint16_values());
  return datum;
}

////////////////////////////////////////////////////////////////////////////////
// Decode()
////////////////////////////////////////////////////////////////////////////////

std::optional<BasicType> Decode(const Datum& datum) {
  const std::vector<int> shape = ShapeVector(datum.shape());
  const bool is_scalar = IsScalar(shape);
  const auto& values = datum.values();

  std::optional<BasicType> float_buffer = DecodeValuesBigEndian<float>(
      values.float_values_buffer(), shape, is_scalar);
  if (float_buffer) return float_buffer;

  std::optional<BasicType> floats =
      DecodeValues(values.float_values(), shape, is_scalar);
  if (floats) return floats;

  std::optional<BasicType> doubles =
      DecodeValues(values.double_values(), shape, is_scalar);
  if (doubles) return doubles;

  std::optional<BasicType> int32s =
      DecodeValues(values.int32_values(), shape, is_scalar);
  if (int32s) return int32s;

  std::optional<BasicType> int64s =
      DecodeValues(values.int64_values(), shape, is_scalar);
  if (int64s) return int64s;

  std::optional<BasicType> uint32s =
      DecodeValues(values.uint32_values(), shape, is_scalar);
  if (uint32s) return uint32s;

  std::optional<BasicType> uint64s =
      DecodeValues(values.uint64_values(), shape, is_scalar);
  if (uint64s) return uint64s;

  std::optional<BasicType> bools =
      DecodeValues(values.bool_values(), shape, is_scalar);
  if (bools) return bools;

  std::optional<BasicType> strings =
      DecodeValues(values.string_values(), shape, is_scalar);
  if (strings) return strings;

  if (!datum.values().bytes_values().empty()) {
    if (is_scalar) {
      return absl::Cord(datum.values().bytes_values(0));
    } else {
      return FillXarrayValuesCord(datum.values().bytes_values(), shape);
    }
  }

  if (!datum.values().bigint_values().empty()) {
    if (is_scalar) {
      return ToMpzClass(datum.values().bigint_values(0));
    } else {
      return FillXarrayValuesMpzClass(datum.values().bigint_values(), shape);
    }
  }

  if (!datum.values().string_values().empty()) {
    if (is_scalar) {
      return datum.values().string_values(0);
    } else {
      // Unimplemented.
    }
  }

  std::optional<BasicType> int8s =
      DecodeValues<int8_t>(values.int8_values(), shape, is_scalar);
  if (int8s) return int8s;

  std::optional<BasicType> int16s =
      DecodeValues<int16_t>(values.int16_values(), shape, is_scalar);
  if (int16s) return int16s;

  std::optional<BasicType> uint8s =
      DecodeValues<uint8_t>(values.uint8_values(), shape, is_scalar);
  if (uint8s) return uint8s;

  std::optional<BasicType> uint16s =
      DecodeValues<uint16_t>(values.uint16_values(), shape, is_scalar);
  if (uint16s) return uint16s;

  return std::nullopt;
}

////////////////////////////////////////////////////////////////////////////////
// DataView
////////////////////////////////////////////////////////////////////////////////

DataView::DataView(const Data* data) : data_(data) { CHECK(data_ != nullptr); }

size_t DataView::size() const {
  switch (data_->value_case()) {
    case Data::kArray:
      return data_->array().values().size();
    case Data::kTuple:
      return data_->tuple().values().size();
    case Data::kDict:
      return data_->dict().values().size();
    default:
      return 0;
  }
}

DataView DataView::operator[](int index) const {
  const auto value_type = data_->value_case();
  CHECK(value_type == Data::kArray || value_type == Data::kTuple)
      << "Expected array or tuple, got: " << data_->value_case();

  CHECK(index >= 0 && index < static_cast<int>(size())) << absl::StrFormat(
      "Expected index between [0, %d), got: %d", size(), index);

  if (value_type == Data::kArray) {
    return DataView(&data_->array().values(index));
  }
  return DataView(&data_->tuple().values(index));
}

DataView DataView::operator[](const std::string& key) const {
  CHECK(data_->value_case() == Data::kDict)
      << "Expected dict, got: " << data_->value_case();

  const auto it = data_->dict().values().find(key);
  CHECK(it != data_->dict().values().end()) << "Non-existing key: " << key;
  return DataView(&it->second);
}

const google::protobuf::Map<std::string, Data>& DataView::items() const {
  return data_->dict().values();
}

// Iterator implementation.

DataView::const_iterator DataView::begin() const {
  // Dicts should use .items() and should always return nothing if using
  // .begin() and .end() directly.
  return DataView::const_iterator(
      this, data_->value_case() == Data::kDict ? size() : 0);
}

DataView::const_iterator DataView::end() const {
  return DataView::const_iterator(this, size());
}

bool DataView::const_iterator::operator==(
    const DataView::const_iterator& rhs) const {
  return view_ == rhs.view_ && index_ == rhs.index_;
}

bool DataView::const_iterator::operator!=(
    const DataView::const_iterator& rhs) const {
  return !(*this == rhs);
}

// Pre-inc version.
const DataView::const_iterator& DataView::const_iterator::operator++() {
  ++index_;
  return *this;
}

const Data& DataView::const_iterator::operator*() const {
  return *(*view_)[index_];
}

}  // namespace envlogger
