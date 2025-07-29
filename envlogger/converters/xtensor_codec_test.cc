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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <gmpxx.h>
#include "envlogger/converters/make_visitor.h"
#include "envlogger/platform/parse_text_proto.h"
#include "envlogger/platform/proto_testutil.h"
#include "envlogger/proto/storage.pb.h"

namespace envlogger {
namespace {

using ::testing::DoubleEq;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::IsEmpty;
using ::testing::IsTrue;
using ::testing::Optional;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

// A functor whose operator() checks that its argument is the same as the one
// given at the construction. This is used as a visitor in std::visit().
struct CheckVisitor {
  explicit CheckVisitor(const BasicType& value) : v(value) {}

  template <typename T>
  void operator()(const T& t) {
    EXPECT_THAT(t, Eq(std::get<T>(v)));
  }

  void operator()(const float f) {
    EXPECT_THAT(f, FloatEq(std::get<float>(v)));
  }
  void operator()(const double d) {
    EXPECT_THAT(d, DoubleEq(std::get<double>(v)));
  }

 private:
  const BasicType v;
};

////////////////////////////////////////////////////////////////////////////////
// Scalars
////////////////////////////////////////////////////////////////////////////////

// float.

TEST(XtensorCodecTest, EncodeScalarFloat32) {
  EXPECT_THAT(Encode(1.23f), EqualsProto(R"pb(
                shape: { dim: { size: -438 } }
                values: { float_values: 1.23 }
              )pb"));
}

TEST(XtensorCodecTest, DecodeScalarFloat32) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { float_values: 1.23 }
  )pb");
  const float value = 1.23f;
  const std::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<float>(*decoded), IsTrue);
  EXPECT_THAT(std::get<float>(*decoded), FloatEq(value));

  // We can also use a library for writing visitors:
  const auto visitor =
      MakeVisitor([value](const float f) { EXPECT_THAT(f, FloatEq(value)); },
                  [](const auto&) { /* catch all overload */ });
  std::visit(visitor, *decoded);
}

TEST(XtensorCodecTest, Float32Identity) {
  const float value = 3.14f;
  const std::optional<BasicType> decoded = Decode(Encode(value));
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
}

// double.

TEST(XtensorCodecTest, EncodeScalarDouble) {
  EXPECT_THAT(Encode(3.14159265358979), EqualsProto(R"pb(
                shape: { dim: { size: -438 } }
                values: { double_values: 3.14159265358979 }
              )pb"));
}

TEST(XtensorCodecTest, DecodeScalarDouble) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { double_values: 3.14159265358979 }
  )pb");
  const double value = 3.14159265358979;
  const std::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<double>(*decoded), IsTrue);
  EXPECT_THAT(std::get<double>(*decoded), DoubleEq(value));
}

TEST(XtensorCodecTest, DoubleIdentity) {
  const double value = 3.14159265358979;
  const std::optional<BasicType> decoded = Decode(Encode(value));
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
}

// int32.

TEST(XtensorCodecTest, EncodeScalarInt32) {
  const Datum expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { int32_values: 3 }
  )pb");
  EXPECT_THAT(Encode(3), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, DecodeScalarInt32) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { int32_values: -32 }
  )pb");
  const int32_t value = -32;
  const std::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<int32_t>(*decoded), IsTrue);
  EXPECT_THAT(std::get<int32_t>(*decoded), Eq(value));
}

TEST(XtensorCodecTest, Int32Identity) {
  const int32_t value = 321;
  const std::optional<BasicType> decoded = Decode(Encode(value));
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
}

// int64.

TEST(XtensorCodecTest, EncodeScalarInt64) {
  const Datum expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { int64_values: 3 }
  )pb");
  const int64_t value = 3;
  EXPECT_THAT(Encode(value), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, DecodeScalarInt64) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { int64_values: -64 }
  )pb");
  const int64_t value = -64;
  const std::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<int64_t>(*decoded), IsTrue);
  EXPECT_THAT(std::get<int64_t>(*decoded), Eq(value));
}

TEST(XtensorCodecTest, Int64Identity) {
  const int64_t value = 123456789012;
  const std::optional<BasicType> decoded = Decode(Encode(value));
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
}

// uint32.

TEST(XtensorCodecTest, EncodeScalarUint32) {
  const Datum expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { uint32_values: 12345 }
  )pb");
  const uint32_t value = 12345;
  EXPECT_THAT(Encode(value), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, DecodeScalarUint32) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { uint32_values: 32 }
  )pb");
  const uint32_t value = 32;
  const std::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<uint32_t>(*decoded), IsTrue);
  EXPECT_THAT(std::get<uint32_t>(*decoded), Eq(value));
}

TEST(XtensorCodecTest, Uint32Identity) {
  // 2^32 - 1 = 4294967295
  const uint32_t value = 4294967295;
  const std::optional<BasicType> decoded = Decode(Encode(value));
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
}

// uint64.

TEST(XtensorCodecTest, EncodeScalarUint64) {
  const Datum expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { uint64_values: 12345 }
  )pb");
  const uint64_t value = 12345;
  EXPECT_THAT(Encode(value), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, DecodeScalarUint64) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { uint64_values: 64 }
  )pb");
  const uint64_t value = 64;
  const std::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<uint64_t>(*decoded), IsTrue);
  EXPECT_THAT(std::get<uint64_t>(*decoded), Eq(value));
}

TEST(XtensorCodecTest, Uint64Identity) {
  // 2^64 - 1 = 9223372036854775807.
  const uint64_t value = 9223372036854775807;
  const std::optional<BasicType> decoded = Decode(Encode(value));
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
}

// bool.

TEST(XtensorCodecTest, EncodeScalarBool) {
  const Datum expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { bool_values: true }
  )pb");
  EXPECT_THAT(Encode(true), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, DecodeScalarBool) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { bool_values: true }
  )pb");
  const std::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(true));
  std::visit(CheckVisitor(true), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<bool>(*decoded), IsTrue);
  EXPECT_THAT(std::get<bool>(*decoded), IsTrue);
}

TEST(XtensorCodecTest, BoolIdentity) {
  const std::optional<BasicType> decoded = Decode(Encode(true));
  EXPECT_THAT(decoded, Optional(true));
  std::visit(CheckVisitor(true), *decoded);
}

// string.

TEST(XtensorCodecTest, EncodeScalarString) {
  const Datum expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { string_values: 'pi' }
  )pb");
  const std::string data = "pi";
  EXPECT_THAT(Encode(data), EqualsProto(expected_proto));
  // Char arrays should also be supported.
  EXPECT_THAT(Encode("pi"), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, DecodeScalarString) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { string_values: 'pi' }
  )pb");
  const std::string value = "pi";
  const std::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<std::string>(*decoded), IsTrue);
  EXPECT_THAT(std::get<std::string>(*decoded), Eq(value));
}

TEST(XtensorCodecTest, StringIdentity) {
  const std::string value = "pi";
  const std::optional<BasicType> decoded = Decode(Encode(value));
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
}

// bytes.

TEST(XtensorCodecTest, BytesEncodeScalar) {
  const Datum expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { bytes_values: 'pi' }
  )pb");
  const absl::Cord value("pi");
  EXPECT_THAT(Encode(value), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, BytesDecodeScalar) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { bytes_values: 'pi' }
  )pb");
  const absl::Cord value("pi");
  const std::optional<BasicType> decoded = Decode(datum);
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<absl::Cord>(*decoded), IsTrue);
  EXPECT_THAT(std::get<absl::Cord>(*decoded), Eq(value));
}

TEST(XtensorCodecTest, BytesIdentity) {
  const absl::Cord value("pi");
  const std::optional<BasicType> decoded = Decode(Encode(value));
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
}

// big ints.

TEST(XtensorCodecTest, BigIntEncodeScalar) {
  const Datum positive_expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { bigint_values: '\x03' }
  )pb");
  EXPECT_THAT(Encode(mpz_class(3)), EqualsProto(positive_expected_proto));
  const Datum negative_expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { bigint_values: '\xfd' }
  )pb");
  EXPECT_THAT(Encode(mpz_class(-3)), EqualsProto(negative_expected_proto));
}

TEST(XtensorCodecTest, BigIntDecodeScalar) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { bigint_values: '\x01\x8e\xe9\x0f\xf6\xc3s\xe0\xeeN?\n\xd2' }
  )pb");
  const mpz_class value("123456789012345678901234567890", /*base=*/10);
  const std::optional<BasicType> decoded = Decode(datum);
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<mpz_class>(*decoded), IsTrue);
  EXPECT_THAT(std::get<mpz_class>(*decoded), Eq(value));
}

TEST(XtensorCodecTest, BigIntIdentity) {
  const mpz_class value("-98765432109876543210", /*base=*/10);
  const std::optional<BasicType> decoded = Decode(Encode(value));
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
}

// int8.

TEST(XtensorCodecTest, EncodeScalarInt8) {
  const Datum expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { int8_values: '\x03' }
  )pb");
  const int8_t value = 3;
  EXPECT_THAT(Encode(value), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, DecodeScalarInt8) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { int8_values: '\xfd' }
  )pb");
  const int8_t value = -3;
  const std::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<int8_t>(*decoded), IsTrue);
  EXPECT_THAT(std::get<int8_t>(*decoded), Eq(value));
}

TEST(XtensorCodecTest, Int8Identity) {
  const int8_t value = -123;
  const std::optional<BasicType> decoded = Decode(Encode(value));
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
}

// int16.

TEST(XtensorCodecTest, EncodeScalarInt16) {
  const Datum expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { int16_values: '\xfe\xd4' }
  )pb");
  const int16_t value = -300;
  EXPECT_THAT(Encode(value), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, DecodeScalarInt16) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { int16_values: '\x07\xd0' }
  )pb");
  const int16_t value = 2000;
  const std::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<int16_t>(*decoded), IsTrue);
  EXPECT_THAT(std::get<int16_t>(*decoded), Eq(value));
}

TEST(XtensorCodecTest, Int16Identity) {
  const int16_t value = -1234;
  const std::optional<BasicType> decoded = Decode(Encode(value));
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
}

// uint8.

TEST(XtensorCodecTest, Uint8EncodeScalar) {
  const Datum expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { uint8_values: '\xfb' }
  )pb");
  const uint8_t value = 251;
  EXPECT_THAT(Encode(value), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, Uint8DecodeScalar) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { uint8_values: '\xed' }
  )pb");
  const uint8_t value = 237;
  const std::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<uint8_t>(*decoded), IsTrue);
  EXPECT_THAT(std::get<uint8_t>(*decoded), Eq(value));
}

TEST(XtensorCodecTest, Uint8Identity) {
  const uint8_t value = 255;
  const std::optional<BasicType> decoded = Decode(Encode(value));
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
}

// uint16.

TEST(XtensorCodecTest, Uint16EncodeScalar) {
  const Datum expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { uint16_values: '\x03\xe8' }
  )pb");
  const uint16_t value = 1000;
  EXPECT_THAT(Encode(value), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, Uint16DecodeScalar) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: -438 } }
    values: { uint16_values: '\x0b\xb8' }
  )pb");
  const uint16_t value = 3000;
  const std::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<uint16_t>(*decoded), IsTrue);
  EXPECT_THAT(std::get<uint16_t>(*decoded), Eq(value));
}

TEST(XtensorCodecTest, Uint16Identity) {
  const uint16_t value = 12345;
  const std::optional<BasicType> decoded = Decode(Encode(value));
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
}

///////////////////////////////////////////////////////////////////////////////
// xt::xarrays
///////////////////////////////////////////////////////////////////////////////

// float.

TEST(XtensorCodecTest, Float32EncodeXtarray) {
  const Datum expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 1 } }
    values: { float_values: 1.23 }
  )pb");
  const xt::xarray<float> value{1.23f};
  EXPECT_THAT(Encode(value, /*as_bytes=*/false), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, Float32EncodeXtarrayAsBytes) {
  const Datum expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 1 } }
    values: { float_values_buffer: "?\235p\244" }
  )pb");
  const xt::xarray<float> value{1.23f};
  EXPECT_THAT(Encode(value, /*as_bytes=*/true), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, Float32DecodeXtarray) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 1 } }
    values: { float_values: 1.23 }
  )pb");
  const xt::xarray<float> value{1.23f};
  const std::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<xt::xarray<float>>(*decoded), IsTrue);
  EXPECT_THAT(std::get<xt::xarray<float>>(*decoded), Eq(value));
}

TEST(XtensorCodecTest, Float32XtarrayIdentity) {
  const xt::xarray<float> value{3.14f};
  const std::optional<BasicType> decoded =
      Decode(Encode(value, /*as_bytes=*/false));
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
}

TEST(XtensorCodecTest, Float32XtarrayIdentityAsBytes) {
  const xt::xarray<float> value{3.14f};
  const std::optional<BasicType> decoded = Decode(Encode(value));
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
}

// double.

TEST(XtensorCodecTest, DoubleXtarray) {
  const Datum expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 1 } }
    values: { double_values: 3.14159265358979 }
  )pb");
  const xt::xarray<double> value{3.14159265358979};
  EXPECT_THAT(Encode(value), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, DoubleDecodeXtarray) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 1 } }
    values: { double_values: 3.14159265358979 }
  )pb");
  const xt::xarray<double> value{3.14159265358979};
  const std::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<xt::xarray<double>>(*decoded), IsTrue);
  EXPECT_THAT(std::get<xt::xarray<double>>(*decoded), Eq(value));
}

// int32.

TEST(XtensorCodecTest, Int32EncodeXtarray) {
  const Datum expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 1 } }
    values: { int32_values: 123 }
  )pb");
  const xt::xarray<int32_t> value{123};
  EXPECT_THAT(Encode(value), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, Int32DecodeXtarray) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 1 } }
    values: { int32_values: -123 }
  )pb");
  const xt::xarray<int32_t> value{-123};
  const std::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<xt::xarray<int32_t>>(*decoded), IsTrue);
  EXPECT_THAT(std::get<xt::xarray<int32_t>>(*decoded), Eq(value));
}

// int64.

TEST(XtensorCodecTest, Int64EncodeXtarray) {
  const Datum expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 1 } }
    values: { int64_values: 123456789123 }
  )pb");
  const xt::xarray<int64_t> value{123456789123};
  EXPECT_THAT(Encode(value), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, Int64DecodeXtarray) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 1 } }
    values: { int64_values: -123456789123 }
  )pb");
  const xt::xarray<int64_t> value{-123456789123};
  const std::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<xt::xarray<int64_t>>(*decoded), IsTrue);
  EXPECT_THAT(std::get<xt::xarray<int64_t>>(*decoded), Eq(value));
}

// uint32.

TEST(XtensorCodecTest, Uint32EncodeXtarray) {
  // 2^32 - 1 = 4294967295
  const Datum expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 1 } }
    values: { uint32_values: 4294967295 }
  )pb");
  const xt::xarray<uint32_t> value{4294967295};
  EXPECT_THAT(Encode(value), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, Uint32DecodeXtarray) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 1 } }
    values: { uint32_values: 123456 }
  )pb");
  const xt::xarray<uint32_t> value{123456};
  const std::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<xt::xarray<uint32_t>>(*decoded), IsTrue);
  EXPECT_THAT(std::get<xt::xarray<uint32_t>>(*decoded), Eq(value));
}

// uint64.

TEST(XtensorCodecTest, Uint64EncodeXtarray) {
  // 2^64 - 1 = 9223372036854775807.
  const Datum expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 1 } }
    values: { uint64_values: 9223372036854775807 }
  )pb");
  const xt::xarray<uint64_t> value{9223372036854775807};
  EXPECT_THAT(Encode(value), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, Uint64DecodeXtarray) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 1 } }
    values: { uint64_values: 1234567890123 }
  )pb");
  const xt::xarray<uint64_t> value{1234567890123};
  const std::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<xt::xarray<uint64_t>>(*decoded), IsTrue);
  EXPECT_THAT(std::get<xt::xarray<uint64_t>>(*decoded), Eq(value));
}

// bool.

TEST(XtensorCodecTest, BoolEncodeXtarray) {
  const Datum expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 1 } }
    values: { bool_values: true }
  )pb");
  const xt::xarray<bool> value{true};
  EXPECT_THAT(Encode(value), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, BoolDecodeXtarray) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 1 } }
    values: { bool_values: true }
  )pb");
  const xt::xarray<bool> value{true};
  const std::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<xt::xarray<bool>>(*decoded), IsTrue);
  EXPECT_THAT(std::get<xt::xarray<bool>>(*decoded), Eq(value));
}

// std::string.

TEST(XtensorCodecTest, StringEncodeXtarray) {
  const Datum expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 1 } }
    values: { string_values: "awesome value" }
  )pb");
  const xt::xarray<std::string> value{"awesome value"};
  EXPECT_THAT(Encode(value), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, StringDecodeXtarray) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 1 } }
    values: { string_values: "nice" }
  )pb");
  const xt::xarray<std::string> value{"nice"};
  const std::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<xt::xarray<std::string>>(*decoded),
              IsTrue);
  const xt::xarray<std::string>& actual =
      std::get<xt::xarray<std::string>>(*decoded);
  EXPECT_THAT(actual, Eq(value));
  EXPECT_THAT(actual.shape(), ElementsAre(1));
}

// bytes.

TEST(XtensorCodecTest, BytesEncodeXtarray) {
  const Datum expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 1 } }
    values: { bytes_values: 'a1b2c3d4e5f6' }
  )pb");
  const xt::xarray<absl::Cord> value{absl::Cord("a1b2c3d4e5f6")};
  EXPECT_THAT(Encode(value), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, BytesDecodeXtarray) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 1 } }
    values: { bytes_values: '6f5e4d3c2b1a' }
  )pb");
  const xt::xarray<absl::Cord> value{absl::Cord("6f5e4d3c2b1a")};
  const std::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<xt::xarray<absl::Cord>>(*decoded), IsTrue);
  EXPECT_THAT(std::get<xt::xarray<absl::Cord>>(*decoded), Eq(value));
}

// big int.

TEST(XtensorCodecTest, BigIntEncodeXtarray) {
  const Datum expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 1 } }
    values: { bigint_values: '\000\253T\251\214\353\037\n\322' }
  )pb");
  const xt::xarray<mpz_class> value{
      mpz_class("12345678901234567890", /*base=*/10)};
  EXPECT_THAT(Encode(value), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, BigIntDecodeXtarray) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 1 } }
    values: { bigint_values: '\000\253T\251\214\353\037\n\322' }
  )pb");
  const xt::xarray<mpz_class> value{
      mpz_class("12345678901234567890", /*base=*/10)};
  const std::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<xt::xarray<mpz_class>>(*decoded), IsTrue);
  EXPECT_THAT(std::get<xt::xarray<mpz_class>>(*decoded), Eq(value));
}

// int8.

TEST(XtensorCodecTest, Int8EncodeXtarray) {
  const Datum expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 1 } }
    values: { int8_values: '\x85' }
  )pb");
  const xt::xarray<int8_t> value{-123};
  EXPECT_THAT(Encode(value), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, Int8DecodeXtarray) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 2 } }
    values: { int8_values: '\x85\x84' }
  )pb");
  const xt::xarray<int8_t> value{-123, -124};
  const std::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<xt::xarray<int8_t>>(*decoded), IsTrue);
  EXPECT_THAT(std::get<xt::xarray<int8_t>>(*decoded), Eq(value));
}

// int16.

TEST(XtensorCodecTest, Int16EncodeXtarray) {
  const Datum expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 1 } }
    values: { int16_values: '\xfe\xa7' }
  )pb");
  const xt::xarray<int16_t> value{-345};
  EXPECT_THAT(Encode(value), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, Int16DecodeXtarray) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 2 } }
    values: { int16_values: '\xfe\xa7\xfe\xa6' }
  )pb");
  const xt::xarray<int16_t> value{-345, -346};
  const std::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<xt::xarray<int16_t>>(*decoded), IsTrue);
  EXPECT_THAT(std::get<xt::xarray<int16_t>>(*decoded), Eq(value));
}

// uint8.

TEST(XtensorCodecTest, Uint8EncodeXtarray) {
  const Datum expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 1 } }
    values: { uint8_values: '\x7b' }
  )pb");
  const xt::xarray<uint8_t> value{123};
  EXPECT_THAT(Encode(value), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, Uint8DecodeXtarray) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 2 } }
    values: { uint8_values: '\x7b\x7a' }
  )pb");
  const xt::xarray<uint8_t> value{123, 122};
  const std::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<xt::xarray<uint8_t>>(*decoded), IsTrue);
  EXPECT_THAT(std::get<xt::xarray<uint8_t>>(*decoded), Eq(value));
}

// uint16.

TEST(XtensorCodecTest, Uint16EncodeXtarray) {
  const Datum expected_proto = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 1 } }
    values: { uint16_values: '\x01Y' }
  )pb");
  const xt::xarray<uint16_t> value{345};
  EXPECT_THAT(Encode(value), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, Uint16DecodeXtarray) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 2 } }
    values: { uint16_values: '\x01Y\x01X' }
  )pb");
  const xt::xarray<uint16_t> value{345, 344};
  const std::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  std::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(std::holds_alternative<xt::xarray<uint16_t>>(*decoded), IsTrue);
  EXPECT_THAT(std::get<xt::xarray<uint16_t>>(*decoded), Eq(value));
}

////////////////////////////////////////////////////////////////////////////////
// DataView tests
////////////////////////////////////////////////////////////////////////////////

TEST(DataViewDeathTest, Nullptr) { EXPECT_DEATH(DataView(nullptr), ".*"); }

TEST(DataViewDeathTest, NullData) {
  const Data* data = nullptr;
  EXPECT_DEATH(DataView{data}, ".*");
}

TEST(DataViewDeathTest, UnsubscriptableDatum) {
  const Data data = ParseTextProtoOrDie(R"pb(
    datum: {
      shape: { dim: { size: 2 } }
      values: { int32_values: [ -1, 1 ] }
    }
  )pb");
  const DataView view(&data);
  EXPECT_DEATH(view[0], "Expected array or tuple, got: 1");
}

TEST(DataViewTest, OutOfBound) {
  const Data data = ParseTextProtoOrDie(R"pb(
    array: {
      values: {
        datum: {
          shape: { dim: { size: 4 } }
          values: { int32_values: [ 1, 4, 9, 16 ] }
        }
      }
    }
  )pb");
  const DataView view(&data);
  EXPECT_DEATH(view[-1],
               R"regex(Expected index between \[0, 1\), got: -1)regex");
  EXPECT_DEATH(view[2], R"regex(Expected index between \[0, 1\), got: 2)regex");
}

TEST(DataViewTest, NonExistingKey) {
  const Data data = ParseTextProtoOrDie(R"pb(
    dict: {
      values: {
        key: "hello"
        value: {
          datum: {
            shape: { dim: { size: 4 } }
            values: { int32_values: [ 1, 4, 9, 16 ] }
          }
        }
      }
    }
  )pb");
  const DataView view(&data);
  ASSERT_DEATH(view["foobar"], "Non-existing key: foobar");
}

TEST(DataViewTest, Datum) {
  const Data data = ParseTextProtoOrDie(R"pb(
    datum: {
      shape: { dim: { size: 2 } }
      values: { int32_values: [ -1, 1 ] }
    }
  )pb");
  const DataView view(&data);
  EXPECT_THAT(view, IsEmpty());
  EXPECT_THAT(*view, EqualsProto(data));
}

TEST(DataViewTest, SingleElementArray) {
  const Data data = ParseTextProtoOrDie(R"pb(
    array: {
      values: {
        datum: {
          shape: { dim: { size: 4 } }
          values: { int32_values: [ 1, 4, 9, 16 ] }
        }
      }
    }
  )pb");
  const DataView view(&data);
  EXPECT_THAT(*view[0], EqualsProto(data.array().values(0)));
}

TEST(DataViewTest, SingleElementTuple) {
  const Data data = ParseTextProtoOrDie(R"pb(
    tuple: {
      values: {
        datum: {
          shape: { dim: { size: 4 } }
          values: { int32_values: [ 1, 4, 9, 16 ] }
        }
      }
    }
  )pb");
  const DataView view(&data);
  EXPECT_THAT(view, SizeIs(1));
  EXPECT_THAT(*view[0], EqualsProto(data.tuple().values(0)));
}

TEST(DataViewTest, SingleElementDict) {
  const Data data = ParseTextProtoOrDie(R"pb(
    dict: {
      values: {
        key: "hello"
        value: {
          datum: {
            shape: { dim: { size: 4 } }
            values: { int32_values: [ 1, 4, 9, 16 ] }
          }
        }
      }
    }
  )pb");
  const DataView view(&data);
  EXPECT_THAT(view, SizeIs(1));
  EXPECT_THAT(*view["hello"], EqualsProto(data.dict().values().at("hello")));
}

TEST(DataViewTest, MultipleElementsArray) {
  const Data data = ParseTextProtoOrDie(R"pb(
    array: {
      values: {
        datum: {
          shape: { dim: { size: 4 } }
          values: { int32_values: [ 1, 4, 9, 16 ] }
        }
      }
      values: {
        datum: {
          shape: { dim: { size: 3 } }
          values: { int32_values: [ 1, 8, 27 ] }
        }
      }
      values: {
        datum: {
          shape: { dim: { size: 5 } }
          values: { int32_values: [ 1, 16, 81, 256, 625 ] }
        }
      }
    }
  )pb");
  const DataView view(&data);
  EXPECT_THAT(view, SizeIs(3));
  // ElementsAre() should iterate over the elements with a const iterator.
  EXPECT_THAT(view, ElementsAre(EqualsProto(data.array().values(0)),
                                EqualsProto(data.array().values(1)),
                                EqualsProto(data.array().values(2))));
  // We can also use .begin() and .end() explicitly.
  const std::vector<Data> v(view.begin(), view.end());
  EXPECT_THAT(v, SizeIs(3));
}

TEST(DataViewTest, MultipleElementsTuple) {
  const Data data = ParseTextProtoOrDie(R"pb(
    tuple: {
      values: {
        datum: {
          shape: { dim: { size: 4 } }
          values: { int32_values: [ 1, 4, 9, 16 ] }
        }
      }
      values: {
        datum: {
          shape: { dim: { size: 3 } }
          values: { int64_values: [ 1, 8, 27 ] }
        }
      }
      values: {
        datum: {
          shape: { dim: { size: 5 } }
          values: { uint32_values: [ 1, 16, 81, 256, 625 ] }
        }
      }
    }
  )pb");
  const DataView view(&data);
  EXPECT_THAT(view, SizeIs(3));
  // ElementsAre() should iterate over the elements with a const iterator.
  EXPECT_THAT(view, ElementsAre(EqualsProto(data.tuple().values(0)),
                                EqualsProto(data.tuple().values(1)),
                                EqualsProto(data.tuple().values(2))));
  // We can also use .begin() and .end() explicitly.
  const std::vector<Data> v(view.begin(), view.end());
  EXPECT_THAT(v, SizeIs(3));
}

TEST(DataViewTest, MultipleElementsDict) {
  const Data data = ParseTextProtoOrDie(R"pb(
    dict: {
      values: {
        key: "hello"
        value: {
          datum: {
            shape: { dim: { size: 4 } }
            values: { int32_values: [ 1, 4, 9, 16 ] }
          }
        }
      }
      values: {
        key: "world"
        value: {
          datum: {
            shape: { dim: { size: 3 } }
            values: { int64_values: [ 1, 8, 27 ] }
          }
        }
      }
      values: {
        key: "hey"
        value: {
          datum: {
            shape: { dim: { size: 5 } }
            values: { uint32_values: [ 1, 16, 81, 256, 625 ] }
          }
        }
      }
    }
  )pb");
  const DataView view(&data);
  EXPECT_THAT(view, SizeIs(3));
  // Check the unordered contents of this dict with .items().
  EXPECT_THAT(view.items(),
              UnorderedElementsAre(
                  Pair("hello", EqualsProto(data.dict().values().at("hello"))),
                  Pair("world", EqualsProto(data.dict().values().at("world"))),
                  Pair("hey", EqualsProto(data.dict().values().at("hey")))));
  // .begin() and .end() on the actual view should return nothing.
  const std::vector<Data> v(view.begin(), view.end());
  EXPECT_THAT(v, IsEmpty());
}

TEST(DataViewTest, ChainingAccessor) {
  // Equivalent to Python object:
  //
  //   data = [{
  //       'a': np.array([1, 2, 3]),
  //       'b': [4, 5, 6],
  //       'd': {
  //           'd1': {
  //               'd2': "hello"
  //           }
  //       },
  //       'e': {
  //           'e1': [{
  //               'e2': 'world'
  //           }]
  //       }
  //   }]
  const Data data = ParseTextProtoOrDie(R"pb(
    array {
      values {
        dict {
          values {
            key: "a"
            value {
              datum {
                shape { dim { size: 3 } }
                values { int64_values: 1 int64_values: 2 int64_values: 3 }
              }
            }
          }
          values {
            key: "b"
            value {
              array {
                values {
                  datum {
                    shape { dim { size: -438 } }
                    values { bigint_values: "\004" }
                  }
                }
                values {
                  datum {
                    shape { dim { size: -438 } }
                    values { bigint_values: "\005" }
                  }
                }
                values {
                  datum {
                    shape { dim { size: -438 } }
                    values { bigint_values: "\006" }
                  }
                }
              }
            }
          }
          values {
            key: "d"
            value {
              dict {
                values {
                  key: "d1"
                  value {
                    dict {
                      values {
                        key: "d2"
                        value {
                          datum {
                            shape { dim { size: -438 } }
                            values { string_values: "hello" }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          values {
            key: "e"
            value {
              dict {
                values {
                  key: "e1"
                  value {
                    array {
                      values {
                        dict {
                          values {
                            key: "e2"
                            value {
                              datum {
                                shape { dim { size: -438 } }
                                values { string_values: "world" }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  )pb");
  const DataView view(&data);
  EXPECT_THAT(*view[0]["a"],
              EqualsProto(data.array().values(0).dict().values().at("a")));
  EXPECT_THAT(
      *view[0]["b"][0],
      EqualsProto(
          data.array().values(0).dict().values().at("b").array().values(0)));
  EXPECT_THAT(*view[0]["e"]["e1"][0]["e2"], EqualsProto(data.array()
                                                            .values(0)
                                                            .dict()
                                                            .values()
                                                            .at("e")
                                                            .dict()
                                                            .values()
                                                            .at("e1")
                                                            .array()
                                                            .values(0)
                                                            .dict()
                                                            .values()
                                                            .at("e2")));
}

}  // namespace
}  // namespace envlogger
