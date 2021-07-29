// Copyright 2021 DeepMind Technologies Limited..
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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
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
using ::testing::IsNull;
using ::testing::IsTrue;
using ::testing::NotNull;
using ::testing::Optional;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

// A functor whose operator() checks that its argument is the same as the one
// given at the construction. This is used as a visitor in absl::visit().
struct CheckVisitor {
  explicit CheckVisitor(const BasicType& value) : v(value) {}

  template <typename T>
  void operator()(const T& t) {
    EXPECT_THAT(t, Eq(absl::get<T>(v)));
  }

  void operator()(const float f) {
    EXPECT_THAT(f, FloatEq(absl::get<float>(v)));
  }
  void operator()(const double d) {
    EXPECT_THAT(d, DoubleEq(absl::get<double>(v)));
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
  const absl::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(absl::holds_alternative<float>(*decoded), IsTrue);
  EXPECT_THAT(absl::get<float>(*decoded), FloatEq(value));

  // We can also use a library for writing visitors:
  const auto visitor =
      MakeVisitor([value](const float f) { EXPECT_THAT(f, FloatEq(value)); },
                  [](const auto&) { /* catch all overload */ });
  absl::visit(visitor, *decoded);
}

TEST(XtensorCodecTest, Float32Identity) {
  const float value = 3.14f;
  const absl::optional<BasicType> decoded = Decode(Encode(value));
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
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
  const absl::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(absl::holds_alternative<double>(*decoded), IsTrue);
  EXPECT_THAT(absl::get<double>(*decoded), DoubleEq(value));
}

TEST(XtensorCodecTest, DoubleIdentity) {
  const double value = 3.14159265358979;
  const absl::optional<BasicType> decoded = Decode(Encode(value));
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
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
  const absl::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(absl::holds_alternative<int32_t>(*decoded), IsTrue);
  EXPECT_THAT(absl::get<int32_t>(*decoded), Eq(value));
}

TEST(XtensorCodecTest, Int32Identity) {
  const int32_t value = 321;
  const absl::optional<BasicType> decoded = Decode(Encode(value));
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
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
  const absl::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(absl::holds_alternative<int64_t>(*decoded), IsTrue);
  EXPECT_THAT(absl::get<int64_t>(*decoded), Eq(value));
}

TEST(XtensorCodecTest, Int64Identity) {
  const int64_t value = 123456789012;
  const absl::optional<BasicType> decoded = Decode(Encode(value));
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
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
  const absl::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(absl::holds_alternative<uint32_t>(*decoded), IsTrue);
  EXPECT_THAT(absl::get<uint32_t>(*decoded), Eq(value));
}

TEST(XtensorCodecTest, Uint32Identity) {
  // 2^32 - 1 = 4294967295
  const uint32_t value = 4294967295;
  const absl::optional<BasicType> decoded = Decode(Encode(value));
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
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
  const absl::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(absl::holds_alternative<uint64_t>(*decoded), IsTrue);
  EXPECT_THAT(absl::get<uint64_t>(*decoded), Eq(value));
}

TEST(XtensorCodecTest, Uint64Identity) {
  // 2^64 - 1 = 9223372036854775807.
  const uint64_t value = 9223372036854775807;
  const absl::optional<BasicType> decoded = Decode(Encode(value));
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
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
  const absl::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(true));
  absl::visit(CheckVisitor(true), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(absl::holds_alternative<bool>(*decoded), IsTrue);
  EXPECT_THAT(absl::get<bool>(*decoded), IsTrue);
}

TEST(XtensorCodecTest, BoolIdentity) {
  const absl::optional<BasicType> decoded = Decode(Encode(true));
  EXPECT_THAT(decoded, Optional(true));
  absl::visit(CheckVisitor(true), *decoded);
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
  const absl::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(absl::holds_alternative<std::string>(*decoded), IsTrue);
  EXPECT_THAT(absl::get<std::string>(*decoded), Eq(value));
}

TEST(XtensorCodecTest, StringIdentity) {
  const std::string value = "pi";
  const absl::optional<BasicType> decoded = Decode(Encode(value));
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
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
  const absl::optional<BasicType> decoded = Decode(datum);
  absl::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(absl::holds_alternative<absl::Cord>(*decoded), IsTrue);
  EXPECT_THAT(absl::get<absl::Cord>(*decoded), Eq(value));
}

TEST(XtensorCodecTest, BytesIdentity) {
  const absl::Cord value("pi");
  const absl::optional<BasicType> decoded = Decode(Encode(value));
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
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
  const absl::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(absl::holds_alternative<int8_t>(*decoded), IsTrue);
  EXPECT_THAT(absl::get<int8_t>(*decoded), Eq(value));
}

TEST(XtensorCodecTest, Int8Identity) {
  const int8_t value = -123;
  const absl::optional<BasicType> decoded = Decode(Encode(value));
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
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
  const absl::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(absl::holds_alternative<int16_t>(*decoded), IsTrue);
  EXPECT_THAT(absl::get<int16_t>(*decoded), Eq(value));
}

TEST(XtensorCodecTest, Int16Identity) {
  const int16_t value = -1234;
  const absl::optional<BasicType> decoded = Decode(Encode(value));
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
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
  const absl::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(absl::holds_alternative<uint8_t>(*decoded), IsTrue);
  EXPECT_THAT(absl::get<uint8_t>(*decoded), Eq(value));
}

TEST(XtensorCodecTest, Uint8Identity) {
  const uint8_t value = 255;
  const absl::optional<BasicType> decoded = Decode(Encode(value));
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
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
  const absl::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(absl::holds_alternative<uint16_t>(*decoded), IsTrue);
  EXPECT_THAT(absl::get<uint16_t>(*decoded), Eq(value));
}

TEST(XtensorCodecTest, Uint16Identity) {
  const uint16_t value = 12345;
  const absl::optional<BasicType> decoded = Decode(Encode(value));
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
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
  EXPECT_THAT(Encode(value), EqualsProto(expected_proto));
}

TEST(XtensorCodecTest, Float32DecodeXtarray) {
  const Datum datum = ParseTextProtoOrDie(R"pb(
    shape: { dim: { size: 1 } }
    values: { float_values: 1.23 }
  )pb");
  const xt::xarray<float> value{1.23f};
  const absl::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(absl::holds_alternative<xt::xarray<float>>(*decoded), IsTrue);
  EXPECT_THAT(absl::get<xt::xarray<float>>(*decoded), Eq(value));
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
  const absl::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(absl::holds_alternative<xt::xarray<double>>(*decoded), IsTrue);
  EXPECT_THAT(absl::get<xt::xarray<double>>(*decoded), Eq(value));
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
  const absl::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(absl::holds_alternative<xt::xarray<int32_t>>(*decoded), IsTrue);
  EXPECT_THAT(absl::get<xt::xarray<int32_t>>(*decoded), Eq(value));
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
  const absl::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(absl::holds_alternative<xt::xarray<int64_t>>(*decoded), IsTrue);
  EXPECT_THAT(absl::get<xt::xarray<int64_t>>(*decoded), Eq(value));
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
  const absl::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(absl::holds_alternative<xt::xarray<uint32_t>>(*decoded), IsTrue);
  EXPECT_THAT(absl::get<xt::xarray<uint32_t>>(*decoded), Eq(value));
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
  const absl::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(absl::holds_alternative<xt::xarray<uint64_t>>(*decoded), IsTrue);
  EXPECT_THAT(absl::get<xt::xarray<uint64_t>>(*decoded), Eq(value));
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
  const absl::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(absl::holds_alternative<xt::xarray<bool>>(*decoded), IsTrue);
  EXPECT_THAT(absl::get<xt::xarray<bool>>(*decoded), Eq(value));
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
  const absl::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(absl::holds_alternative<xt::xarray<std::string>>(*decoded),
              IsTrue);
  const xt::xarray<std::string>& actual =
      absl::get<xt::xarray<std::string>>(*decoded);
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
  const absl::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(absl::holds_alternative<xt::xarray<absl::Cord>>(*decoded),
              IsTrue);
  EXPECT_THAT(absl::get<xt::xarray<absl::Cord>>(*decoded), Eq(value));
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
  const absl::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(absl::holds_alternative<xt::xarray<int8_t>>(*decoded), IsTrue);
  EXPECT_THAT(absl::get<xt::xarray<int8_t>>(*decoded), Eq(value));
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
  const absl::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(absl::holds_alternative<xt::xarray<int16_t>>(*decoded), IsTrue);
  EXPECT_THAT(absl::get<xt::xarray<int16_t>>(*decoded), Eq(value));
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
  const absl::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(absl::holds_alternative<xt::xarray<uint8_t>>(*decoded), IsTrue);
  EXPECT_THAT(absl::get<xt::xarray<uint8_t>>(*decoded), Eq(value));
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
  const absl::optional<BasicType> decoded = Decode(datum);
  EXPECT_THAT(decoded, Optional(value));
  absl::visit(CheckVisitor(value), *decoded);
  // Directly checking for the value and getting it should also be supported.
  EXPECT_THAT(absl::holds_alternative<xt::xarray<uint16_t>>(*decoded), IsTrue);
  EXPECT_THAT(absl::get<xt::xarray<uint16_t>>(*decoded), Eq(value));
}

////////////////////////////////////////////////////////////////////////////////
// DataView tests
////////////////////////////////////////////////////////////////////////////////

TEST(DataViewTest, Nullptr) {
  const DataView view(/*data=*/nullptr);
  EXPECT_THAT(view.Type(), Eq(Data::VALUE_NOT_SET));
  EXPECT_THAT(view, IsEmpty());
  EXPECT_THAT(view[0], IsNull());
}

TEST(DataViewTest, Datum) {
  const Data data = ParseTextProtoOrDie(R"pb(
    datum: {
      shape: { dim: { size: 2 } }
      values: { int32_values: [ -1, 1 ] }
    }
  )pb");
  const DataView view(&data);
  EXPECT_THAT(view.Type(), Eq(Data::kDatum));
  EXPECT_THAT(view, IsEmpty());
  EXPECT_THAT(view[0], IsNull());
  EXPECT_THAT(view.data(), NotNull());
  EXPECT_THAT(*view.data(), EqualsProto(data));
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
  EXPECT_THAT(view.Type(), Eq(Data::kArray));
  EXPECT_THAT(view, SizeIs(1));
  const auto result = view[0];
  EXPECT_THAT(result, NotNull());
  EXPECT_THAT(*result, EqualsProto(data.array().values(0)));
  EXPECT_THAT(view[-1], IsNull());
  EXPECT_THAT(view[1234561], IsNull());
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
  EXPECT_THAT(view.Type(), Eq(Data::kTuple));
  EXPECT_THAT(view, SizeIs(1));
  const auto result = view[0];
  EXPECT_THAT(result, NotNull());
  EXPECT_THAT(*result, EqualsProto(data.tuple().values(0)));
  EXPECT_THAT(view[-1], IsNull());
  EXPECT_THAT(view[1234561], IsNull());
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
  EXPECT_THAT(view.Type(), Eq(Data::kDict));
  EXPECT_THAT(view, SizeIs(1));
  const auto result = view["hello"];
  EXPECT_THAT(result, NotNull());
  EXPECT_THAT(*result, EqualsProto(data.dict().values().at("hello")));
  EXPECT_THAT(view["doesnotexist"], IsNull());
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
  EXPECT_THAT(view.Type(), Eq(Data::kArray));
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
  EXPECT_THAT(view.Type(), Eq(Data::kTuple));
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
  EXPECT_THAT(view.Type(), Eq(Data::kDict));
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

}  // namespace
}  // namespace envlogger
