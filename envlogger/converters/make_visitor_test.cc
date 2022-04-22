// Copyright 2022 DeepMind Technologies Limited..
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

#include "envlogger/converters/make_visitor.h"

#include <cstddef>
#include <memory>
#include <string>
#include <variant>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace envlogger {
namespace {

using ::testing::Eq;
using ::testing::IsFalse;
using ::testing::IsTrue;

TEST(MakeVisitorTest, SanityCheck) {
  bool executed_char_visitor = false;
  bool executed_double_visitor = false;

  const auto visitor = MakeVisitor(
      [&](char) {
        executed_char_visitor = true;
        executed_double_visitor = false;
      },
      [&](double) {
        executed_char_visitor = false;
        executed_double_visitor = true;
      },
      [&](int) { FAIL() << "This shouldn't be called."; });

  std::visit(visitor, std::variant<char, double>('x'));
  EXPECT_THAT(executed_char_visitor, IsTrue());
  EXPECT_THAT(executed_double_visitor, IsFalse());

  std::visit(visitor, std::variant<char, double>(1.5));
  EXPECT_THAT(executed_char_visitor, IsFalse());
  EXPECT_THAT(executed_double_visitor, IsTrue());
}

struct SizeOfVisitor {
  template <class T>
  std::size_t operator()(const T&) const {
    return sizeof(T);
  }
};

struct Padded {
  char pad[64];
};
constexpr size_t kPaddedSize = sizeof(Padded);

TEST(MakeVisitorTest, TemplatedVisitor) {
  using VariantT = std::variant<char, double, std::nullptr_t, Padded>;

  const auto visitor =
      MakeVisitor([](std::nullptr_t) -> std::size_t { return 0; },  //
                  SizeOfVisitor{},                                  //
                  [](char) -> std::size_t { return -1; });
  EXPECT_THAT(std::visit(visitor, VariantT('x')), Eq(static_cast<size_t>(-1)));
  EXPECT_THAT(std::visit(visitor, VariantT(1.5)), Eq(sizeof(double)));
  EXPECT_THAT(std::visit(visitor, VariantT(nullptr)), Eq(0));
  EXPECT_THAT(std::visit(visitor, VariantT(Padded{})), Eq(kPaddedSize));
}

}  // namespace
}  // namespace envlogger
