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

#ifndef THIRD_PARTY_PY_ENVLOGGER_PLATFORM_TEST_MACROS_H_
#define THIRD_PARTY_PY_ENVLOGGER_PLATFORM_TEST_MACROS_H_

#include "gtest/gtest.h"
#include "envlogger/platform/status_macros.h"

#define CONCAT_IMPL(x, y) x##y
#define CONCAT_MACRO(x, y) CONCAT_IMPL(x, y)

#define ENVLOGGER_ASSERT_OK_AND_ASSIGN(lhs, rexpr)                           \
  ENVLOGGER_ASSERT_OK_AND_ASSIGN_IMPL(CONCAT_MACRO(_status_or, __COUNTER__), \
                                      lhs, rexpr)

#define ENVLOGGER_ASSERT_OK_AND_ASSIGN_IMPL(statusor, lhs, rexpr) \
  auto statusor = (rexpr);                                        \
  ASSERT_TRUE(statusor.status().ok()) << statusor.status();       \
  lhs = std::move(statusor.value())

#define ENVLOGGER_EXPECT_OK(expr) ENVLOGGER_CHECK_OK(expr)

#endif  // THIRD_PARTY_PY_ENVLOGGER_PLATFORM_TEST_MACROS_H_
