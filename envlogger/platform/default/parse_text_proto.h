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

#ifndef THIRD_PARTY_PY_ENVLOGGER_PLATFORM_DEFAULT_PARSE_TEXT_PROTO_H_
#define THIRD_PARTY_PY_ENVLOGGER_PLATFORM_DEFAULT_PARSE_TEXT_PROTO_H_

#include "glog/logging.h"
#include "google/protobuf/text_format.h"

namespace envlogger {
// Forward declarations for the friend statement.
namespace internal {
class ParseProtoHelper;
}  // namespace internal
internal::ParseProtoHelper ParseTextProtoOrDie(const std::string& input);

namespace internal {
// Helper class to automatically infer the type of the protocol buffer message.
class ParseProtoHelper {
 public:
  template <typename T>
  operator T() {
    T result;
    CHECK(google::protobuf::TextFormat::ParseFromString(input_, &result));
    return result;
  }

 private:
  friend ParseProtoHelper envlogger::ParseTextProtoOrDie(
      const std::string& input);
  ParseProtoHelper(const std::string& input) : input_(input) {}

  const std::string& input_;
};
}  // namespace internal

// Parses the specified ASCII protocol buffer message or dies.
internal::ParseProtoHelper ParseTextProtoOrDie(const std::string& input) {
  return internal::ParseProtoHelper(input);
}

// Parses the specified ASCII protocol buffer message and returns it or dies.
template <typename T>
T ParseTextOrDie(const std::string& input) {
  return ParseTextProtoOrDie(input);
}

}  // namespace envlogger

#endif  // THIRD_PARTY_PY_ENVLOGGER_PLATFORM_DEFAULT_PARSE_TEXT_PROTO_H_
