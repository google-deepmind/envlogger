// Copyright 2023 DeepMind Technologies Limited..
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

#ifndef THIRD_PARTY_PY_ENVLOGGER_PLATFORM_FILESYSTEM_H_
#define THIRD_PARTY_PY_ENVLOGGER_PLATFORM_FILESYSTEM_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "envlogger/platform/default/filesystem.h"

namespace envlogger {
namespace file {

// Join multiple paths together.
std::string JoinPath(absl::string_view dirname, absl::string_view basename);

// Creates a directory with the given path.
// Fails if the directory cannot be created (e.g. it already exists).
absl::Status CreateDir(absl::string_view path);

// Returns the list of subdirectories under the specified path. If the sentinel
// is not empty, then only subdirectories that contain a file with that name
// will be present.
absl::StatusOr<std::vector<std::string>> GetSubdirectories(
    absl::string_view path, absl::string_view sentinel = "");

// Recursively deletes the specified path.
absl::Status RecursivelyDelete(absl::string_view path);

// Returns the file size in bytes.
absl::StatusOr<int64_t> GetSize(absl::string_view path);

}  // namespace file
}  // namespace envlogger

#endif  // THIRD_PARTY_PY_ENVLOGGER_PLATFORM_FILESYSTEM_H_
