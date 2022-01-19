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

#include "envlogger/platform/default/filesystem.h"

#include <fcntl.h>

#include <filesystem>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "envlogger/platform/status_macros.h"

namespace envlogger {
namespace file {

std::string JoinPath(absl::string_view dirname, absl::string_view basename) {
  return std::filesystem::path(dirname) / basename;
}

absl::Status CreateDir(absl::string_view path) {
  if (!std::filesystem::create_directory(path)) {
    return absl::InternalError(
        absl::StrCat("Unable to create directory ", path));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::string>> GetSubdirectories(
    absl::string_view path, absl::string_view sentinel) {
  std::vector<std::string> subdirs;
  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    if (entry.is_directory()) {
      if (!sentinel.empty() &&
          !std::filesystem::exists(entry.path() / sentinel)) {
        continue;
      }
      subdirs.push_back(entry.path());
    }
  }
  return subdirs;
}

absl::Status RecursivelyDelete(absl::string_view path) {
  if (!std::filesystem::remove_all(path)) {
    return absl::InternalError(
        absl::StrCat("Unable to recursively delete directory ", path));
  }
  return absl::OkStatus();
}

absl::StatusOr<int64_t> GetSize(absl::string_view path) {
  std::error_code ec;
  const std::uintmax_t size = std::filesystem::file_size(path, ec);
  if (ec) {
    return absl::NotFoundError(absl::StrCat("Could not find file ", path));
  }
  return static_cast<int64_t>(size);
}

int GetFileMode(absl::string_view mode) {
  if (mode == "r") {
    return O_RDONLY;
  }
  if (mode == "w") {
    return O_WRONLY | O_CREAT | O_TRUNC;
  }
  return 0;
}

}  // namespace file
}  // namespace envlogger
