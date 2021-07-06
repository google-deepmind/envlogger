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

#ifndef THIRD_PARTY_PY_ENVLOGGER_PLATFORM_DEFAULT_RIEGELI_FILE_READER_H_
#define THIRD_PARTY_PY_ENVLOGGER_PLATFORM_DEFAULT_RIEGELI_FILE_READER_H_

#include "absl/strings/string_view.h"
#include "envlogger/platform/default/filesystem.h"
#include "riegeli/bytes/fd_reader.h"

namespace envlogger {

// Helper class for reading Riegeli files.
template <typename Src = ::riegeli::OwnedFd>
class RiegeliFileReader : public ::riegeli::FdReader<Src> {
 public:
  // Creates a closed `FileReader`.
  RiegeliFileReader() noexcept {}

  explicit RiegeliFileReader(absl::string_view filename, absl::string_view mode)
      : ::riegeli::FdReader<Src>(filename, file::GetFileMode(mode)) {}
};

}  // namespace envlogger

#endif  // THIRD_PARTY_PY_ENVLOGGER_PLATFORM_DEFAULT_RIEGELI_FILE_READER_H_
