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

#ifndef THIRD_PARTY_PY_ENVLOGGER_PLATFORM_DEFAULT_RIEGELI_FILE_WRITER_H_
#define THIRD_PARTY_PY_ENVLOGGER_PLATFORM_DEFAULT_RIEGELI_FILE_WRITER_H_

#include "absl/strings/string_view.h"
#include "envlogger/platform/default/filesystem.h"
#include "riegeli/base/object.h"
#include "riegeli/bytes/fd_writer.h"

namespace envlogger {

// Helper class for writing Riegeli files.
template <typename Dest = ::riegeli::OwnedFd>
class RiegeliFileWriter : public ::riegeli::FdWriter<Dest> {
 public:
  // Creates a closed `FileWriter`.
  explicit RiegeliFileWriter(riegeli::Closed) noexcept
      : ::riegeli::FdWriter<Dest>(riegeli::kClosed) {}

  explicit RiegeliFileWriter(absl::string_view filename, absl::string_view mode)
      : ::riegeli::FdWriter<Dest>(filename, file::GetFileMode(mode)) {}
};

}  // namespace envlogger

#endif  // THIRD_PARTY_PY_ENVLOGGER_PLATFORM_DEFAULT_RIEGELI_FILE_WRITER_H_
