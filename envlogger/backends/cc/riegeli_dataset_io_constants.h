// Copyright 2024 DeepMind Technologies Limited..
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

#ifndef THIRD_PARTY_PY_ENVLOGGER_BACKENDS_CC_RIEGELI_DATASET_IO_CONSTANTS_H_
#define THIRD_PARTY_PY_ENVLOGGER_BACKENDS_CC_RIEGELI_DATASET_IO_CONSTANTS_H_

#include "absl/strings/string_view.h"

namespace envlogger {
// These constants are used internally in our codebase and should not be relied
// upon by clients.
namespace internal {
// The riegeli filename for metadata set at Init() time.
inline constexpr absl::string_view kMetadataFilename = "metadata.riegeli";
// Steps (timesteps, actions and per-step metadata).
inline constexpr absl::string_view kStepsFilename = "steps.riegeli";
// Step offsets for faster seeking into kStepsFilename.
inline constexpr absl::string_view kStepOffsetsFilename =
    "step_offsets.riegeli";
// Episodic metadata.
inline constexpr absl::string_view kEpisodeMetadataFilename =
    "episode_metadata.riegeli";
// Episode offsets for faster seeking into kEpisodeMetadataFilename and also
// episode starts.
inline constexpr absl::string_view kEpisodeIndexFilename =
    "episode_index.riegeli";
}  // namespace internal
}  // namespace envlogger

#endif  // THIRD_PARTY_PY_ENVLOGGER_BACKENDS_CC_RIEGELI_DATASET_IO_CONSTANTS_H_
