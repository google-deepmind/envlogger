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

// Struct for efficiently retrieving episodic information.
#ifndef THIRD_PARTY_PY_ENVLOGGER_BACKENDS_CC_EPISODE_INFO_H_
#define THIRD_PARTY_PY_ENVLOGGER_BACKENDS_CC_EPISODE_INFO_H_

#include <cstdint>

#include "absl/types/optional.h"
#include "envlogger/proto/storage.pb.h"

namespace envlogger {

struct EpisodeInfo {
  // The step index where this episode starts.
  // This can be either a local step within a single trajectory file, or a
  // global step across many shards.
  int64_t start;
  // The number of steps in this episode.
  int64_t num_steps;
  // Optional metadata which is only filled if requested.
  absl::optional<Data> metadata;
};

}  // namespace envlogger

#endif  // THIRD_PARTY_PY_ENVLOGGER_BACKENDS_CC__EPISODE_INFO_H_
