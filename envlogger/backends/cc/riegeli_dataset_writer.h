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

#ifndef THIRD_PARTY_PY_ENVLOGGER_BACKENDS_CC_RIEGELI_DATASET_WRITER_H_
#define THIRD_PARTY_PY_ENVLOGGER_BACKENDS_CC_RIEGELI_DATASET_WRITER_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/message.h"
#include <cstdint>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "envlogger/backends/cc/episode_info.h"
#include "envlogger/backends/cc/riegeli_shard_reader.h"
#include "envlogger/backends/cc/riegeli_shard_writer.h"
#include "envlogger/proto/storage.pb.h"

namespace envlogger {

// Automates creating trajectories that can be efficiently read from disk by
// RiegeliDatasetReader.
class RiegeliDatasetWriter {
 public:
  RiegeliDatasetWriter() = default;

  // Initializes this writer to the following `data_dir`.
  // `metadata` is a client-specific payload.
  // `max_episodes_per_shard` determines the maximum number of episodes a single
  // RiegeliShardWriter shard will hold. If non-positive, a single shard file
  // will hold all steps and episodes.
  //
  // IMPORTANT: `data_dir` MUST exist _before_ calling Init().
  absl::Status Init(
      std::string data_dir, const Data& metadata = Data(),
      int64_t max_episodes_per_shard = 0,
      std::string writer_options = "transpose,brotli:6,chunk_size:1M");

  // Adds `data` to the trajectory and marks it as a new episode if
  // `is_new_episode==true`.
  // Returns true if `data` has been written, false otherwise.
  bool AddStep(const google::protobuf::Message& data, bool is_new_episode = false);

  // Sets episodic metadata for the _current_ episode.
  // This can be called multiple times but the value will be written only when a
  // new episode comes in or when this writer is about to be destructed.
  // Notice that calling this before an `AddStep(..., /*is_new_episode=*/true)`
  // is called leads to this writer ignoring the `data` that's passed.
  void SetEpisodeMetadata(const Data& data);

  void Flush();

  // Finalizes all writes and releases all handles.
  void Close();

 private:
  std::string data_dir_;
  std::string writer_options_;
  int64_t max_episodes_per_shard_ = 0;
  int episode_counter_ = 0;
  RiegeliShardWriter writer_;
};

}  // namespace envlogger

#endif  // THIRD_PARTY_PY_ENVLOGGER_BACKENDS_CC_RIEGELI_DATASET_WRITER_H_
