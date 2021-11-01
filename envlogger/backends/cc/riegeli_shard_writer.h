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

#ifndef THIRD_PARTY_PY_ENVLOGGER_BACKENDS_CC_RIEGELI_SHARD_WRITER_H_
#define THIRD_PARTY_PY_ENVLOGGER_BACKENDS_CC_RIEGELI_SHARD_WRITER_H_

#include <cstdint>
#include <vector>

#include "google/protobuf/message.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "envlogger/backends/cc/episode_info.h"
#include "envlogger/platform/riegeli_file_writer.h"
#include "envlogger/proto/storage.pb.h"
#include "riegeli/base/object.h"
#include "riegeli/records/record_writer.h"

namespace envlogger {

// Creates Riegeli index files to allow for efficient record lookup from another
// Riegeli file (i.e. an external index).
class RiegeliShardWriter {
 public:
  RiegeliShardWriter() = default;
  ~RiegeliShardWriter();


  // IMPORTANT: The directory where these file live MUST exist _before_ calling
  //            Init().
  //
  // steps_filepath:
  //   Path to riegeli file that'll contain the actual trajectory data as Data
  //   objects, including timesteps (observations, rewards, discounts), actions
  //   and step metadata.
  //   Each riegeli entry corresponds to a single step.
  //   Each entry can be seeked in O(1) time via an external index
  //   (step_offsets_filepath).
  // step_offsets_filepath:
  //   An external index into steps_filepath. This allows clients to access any
  //   individual step in O(1). Riegeli entries are Datum objects.
  // episode_metadata_filepath:
  //   Path to riegeli file that'll contain episodic metadata (e.g. discounted
  //   returns, summarized data etc). Riegeli entries are Data objects.
  //   Each entry corresponds to the metadata of a single episode.
  //   Metadata is stored only for episodes that have it (i.e. if an episode
  //   does not store any metadata, nothing is stored here).
  //   Each entry can be seeked in O(1) time via an external index
  //   (episode_index_filepath).
  // episode_index_filepath:
  //   Path to a riegeli file containing (|Episodes| x 2) tensors where:
  //   * 1st dim is a step index indicating the start of the episode.
  //   * 2nd dim is a riegeli offset for optional metadata (for looking up into
  //     episode_metadata_filepath). -1 indicates that no metadata exists for a
  //     specific episode.
  //   Each riegeli entry is a Datum object.
  absl::Status Init(absl::string_view steps_filepath,
                    absl::string_view step_offsets_filepath,
                    absl::string_view episode_metadata_filepath,
                    absl::string_view episode_index_filepath,
                    absl::string_view writer_options);

  void AddStep(const google::protobuf::Message& data, bool is_new_episode = false);

  // Sets episodic metadata for the _current_ episode.
  // This can be called multiple times but the value will be written only when a
  // new episode comes in or when this writer is about to be destructed.
  // Notice that calling this before an `AddStep(..., /*is_new_episode=*/true)`
  // is called leads to this writer ignoring the `data` that's passed.
  void SetEpisodeMetadata(const Data& data);

  // Flushes the index to disk.
  void Flush();

  // Finalizes all writes and releases all handles.
  void Close();

 private:
  // The number of steps in the last flush.
  int num_steps_at_flush_ = 0;

  std::vector<int64_t> step_offsets_;

  // The first steps of each episodes.
  // Episode index -> Step index.
  std::vector<int64_t> episode_starts_;
  // The riegeli offset into `episode_metadata_writer_`.
  std::vector<int64_t> episode_offsets_;

  // Metadata for the _current_ episode.
  absl::optional<Data> episode_metadata_;

  // Steps, episodes and their riegeli numeric offsets.
  riegeli::RecordWriter<RiegeliFileWriter<>> steps_writer_{riegeli::kClosed};
  riegeli::RecordWriter<RiegeliFileWriter<>> step_offsets_writer_{
      riegeli::kClosed};
  riegeli::RecordWriter<RiegeliFileWriter<>> episode_metadata_writer_{
      riegeli::kClosed};
  riegeli::RecordWriter<RiegeliFileWriter<>> episode_index_writer_{
      riegeli::kClosed};

};

}  // namespace envlogger

#endif  // THIRD_PARTY_PY_ENVLOGGER_BACKENDS_CC_RIEGELI_SHARD_WRITER_H_
