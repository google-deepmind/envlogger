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

#ifndef THIRD_PARTY_PY_ENVLOGGER_BACKENDS_CC_RIEGELI_DATASET_READER_H_
#define THIRD_PARTY_PY_ENVLOGGER_BACKENDS_CC_RIEGELI_DATASET_READER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/message.h"
#include <cstdint>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "envlogger/backends/cc/episode_info.h"
#include "envlogger/backends/cc/riegeli_shard_reader.h"
#include "envlogger/backends/cc/riegeli_shard_writer.h"
#include "envlogger/proto/storage.pb.h"

namespace envlogger {

// A RiegeliDatasetReader contains trajectory information for the entire
// lifetime of a single Reinforcement Learning (RL) actor. Each
// RiegeliDatasetReader represents a one dimensional trajectory composed of 0 or
// more steps. Steps can be grouped into non-overlapping, contiguous episodes
// for episodic RL environments.
//
// Internally, a RiegeliDatasetReader is sharded into chronologically ordered
// sub directories called "timestamp directories". Each of these sub directories
// contain their own trajectories and indexes, so in order to find a specific
// step or episode we first need to determine the sub directory and then their
// internal index. An episode is never split between two shards.
class RiegeliDatasetReader {
 public:
  RiegeliDatasetReader() = default;
  RiegeliDatasetReader(RiegeliDatasetReader&&) = default;
  ~RiegeliDatasetReader();

  // Clones a RiegeliDatasetReader.
  // The cloned reader can be safely used in a different thread.
  absl::StatusOr<RiegeliDatasetReader> Clone();

  // Releases resources and closes the underlying files.
  void Close();

  absl::Status Init(absl::string_view data_dir);

  // Returns metadata associated with this RiegeliDatasetReader.
  std::optional<Data> Metadata() const;

  int64_t NumSteps() const;
  int64_t NumEpisodes() const;

  // Returns step data at index `step_index`.
  // Returns nullopt if `step_index` is not in [0, NumSteps()).
  template <typename T = Data>
  std::optional<T> Step(int64_t step_index);

  // Returns information for accessing a specific episode.
  //
  // NOTE: `start` in the returned object refers to the "global" step index, not
  // the local one in the shard. That is, this value can be passed to Step()
  // below without any modification.
  //
  // Returns nullopt if `episode_index` is not in [0, NumEpisodes()).
  std::optional<EpisodeInfo> Episode(int64_t episode_index,
                                     bool include_metadata = false);

  // Returns a shard reader for the specified episode index.
  absl::StatusOr<RiegeliShardReader> GetShard(int64_t episode_index);

 private:
  // A Shard represents a single timestamp directory.
  struct Shard {
    // The path to the timestamp directory.
    std::string timestamp_dir;
    // The index internal to this timestamp directory.
    RiegeliShardReader index;

    // The global step index at which this shard starts.
    int64_t global_step_index = -1;

    // The cumulative number of steps up to this shard (in the order that it's
    // inserted in shards_).
    int64_t cumulative_steps = 0;
    // The cumulative number of episodes up to this shard (in the order that
    // it's inserted in shards_).
    int64_t cumulative_episodes = 0;
  };

  // Returns the first element in `shards_` that is not greater than
  // `global_index`.
  std::pair<Shard*, int64_t> FindShard(
      int64_t global_index, std::function<int64_t(const Shard&)> extractor);

  // The total number of steps across all timestamp directories.
  int64_t total_num_steps_ = 0;
  // The total number of episodes across all timestamp directories.
  // Note: This implementation differs from the Python one because the latter
  // depends on an explicit "max_episodes_per_file" entry in the metadata while
  // this one calculates the same thing via binary search.
  int64_t total_num_episodes_ = 0;

  std::optional<Data> metadata_;

  // The list of all shards in this RiegeliDatasetReader.
  std::vector<Shard> shards_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation details of RiegeliDatasetReader::Step.
////////////////////////////////////////////////////////////////////////////////

template <typename T>
std::optional<T> RiegeliDatasetReader::Step(int64_t step_index) {
  if (step_index < 0 || step_index >= NumSteps()) return absl::nullopt;

  Shard* shard = nullptr;
  int64_t local_step_index = -1;
  std::tie(shard, local_step_index) =
      FindShard(step_index, [](const RiegeliDatasetReader::Shard& shard) {
        return shard.cumulative_steps;
      });
  return shard->index.Step<T>(local_step_index);
}

}  // namespace envlogger

#endif  // THIRD_PARTY_PY_ENVLOGGER_BACKENDS_CC_RIEGELI_DATASET_READER_H_
