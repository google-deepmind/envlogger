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

#include "envlogger/backends/cc/riegeli_dataset_reader.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "google/protobuf/message.h"
#include <cstdint>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "envlogger/backends/cc/episode_info.h"
#include "envlogger/backends/cc/riegeli_shard_reader.h"
#include "envlogger/backends/cc/riegeli_shard_writer.h"
#include "envlogger/backends/cc/riegeli_dataset_io_constants.h"
#include "envlogger/platform/bundle.h"
#include "envlogger/platform/filesystem.h"
#include "envlogger/platform/riegeli_file_reader.h"
#include "envlogger/platform/status_macros.h"
#include "envlogger/proto/storage.pb.h"
#include "riegeli/base/base.h"
#include "riegeli/records/record_reader.h"
#include "riegeli/records/record_writer.h"

namespace envlogger {

namespace {

// Returns the first Data record in the file pointed by `filepath`.
absl::StatusOr<Data> ReadFirstRiegeliRecord(const absl::string_view filepath) {
  riegeli::RecordReader reader{RiegeliFileReader(filepath)};
  ENVLOGGER_RETURN_IF_ERROR(reader.status());

  Data data;
  if (!reader.ReadRecord(data)) {
    return absl::InternalError("Failed to parse riegeli record.");
  }

  if (!reader.Close()) return reader.status();

  return data;
}

}  // namespace

absl::Status RiegeliDatasetReader::Init(absl::string_view data_dir) {
  ENVLOGGER_ASSIGN_OR_RETURN(
      std::vector<std::string> matches,
      file::GetSubdirectories(data_dir, internal::kStepOffsetsFilename));

  // Sort matches to have a deterministic and increasing order.
  std::sort(std::begin(matches), std::end(matches));

  // Allocate space to avoid reallocation so that shards can be initialized in
  // parallel.
  shards_.resize(matches.size());
  // Create a container for storing the results of all initializations.
  // The size of this container is |matches| + 1 (from reading metadata)
  std::vector<absl::Status> init_results(matches.size() + 1);
  const int metadata_index = matches.size();

  auto initialize_shard =
      [](absl::string_view timestamp_dir,
         RiegeliDatasetReader::Shard* shard) -> absl::Status {
    const std::string steps_file =
        file::JoinPath(timestamp_dir, internal::kStepsFilename);
    const std::string step_offsets_file =
        file::JoinPath(timestamp_dir, internal::kStepOffsetsFilename);
    const std::string episode_metadata_file =
        file::JoinPath(timestamp_dir, internal::kEpisodeMetadataFilename);
    const std::string episode_index_file =
        file::JoinPath(timestamp_dir, internal::kEpisodeIndexFilename);
    return shard->index.Init(steps_file, step_offsets_file,
                             episode_metadata_file, episode_index_file);
  };

  thread::Bundle bundle;
  for (size_t i = 0; i < matches.size(); ++i) {
    const std::string& timestamp_dir = matches[i];
    auto* shard = &shards_[i];
    bundle.Add([shard, &timestamp_dir, initialize_shard,
                init_result = &init_results[i]]() {
      shard->timestamp_dir = timestamp_dir;
      *init_result = initialize_shard(timestamp_dir, shard);
    });
  }

  // Read metadata.
  bundle.Add([metadata = &metadata_, data_dir,
              init_result = &init_results[metadata_index]]() {
    const std::string metadata_file =
        file::JoinPath(data_dir, internal::kMetadataFilename);
    if (auto data_or = ReadFirstRiegeliRecord(metadata_file); data_or.ok()) {
      *metadata = std::move(*data_or);
    } else {
      *init_result = data_or.status();
    }
  });

  // Execute everything in parallel.
  bundle.JoinAll();

  // Build cumulative maps for faster retrieval.
  total_num_steps_ = 0;
  total_num_episodes_ = 0;
  for (auto& shard : shards_) {
    shard.global_step_index = total_num_steps_;
    shard.cumulative_steps = total_num_steps_;
    shard.cumulative_episodes = total_num_episodes_;
    total_num_steps_ += shard.index.NumSteps();
    total_num_episodes_ += shard.index.NumEpisodes();
  }

  // Return an error if anything went wrong at initialization.
  for (const auto& status : init_results) ENVLOGGER_RETURN_IF_ERROR(status);

  return absl::OkStatus();
}

absl::optional<Data> RiegeliDatasetReader::Metadata() const {
  return metadata_;
}

int64_t RiegeliDatasetReader::NumSteps() const { return total_num_steps_; }

int64_t RiegeliDatasetReader::NumEpisodes() const {
  return total_num_episodes_;
}

absl::optional<EpisodeInfo> RiegeliDatasetReader::Episode(
    int64_t episode_index, bool include_metadata) {
  if (episode_index < 0 || episode_index >= NumEpisodes()) return absl::nullopt;

  Shard* shard = nullptr;
  int64_t local_episode_index = -1;
  std::tie(shard, local_episode_index) =
      FindShard(episode_index, [](const RiegeliDatasetReader::Shard& shard) {
        return shard.cumulative_episodes;
      });

  auto episode = shard->index.Episode(local_episode_index, include_metadata);
  // Change episode index from local to global.
  episode->start += shard->global_step_index;

  return *episode;
}

absl::StatusOr<RiegeliShardReader> RiegeliDatasetReader::GetShard(
    int64_t episode_index) {
  if (episode_index < 0 || episode_index >= NumEpisodes()) {
    return absl::OutOfRangeError(
        absl::StrCat("Out of range episode index: ", episode_index));
  }

  Shard* shard = nullptr;
  int64_t local_episode_index = -1;
  std::tie(shard, local_episode_index) =
      FindShard(episode_index, [](const RiegeliDatasetReader::Shard& shard) {
        return shard.cumulative_episodes;
      });

  return shard->index.Clone();
}

RiegeliDatasetReader::~RiegeliDatasetReader() { Close(); }

void RiegeliDatasetReader::Close() {
  for (auto& shard : shards_) shard.index.Close();
}

std::pair<RiegeliDatasetReader::Shard*, int64_t>
RiegeliDatasetReader::FindShard(
    int64_t global_index, std::function<int64_t(const Shard&)> extractor) {
  // Determine the shard by binary searching shards_.
  auto it = std::lower_bound(
      std::rbegin(shards_), std::rend(shards_), global_index,
      [extractor](const Shard& a, int64_t b) { return extractor(a) > b; });

  Shard* shard = std::addressof(*it);
  const int64_t local_index = global_index - extractor(*it);
  return {shard, local_index};
}

}  // namespace envlogger
