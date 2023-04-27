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

#include "envlogger/backends/cc/riegeli_shard_reader.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include <cstdint>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/variant.h"
#include "envlogger/backends/cc/episode_info.h"
#include "envlogger/converters/make_visitor.h"
#include "envlogger/converters/xtensor_codec.h"
#include "envlogger/platform/filesystem.h"
#include "envlogger/platform/riegeli_file_reader.h"
#include "envlogger/platform/status_macros.h"
#include "envlogger/proto/storage.pb.h"
#include "riegeli/records/record_reader.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xaxis_iterator.hpp"

namespace envlogger {
namespace {


absl::StatusOr<riegeli::RecordReader<RiegeliFileReader>> CreateReader(
    absl::string_view filepath) {
  VLOG(1) << "Creating the reader for filepath: " << filepath;
  std::string reader_filepath(filepath);
  riegeli::RecordReader reader{RiegeliFileReader(reader_filepath)};
  if (!reader.ok()) return reader.status();
  return reader;
}

}  // namespace


RiegeliShardReader::~RiegeliShardReader() { Close(); }

absl::Status RiegeliShardReader::Init(
    absl::string_view steps_filepath, absl::string_view step_offsets_filepath,
    absl::string_view episode_metadata_filepath,
    absl::string_view episode_index_filepath) {
  // Initialize first the structure to the shard (empty structure).
  shard_ = std::make_shared<ShardData>();

  if (step_offsets_filepath.empty()) {
    return absl::NotFoundError(absl::StrCat(
        "Could not find step_offsets_filepath==", step_offsets_filepath));
  }

  std::vector<int64_t>* steps = &shard_->step_offsets;
  std::vector<int64_t>* episode_starts = &shard_->episode_starts;
  std::vector<int64_t>* episode_metadata_offsets =
      &shard_->episode_metadata_offsets;
  const auto steps_visitor = MakeVisitor(
      [steps](const xt::xarray<int64_t>& a) {
        steps->insert(std::end(*steps), std::begin(a), std::end(a));
      },
      [](const auto&) { /* catch all overload */ });
  const auto episode_index_visitor = MakeVisitor(
      [episode_starts, episode_metadata_offsets](const xt::xarray<int64_t>& a) {
        for (auto it = xt::axis_begin(a), end = xt::axis_end(a); it != end;
             ++it) {
          episode_starts->push_back((*it)(0));
          episode_metadata_offsets->push_back((*it)(1));
        }
      },
      [](const auto&) { /* catch all overload */ });

  Datum value;

  // Read step offsets.
  ENVLOGGER_ASSIGN_OR_RETURN(riegeli::RecordReader step_offsets_reader,
                             CreateReader(step_offsets_filepath));
  while (step_offsets_reader.ReadRecord(value)) {
    const std::optional<BasicType> step_offsets_decoded = Decode(value);
    if (step_offsets_decoded) absl::visit(steps_visitor, *step_offsets_decoded);
  }

  // Read episode index.
  ENVLOGGER_ASSIGN_OR_RETURN(riegeli::RecordReader episode_index_reader,
                             CreateReader(episode_index_filepath));
  while (episode_index_reader.ReadRecord(value)) {
    const std::optional<BasicType> episode_index_decoded = Decode(value);
    if (episode_index_decoded)
      absl::visit(episode_index_visitor, *episode_index_decoded);
  }

  if (shard_->step_offsets.empty()) {
    return absl::NotFoundError(
        absl::StrCat("Empty steps in ", step_offsets_filepath));
  }

  VLOG(1) << "step_offsets.size(): " << shard_->step_offsets.size();
  VLOG(1) << "episode_starts.size(): " << shard_->episode_starts.size();

  ENVLOGGER_ASSIGN_OR_RETURN(steps_reader_, CreateReader(steps_filepath));
  ENVLOGGER_ASSIGN_OR_RETURN(episode_metadata_reader_,
                             CreateReader(episode_metadata_filepath));

  shard_->steps_filepath = steps_filepath;
  shard_->episode_metadata_filepath = episode_metadata_filepath;
  return absl::OkStatus();
}

absl::StatusOr<RiegeliShardReader> RiegeliShardReader::Clone() {
  RiegeliShardReader cloned_reader;
  cloned_reader.shard_ = shard_;
  ENVLOGGER_ASSIGN_OR_RETURN(cloned_reader.steps_reader_,
                             CreateReader(shard_->steps_filepath));
  if (!shard_->episode_metadata_filepath.empty()) {
    ENVLOGGER_ASSIGN_OR_RETURN(cloned_reader.episode_metadata_reader_,
                               CreateReader(shard_->episode_metadata_filepath));
  }
  return cloned_reader;
}

int64_t RiegeliShardReader::NumSteps() const {
  return shard_->step_offsets.size();
}

int64_t RiegeliShardReader::NumEpisodes() const {
  return shard_->episode_starts.size();
}

std::optional<EpisodeInfo> RiegeliShardReader::Episode(int64_t episode_index,
                                                       bool include_metadata) {
  const auto& episode_starts = shard_->episode_starts;
  const auto& step_offsets = shard_->step_offsets;
  const auto& episode_metadata_offsets = shard_->episode_metadata_offsets;
  if (episode_index < 0 ||
      episode_index >= static_cast<int64_t>(episode_starts.size())) {
    return std::nullopt;
  }

  const int64_t start_index = episode_starts[episode_index];
  const int64_t num_steps =
      episode_index + 1 < static_cast<int64_t>(episode_starts.size())
          ? episode_starts[episode_index + 1] - start_index
          : step_offsets.size() - start_index;
  EpisodeInfo episode_info{start_index, num_steps};
  if (include_metadata &&
      episode_metadata_offsets.size() == episode_starts.size()) {
    if (const int64_t offset = episode_metadata_offsets[episode_index];
        offset > 0 && episode_metadata_reader_.Seek(offset)) {
      Data metadata;
      const bool read_status = episode_metadata_reader_.ReadRecord(metadata);
      if (read_status) {
        episode_info.metadata = std::move(metadata);
      } else {
        VLOG(1) << "Failed to read metadata for episode " << episode_index
                << " using offset " << offset
                << ". reader status: " << episode_metadata_reader_.status();
      }
    } else {
      VLOG(1) << "No metadata for episode " << episode_index
              << ". reader status: " << episode_metadata_reader_.status();
    }
  }

  return episode_info;
}

void RiegeliShardReader::Close() {
  const bool steps_close_status = steps_reader_.Close();
  VLOG(1) << "steps_close_status: " << steps_close_status;
  const bool episode_metadata_reader_close_status =
      episode_metadata_reader_.Close();
  VLOG(1) << "episode_metadata_reader_close_status: "
          << episode_metadata_reader_close_status;
}

}  // namespace envlogger
