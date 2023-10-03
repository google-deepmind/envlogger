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

#include "envlogger/backends/cc/riegeli_shard_writer.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "glog/logging.h"
#include "google/protobuf/message.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "envlogger/converters/xtensor_codec.h"
#include "envlogger/platform/riegeli_file_writer.h"
#include "envlogger/platform/status_macros.h"
#include "envlogger/proto/storage.pb.h"
#include "riegeli/base/types.h"
#include "riegeli/records/record_position.h"
#include "riegeli/records/record_writer.h"
#include "xtensor/xadapt.hpp"
#include "xtensor/xview.hpp"

namespace envlogger {
namespace {

// Writes any remaining episodic information which can only be written once an
// episode is finalized.
void WriteLastEpisodeIndex(std::vector<int64_t>* episode_starts,
                           std::optional<Data>* episode_metadata,
                           riegeli::RecordWriterBase* episode_metadata_writer,
                           riegeli::RecordWriterBase* episode_index_writer) {
  if (episode_starts->empty()) return;

  int64_t episode_offset = -1;
  if (*episode_metadata) {
    if (!episode_metadata_writer->WriteRecord(**episode_metadata)) {
      VLOG(0) << "episode_metadata_writer->status(): "
              << episode_metadata_writer->status();
    }
    episode_offset = episode_metadata_writer->LastPos().get().numeric();
    if (!episode_metadata_writer->Flush(riegeli::FlushType::kFromMachine)) {
      VLOG(0) << "episode_metadata_writer->status(): "
              << episode_metadata_writer->status();
    }
    episode_metadata->reset();
  }

  xt::xarray<int64_t> episode_index = {{(*episode_starts)[0], episode_offset}};
  if (const Datum episode_index_datum = Encode(episode_index);
      !episode_index_writer->WriteRecord(episode_index_datum) ||
      !episode_index_writer->Flush(riegeli::FlushType::kFromMachine)) {
    VLOG(0) << "episode_index_writer->status(): "
            << episode_index_writer->status();
  }
  episode_starts->clear();
}

}  // namespace

absl::Status RiegeliShardWriter::Init(
    absl::string_view steps_filepath, absl::string_view step_offsets_filepath,
    absl::string_view episode_metadata_filepath,
    absl::string_view episode_index_filepath,
    absl::string_view writer_options) {
  // Write any remaining episodic data that can result from reusing this object.
  WriteLastEpisodeIndex(&episode_starts_, &episode_metadata_,
                        &episode_metadata_writer_, &episode_index_writer_);

  num_steps_at_flush_ = 0;
  riegeli::RecordWriterBase::Options options;
  ENVLOGGER_RETURN_IF_ERROR(options.FromString(writer_options));
  steps_writer_.Reset(RiegeliFileWriter(steps_filepath), options);
  ENVLOGGER_RETURN_IF_ERROR(steps_writer_.status());
  step_offsets_writer_.Reset(RiegeliFileWriter(step_offsets_filepath), options);
  ENVLOGGER_RETURN_IF_ERROR(step_offsets_writer_.status());
  episode_metadata_writer_.Reset(RiegeliFileWriter(episode_metadata_filepath),
                                 options);
  ENVLOGGER_RETURN_IF_ERROR(episode_metadata_writer_.status());
  episode_index_writer_.Reset(RiegeliFileWriter(episode_index_filepath),
                              options);
  ENVLOGGER_RETURN_IF_ERROR(episode_index_writer_.status());
  return absl::OkStatus();
}

RiegeliShardWriter::~RiegeliShardWriter() { Close(); }

void RiegeliShardWriter::Flush() {
  VLOG(2) << "Flush()";
  if (step_offsets_.empty()) {
    VLOG(0) << "Empty steps, nothing to do. ~Flush()";
    return;
  }

  const size_t num_steps = step_offsets_.size();
  const size_t num_episodes = episode_starts_.size();
  num_steps_at_flush_ += num_steps;

  xt::xarray<int64_t> step_offsets = xt::adapt(step_offsets_, {num_steps});
  xt::xarray<int64_t> episode_starts =
      xt::adapt(episode_starts_, {num_episodes});

  // Flush steps.
  if (!steps_writer_.Flush(riegeli::FlushType::kFromMachine)) {
    VLOG(0) << "steps_writer_.status(): " << steps_writer_.status();
  }

    // Write and flush step offsets.
    if (const Datum step_offsets_datum = Encode(step_offsets);
        !step_offsets_writer_.WriteRecord(step_offsets_datum) ||
        !step_offsets_writer_.Flush(riegeli::FlushType::kFromMachine)) {
      VLOG(0) << "step_offsets_writer_.status(): "
              << step_offsets_writer_.status();
    }

    // Episode index.
    if (num_episodes > 1) {
      // Flush episodic metadata.
      if (!episode_metadata_writer_.Flush(riegeli::FlushType::kFromMachine)) {
        VLOG(0) << "episode_metadata_writer_.status(): "
                << episode_metadata_writer_.status();
      }

      // Create episodic index, then write it and flush it.
      xt::xarray<int64_t> episode_offsets =
          xt::adapt(episode_offsets_, {num_episodes - 1});
      xt::xarray<int64_t> episode_index = xt::stack(
          xt::xtuple(
              xt::view(episode_starts, xt::range(xt::placeholders::_, -1)),
              episode_offsets),
          1);
      if (const Datum episode_index_datum = Encode(episode_index);
          !episode_index_writer_.WriteRecord(episode_index_datum) ||
          !episode_index_writer_.Flush(riegeli::FlushType::kFromMachine)) {
        VLOG(0) << "episode_index_writer_.status(): "
                << episode_index_writer_.status();
      }

      // Clear N-1 episode starts and all episode offsets.
      const int64_t latest_episode_start = *episode_starts_.rbegin();
      episode_starts_.clear();
      episode_starts_.push_back(latest_episode_start);
      episode_offsets_.clear();
    }
  step_offsets_.clear();

  VLOG(2) << "~Flush()";
}

void RiegeliShardWriter::Close() {
  Flush();

  // Write any remaining episode metadata and episode start.
  WriteLastEpisodeIndex(&episode_starts_, &episode_metadata_,
                        &episode_metadata_writer_, &episode_index_writer_);

  // Close everything.
  const bool steps_close_status = steps_writer_.Close();
  VLOG(1) << "steps_close_status: " << steps_close_status;
    const bool step_offsets_close_status = step_offsets_writer_.Close();
    VLOG(1) << "step_offsets_close_status: " << step_offsets_close_status;
    const bool episodes_close_status = episode_metadata_writer_.Close();
    VLOG(1) << "episodes_close_status: " << episodes_close_status;
    const bool episode_offsets_close_status = episode_index_writer_.Close();
    VLOG(1) << "episode_offsets_close_status: " << episode_offsets_close_status;
}

bool RiegeliShardWriter::AddStep(const google::protobuf::Message& data,
                                 bool is_new_episode) {
  if (is_new_episode) {
    // Write episode metadata for the just-finished episode.
    int64_t episode_offset = -1;
    if (episode_metadata_) {
      if (!episode_metadata_writer_.WriteRecord(*episode_metadata_)) {
        VLOG(0) << "episode_metadata_writer_.status(): "
                << episode_metadata_writer_.status();
      }
      episode_offset = episode_metadata_writer_.LastPos().get().numeric();
      episode_metadata_.reset();
    }
    if (!episode_starts_.empty()) episode_offsets_.push_back(episode_offset);

    const int start = step_offsets_.size() + num_steps_at_flush_;
    episode_starts_.push_back(start);
  }

  if (!steps_writer_.is_open()) {
    LOG(ERROR)
        << "steps_writer_ has not been opened yet! Please ensure that "
           "`Init()` has been called and that the first step of the episode "
           "has been added.";
    return false;
  }

  if (!steps_writer_.WriteRecord(data)) {
    LOG(ERROR) << "Failed to write record! steps_writer_.status(): "
               << steps_writer_.status()
               << ". `data`: " << data.Utf8DebugString();
    return false;
  }

  step_offsets_.push_back(steps_writer_.LastPos().get().numeric());
  return true;
}

void RiegeliShardWriter::SetEpisodeMetadata(const Data& data) {
  if (!episode_metadata_writer_.is_open()) {
    VLOG(0)
        << "Trying to set episodic metadata without an episode metadata file. "
           "The payload will be ignored. data: "
        << data.DebugString();
    return;
  }
  if (episode_starts_.empty()) {
    VLOG(0) << "Trying to set episodic metadata without an episode on the way. "
               "Ensure that `AddStep(..., /*is_new_episode=*/true)` was called "
               "before `SetEpisodeMetadata().`";
    return;
  }

  episode_metadata_ = data;
}

}  // namespace envlogger
