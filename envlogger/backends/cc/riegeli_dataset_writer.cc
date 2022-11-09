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

#include "envlogger/backends/cc/riegeli_dataset_writer.h"

#include <algorithm>
#include <cstdint>
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
#include "envlogger/backends/cc/episode_info.h"
#include "envlogger/backends/cc/riegeli_dataset_io_constants.h"
#include "envlogger/backends/cc/riegeli_shard_reader.h"
#include "envlogger/backends/cc/riegeli_shard_writer.h"
#include "envlogger/platform/filesystem.h"
#include "envlogger/platform/riegeli_file_writer.h"
#include "envlogger/platform/status_macros.h"
#include "envlogger/proto/storage.pb.h"
#include "riegeli/base/types.h"
#include "riegeli/records/record_reader.h"
#include "riegeli/records/record_writer.h"

namespace envlogger {

namespace {

// Writes `data` as a single record in the Riegeli file pointed by `filepath`.
absl::Status WriteSingleRiegeliRecord(const absl::string_view filepath,
                                      const Data& data) {
  riegeli::RecordWriter writer(
      RiegeliFileWriter(filepath),
      riegeli::RecordWriterBase::Options().set_transpose(true));
  if (!writer.WriteRecord(data)) return writer.status();
  if (!writer.Flush(riegeli::FlushType::kFromMachine)) return writer.status();
  if (!writer.Close()) return writer.status();

  return absl::OkStatus();
}

std::string NewTimestampDirName(absl::Time time) {
  return absl::FormatTime("%Y%m%dT%H%M%S%E6f", time, absl::UTCTimeZone());
}

absl::Status CreateRiegeliShardWriter(absl::string_view data_dir,
                                      absl::string_view writer_options,
                                      RiegeliShardWriter* writer) {
  const std::string dirname = NewTimestampDirName(absl::Now());
  const std::string timestamp_dir = file::JoinPath(data_dir, dirname);
  ENVLOGGER_RETURN_IF_ERROR(file::CreateDir(timestamp_dir));
  writer->Flush();  // Flush before creating a new one.
  ENVLOGGER_RETURN_IF_ERROR(writer->Init(
      /*steps_filepath=*/file::JoinPath(timestamp_dir,
                                        internal::kStepsFilename),
      /*step_offsets_filepath=*/
      file::JoinPath(timestamp_dir, internal::kStepOffsetsFilename),
      /*episode_metadata_filepath=*/
      file::JoinPath(timestamp_dir, internal::kEpisodeMetadataFilename),
      /*episode_index_filepath=*/
      file::JoinPath(timestamp_dir, internal::kEpisodeIndexFilename),
      writer_options));
  return absl::OkStatus();
}

}  // namespace

absl::Status RiegeliDatasetWriter::Init(std::string data_dir,
                                        const Data& metadata,
                                        int64_t max_episodes_per_shard,
                                        std::string writer_options,
                                        int episode_counter) {
  if (data_dir.empty()) return absl::NotFoundError("Empty data_dir.");

  data_dir_ = data_dir;
  writer_options_ = writer_options;
  const std::string metadata_filepath =
      file::JoinPath(data_dir, internal::kMetadataFilename);
  if (!file::GetSize(metadata_filepath).ok()) {
    // If metadata does not yet exist, write it.
    ENVLOGGER_RETURN_IF_ERROR(WriteSingleRiegeliRecord(
        /*filepath=*/metadata_filepath, /*data=*/metadata));
  }
  max_episodes_per_shard_ = max_episodes_per_shard;
  if (max_episodes_per_shard_ <= 0) {
    ENVLOGGER_RETURN_IF_ERROR(
        CreateRiegeliShardWriter(data_dir, writer_options_, &writer_));
  }
  episode_counter_ = episode_counter;

  return absl::OkStatus();
}

bool RiegeliDatasetWriter::AddStep(const google::protobuf::Message& data,
                                   bool is_new_episode) {
  if (is_new_episode) {
    if (max_episodes_per_shard_ > 0 &&
        episode_counter_++ % max_episodes_per_shard_ == 0) {
      ENVLOGGER_CHECK_OK(
          CreateRiegeliShardWriter(data_dir_, writer_options_, &writer_));
    }
  }
  return writer_.AddStep(data, is_new_episode);
}

void RiegeliDatasetWriter::SetEpisodeMetadata(const Data& data) {
  writer_.SetEpisodeMetadata(data);
}

void RiegeliDatasetWriter::Flush() { writer_.Flush(); }

void RiegeliDatasetWriter::Close() { writer_.Close(); }

}  // namespace envlogger
