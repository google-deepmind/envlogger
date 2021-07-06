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

#include "envlogger/backends/cc/riegeli_shard_reader.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include <cstdint>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include <gmpxx.h>
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


// Converts an arbitrary precision integer to int64.
// Notice that narrowing can definitely occur, so this should only be used for
// values that we know should fit in an int64.
int64_t MpzToInt64(const mpz_class& z) {
  int64_t result = 0;
  mpz_export(/*rop=*/&result, /*countp=*/nullptr, /*order=*/-11,
             /*size=*/sizeof(result), /*endian=*/-1, /*nails=*/0,
             z.get_mpz_t());
  return result;
}

absl::StatusOr<riegeli::RecordReader<RiegeliFileReader<>>> CreateReader(
    absl::string_view filepath) {
  VLOG(1) << "Creating the reader for filepath: " << filepath;
  std::string reader_filepath(filepath);
  riegeli::RecordReader reader(RiegeliFileReader<>(reader_filepath, "r"));
  if (!reader.healthy()) return reader.status();
  return reader;
}

}  // namespace

absl::Status RiegeliShardReader::Init(absl::string_view index_filepath,
                                      absl::string_view trajectories_filepath) {
  if (index_filepath.empty()) {
    return absl::NotFoundError(
        absl::StrCat("Could not find index_filepath==", index_filepath));
  }

  std::vector<int64_t>* steps = &step_offsets_;
  std::vector<int64_t>* episodes = &episode_starts_;
  std::vector<int64_t> episode_offsets;
  const auto steps_visitor = MakeVisitor(
      [steps](const mpz_class& z) { steps->push_back(MpzToInt64(z)); },
      [steps](const xt::xarray<int64_t>& a) {
        steps->insert(std::end(*steps), std::begin(a), std::end(a));
      },
      [](const auto&) { /* catch all overload */ });
  const auto episodes_visitor = MakeVisitor(
      [episodes](const mpz_class& z) { episodes->push_back(MpzToInt64(z)); },
      [episodes](const xt::xarray<int64_t>& a) {
        episodes->insert(std::end(*episodes), std::begin(a), std::end(a));
      },
      [](const auto&) { /* catch all overload */ });
  const auto episodes_offsets_visitor = MakeVisitor(
      [&episode_offsets](const mpz_class& z) {
        episode_offsets.push_back(MpzToInt64(z));
      },
      [&episode_offsets](const xt::xarray<int64_t>& a) {
        episode_offsets.insert(std::end(episode_offsets), std::begin(a),
                               std::end(a));
      },
      [](const auto&) { /* catch all overload */ });

  ENVLOGGER_ASSIGN_OR_RETURN(riegeli::RecordReader index_reader,
                             CreateReader(index_filepath));

  Data value;
  while (index_reader.ReadRecord(value)) {
    DataView view(&value);
    if (view.Type() != Data::kDict) continue;

    // Read step offsets.
    if (const Data* data = view["step_offsets_array"]; data != nullptr) {
      const absl::optional<BasicType> decoded = Decode(data->datum());
      if (decoded) absl::visit(steps_visitor, *decoded);
    } else if (const Data* offsets = view["step_offsets"]; offsets != nullptr) {
      // This is a legacy format that's much slower than the above.
      for (const Data& data : offsets->array().values()) {
        const absl::optional<BasicType> decoded = Decode(data.datum());
        if (decoded) absl::visit(steps_visitor, *decoded);
      }
    }
    // Read episode starts.
    if (const Data* data = view["episode_starts_array"]; data != nullptr) {
      const absl::optional<BasicType> decoded = Decode(data->datum());
      if (decoded) absl::visit(episodes_visitor, *decoded);
    } else if (const Data* data = view["episode_start_indices_array"];
               data != nullptr) {
      // This is a legacy format where the episode elements are stored as file
      // offsets. This is slower than the version with "episode_starts_array".
      const absl::optional<BasicType> decoded = Decode(data->datum());
      if (decoded) absl::visit(episodes_offsets_visitor, *decoded);
    } else if (const Data* indices = view["episode_start_indices"];
               indices != nullptr) {
      // This is another legacy format where the episode elements are stored as
      // file offsets but is even slower than "episode_start_indices_array"
      // because each episode element is store as a separate Datum instead of a
      // dense array.
      for (const Data& data : indices->array().values()) {
        const absl::optional<BasicType> decoded = Decode(data.datum());
        if (decoded) absl::visit(episodes_offsets_visitor, *decoded);
      }
    }
  }

  if (step_offsets_.empty()) {
    return absl::NotFoundError(absl::StrCat("Empty steps in ", index_filepath));
  }

  VLOG(1) << "step_offsets_.size(): " << step_offsets_.size();
  VLOG(1) << "Building episode_starts_...";
  // Note: This is currently O(|steps| + |episodes|). We could do it in
  //       O(|episodes| * log(|steps|)) but it requires more coding and we do
  //       not know if the performance will be better. Depending on the actual
  //       value of |episodes|, the latter could be much worse than 2 linear
  //       passes (e.g. if most episodes have length 1).
  for (auto step = step_offsets_.begin(), episode = episode_offsets.begin();
       step != step_offsets_.end() && episode != episode_offsets.end();
       ++episode) {
    step = std::find_if(step, step_offsets_.end(),
                        [episode](const int64_t step_offset) {
                          return step_offset >= *episode;
                        });
    // Skip episode offsets that are not in the expected range.
    if (step != step_offsets_.end()) {
      episode_starts_.push_back(std::distance(step_offsets_.begin(), step));
    }
  }
  VLOG(1) << "Done building episode_starts_...";

  VLOG(1) << "episode_starts_.size(): " << episode_starts_.size();

  ENVLOGGER_ASSIGN_OR_RETURN(steps_reader_,
                             CreateReader(trajectories_filepath));

  return absl::OkStatus();
}

RiegeliShardReader::~RiegeliShardReader() { Close(); }

absl::Status RiegeliShardReader::Init(
    absl::string_view steps_filepath, absl::string_view step_offsets_filepath,
    absl::string_view episode_metadata_filepath,
    absl::string_view episode_index_filepath) {
  if (step_offsets_filepath.empty()) {
    return absl::NotFoundError(absl::StrCat(
        "Could not find step_offsets_filepath==", step_offsets_filepath));
  }

  std::vector<int64_t>* steps = &step_offsets_;
  std::vector<int64_t>* episode_starts = &episode_starts_;
  std::vector<int64_t>* episode_metadata_offsets = &episode_metadata_offsets_;
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
    const absl::optional<BasicType> step_offsets_decoded = Decode(value);
    if (step_offsets_decoded) absl::visit(steps_visitor, *step_offsets_decoded);
  }

  // Read episode index.
  ENVLOGGER_ASSIGN_OR_RETURN(riegeli::RecordReader episode_index_reader,
                             CreateReader(episode_index_filepath));
  while (episode_index_reader.ReadRecord(value)) {
    const absl::optional<BasicType> episode_index_decoded = Decode(value);
    if (episode_index_decoded)
      absl::visit(episode_index_visitor, *episode_index_decoded);
  }

  if (step_offsets_.empty()) {
    return absl::NotFoundError(
        absl::StrCat("Empty steps in ", step_offsets_filepath));
  }

  VLOG(1) << "step_offsets_.size(): " << step_offsets_.size();
  VLOG(1) << "episode_starts_.size(): " << episode_starts_.size();

  ENVLOGGER_ASSIGN_OR_RETURN(steps_reader_, CreateReader(steps_filepath));
  ENVLOGGER_ASSIGN_OR_RETURN(episode_metadata_reader_,
                             CreateReader(episode_metadata_filepath));

  return absl::OkStatus();
}

int64_t RiegeliShardReader::NumSteps() const { return step_offsets_.size(); }

int64_t RiegeliShardReader::NumEpisodes() const {
  return episode_starts_.size();
}

absl::optional<EpisodeInfo> RiegeliShardReader::Episode(int64_t episode_index,
                                                        bool include_metadata) {
  if (episode_index < 0 ||
      episode_index >= static_cast<int64_t>(episode_starts_.size())) {
    return absl::nullopt;
  }

  const int64_t start_index = episode_starts_[episode_index];
  const int64_t num_steps =
      episode_index + 1 < static_cast<int64_t>(episode_starts_.size())
          ? episode_starts_[episode_index + 1] - start_index
          : step_offsets_.size() - start_index;
  EpisodeInfo episode_info{start_index, num_steps};
  if (include_metadata &&
      episode_metadata_offsets_.size() == episode_starts_.size()) {
    if (const int64_t offset = episode_metadata_offsets_[episode_index];
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
