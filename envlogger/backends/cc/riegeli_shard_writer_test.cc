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
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/flags/flag.h"
#include "envlogger/backends/cc/episode_info.h"
#include "envlogger/converters/xtensor_codec.h"
#include "envlogger/platform/filesystem.h"
#include "envlogger/platform/parse_text_proto.h"
#include "envlogger/platform/proto_testutil.h"
#include "envlogger/platform/riegeli_file_reader.h"
#include "envlogger/platform/test_macros.h"
#include "envlogger/proto/storage.pb.h"
#include "riegeli/records/record_reader.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xaxis_iterator.hpp"

namespace envlogger {
namespace {

using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::FloatEq;
using ::testing::IsTrue;
using ::testing::Key;
using ::testing::Lt;
using ::testing::SizeIs;
using ::testing::Value;

// A simple matcher to compare the output of TrajectoryReader::Episode().
MATCHER_P2(EqualsEpisode, start_index, num_steps, "") {
  return Value(arg.start, start_index) && Value(arg.num_steps, num_steps);
}

TEST(RiegeliShardWriterTest, KStepsIndex) {
  const std::string steps_filename =
      file::JoinPath(getenv("TEST_TMPDIR"), "my_steps.riegeli");
  const std::string step_offsets_filename = file::JoinPath(
      getenv("TEST_TMPDIR"), "my_step_offsets.riegeli");
  const std::string episode_metadata_filename = file::JoinPath(
      getenv("TEST_TMPDIR"), "my_episode_metadata.riegeli");
  const std::string episode_index_filename = file::JoinPath(
      getenv("TEST_TMPDIR"), "my_episode_index.riegeli");
  // Write some predictable data.
  {
    RiegeliShardWriter writer;
    ENVLOGGER_EXPECT_OK(
        writer.Init(/*steps_filepath=*/steps_filename,
                    /*step_offsets_filepath=*/step_offsets_filename,
                    /*episode_metadata_filepath=*/episode_metadata_filename,
                    /*episode_index_filepath=*/episode_index_filename,
                    /*writer_options=*/"transpose,brotli:6,chunk_size:1M"));
    writer.AddStep(
        ParseTextOrDie<Data>(R"pb(datum: { values: { float_values: 1.0 } })pb"),
        /*is_new_episode=*/true);
    writer.AddStep(ParseTextOrDie<Data>(
        R"pb(datum: { values: { float_values: 2.0 } })pb"));
    writer.AddStep(
        ParseTextOrDie<Data>(R"pb(datum: { values: { float_values: 3.0 } })pb"),
        /*is_new_episode=*/true);
    writer.SetEpisodeMetadata(ParseTextOrDie<Data>(
        R"pb(datum: { values: { int32_values: 12345 } })pb"));
    writer.AddStep(
        ParseTextOrDie<Data>(R"pb(datum: { values: { float_values: 4.0 } })pb"),
        /*is_new_episode=*/true);
    writer.SetEpisodeMetadata(ParseTextOrDie<Data>(
        R"pb(datum: { values: { int32_values: 54321 } })pb"));
    writer.AddStep(ParseTextOrDie<Data>(
        R"pb(datum: { values: { float_values: 5.0 } })pb"));
    writer.Flush();
  }

  // Check step data.
  std::vector<float> steps;
  riegeli::RecordReader steps_reader{RiegeliFileReader(steps_filename)};
  Data step_data;
  while (steps_reader.ReadRecord(step_data)) {
    const auto step_decoded = Decode(step_data.datum());
    EXPECT_THAT(step_decoded.has_value(), IsTrue());
    EXPECT_THAT(std::holds_alternative<xt::xarray<float>>(*step_decoded),
                IsTrue());
    const xt::xarray<float>& step_chunk =
        std::get<xt::xarray<float>>(*step_decoded);
    for (const float f : step_chunk) steps.push_back(f);
  }
  EXPECT_THAT(steps, ElementsAre(FloatEq(1.0f), FloatEq(2.0f), FloatEq(3.0f),
                                 FloatEq(4.0f), FloatEq(5.0f)));

  // Check step offsets.
  riegeli::RecordReader step_offsets_reader{
      RiegeliFileReader(step_offsets_filename)};
  Datum step_offsets_datum;
  EXPECT_THAT(step_offsets_reader.ReadRecord(step_offsets_datum), IsTrue);
  const auto step_offsets_decoded = Decode(step_offsets_datum);
  EXPECT_THAT(step_offsets_decoded.has_value(), IsTrue());
  EXPECT_THAT(
      std::holds_alternative<xt::xarray<int64_t>>(*step_offsets_decoded),
      IsTrue());
  const xt::xarray<int64_t>& step_offsets =
      std::get<xt::xarray<int64_t>>(*step_offsets_decoded);
  EXPECT_THAT(step_offsets, SizeIs(5)) << "Expected 5 steps.";
  // Try to access each step using this offset.
  for (size_t i = 0; i < step_offsets.size(); ++i) {
    steps_reader.Seek(step_offsets(i));
    Data step;
    EXPECT_THAT(steps_reader.ReadRecord(step), IsTrue());
    const auto step_decoded = Decode(step.datum());
    EXPECT_THAT(step_decoded.has_value(), IsTrue());
    EXPECT_THAT(std::holds_alternative<xt::xarray<float>>(*step_decoded),
                IsTrue());
    const xt::xarray<float>& step_chunk =
        std::get<xt::xarray<float>>(*step_decoded);
    EXPECT_THAT(step_chunk, SizeIs(1));
    EXPECT_THAT(step_chunk(0), FloatEq(steps[i]));
  }

  // Check episode metadata.
  std::vector<int32_t> episode_metadata;
  riegeli::RecordReader episode_metadata_reader{
      RiegeliFileReader(episode_metadata_filename)};
  Data episode_data;
  while (episode_metadata_reader.ReadRecord(episode_data)) {
    const auto episode_decoded = Decode(episode_data.datum());
    EXPECT_THAT(episode_decoded.has_value(), IsTrue());
    EXPECT_THAT(std::holds_alternative<xt::xarray<int32_t>>(*episode_decoded),
                IsTrue());
    const xt::xarray<int32_t>& episode_chunk =
        std::get<xt::xarray<int32_t>>(*episode_decoded);
    for (const int32_t i : episode_chunk) episode_metadata.push_back(i);
  }
  EXPECT_THAT(episode_metadata, ElementsAre(12345, 54321));

  // Check episode index.
  std::vector<int64_t> episode_starts;
  std::vector<int64_t> episode_offsets;
  riegeli::RecordReader episode_index_reader{
      RiegeliFileReader(episode_index_filename)};
  Datum episode_index_datum;
  while (episode_index_reader.ReadRecord(episode_index_datum)) {
    const auto episode_index_decoded = Decode(episode_index_datum);
    EXPECT_THAT(episode_index_decoded.has_value(), IsTrue());
    EXPECT_THAT(
        std::holds_alternative<xt::xarray<int64_t>>(*episode_index_decoded),
        IsTrue());
    const auto& partial = std::get<xt::xarray<int64_t>>(*episode_index_decoded);
    for (auto it = xt::axis_begin(partial), end = xt::axis_end(partial);
         it != end; ++it) {
      episode_starts.push_back((*it)(0));
      episode_offsets.push_back((*it)(1));
    }
  }

  // Check episode starts.
  EXPECT_THAT(episode_starts, ElementsAre(0, 2, 3));

  // Use episode offsets to lookup episode metadata.
  std::vector<int32_t> actual_episode_metadata;
  for (const int64_t offset : episode_offsets) {
    if (offset > 0) {
      Data episode_data;
      episode_metadata_reader.Seek(offset);
      episode_metadata_reader.ReadRecord(episode_data);
      const auto episode_data_decoded = Decode(episode_data.datum());
      EXPECT_THAT(episode_data_decoded.has_value(), IsTrue());
      EXPECT_THAT(
          std::holds_alternative<xt::xarray<int32_t>>(*episode_data_decoded),
          IsTrue());
      const auto& partial =
          std::get<xt::xarray<int32_t>>(*episode_data_decoded);
      EXPECT_THAT(partial, SizeIs(1));
      actual_episode_metadata.push_back(partial(0));
    }
  }
  EXPECT_THAT(actual_episode_metadata, ElementsAre(12345, 54321));
}

}  // namespace
}  // namespace envlogger
