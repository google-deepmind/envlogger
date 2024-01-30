// Copyright 2024 DeepMind Technologies Limited..
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

#include <cstdint>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "envlogger/backends/cc/episode_info.h"
#include "envlogger/converters/xtensor_codec.h"
#include "envlogger/platform/filesystem.h"
#include "envlogger/platform/parse_text_proto.h"
#include "envlogger/platform/proto_testutil.h"
#include "envlogger/platform/riegeli_file_writer.h"
#include "envlogger/platform/test_macros.h"
#include "envlogger/proto/storage.pb.h"
#include "riegeli/base/types.h"
#include "riegeli/records/record_position.h"
#include "riegeli/records/record_writer.h"
#include "xtensor/xadapt.hpp"

namespace envlogger {
namespace {

using ::testing::Eq;
using ::testing::IsTrue;
using ::testing::Not;
using ::testing::Value;

// A simple matcher to compare the output of RiegeliShardReader::Episode().
MATCHER_P2(EqualsEpisode, start_index, num_steps, "") {
  return Value(arg.start, start_index) && Value(arg.num_steps, num_steps);
}

TEST(RiegeliShardReaderTest, EmptyIndexFilename) {
  RiegeliShardReader reader;
  EXPECT_TRUE(absl::IsNotFound(reader.Init(
      /*steps_filepath=*/"", /*step_offsets_filepath=*/"",
      /*episode_metadata_filepath=*/"", /*episode_index_filepath=*/"")));
  EXPECT_THAT(reader.NumSteps(), Eq(0));
  EXPECT_THAT(reader.NumEpisodes(), Eq(0));
  EXPECT_THAT(reader.Step(0), Eq(std::nullopt));
  EXPECT_THAT(reader.Episode(0), Eq(std::nullopt));
  reader.Close();  // (optional) Close the reader and free up its resources.
}

TEST(RiegeliShardReaderTest, NonEmptySingleEpisode) {
  const std::string steps_filename =
      file::JoinPath(getenv("TEST_TMPDIR"), "steps.riegeli");
  const std::string step_offsets_filename =
      file::JoinPath(getenv("TEST_TMPDIR"), "step_offsets.riegeli");
  const std::string episode_metadata_filename = file::JoinPath(
      getenv("TEST_TMPDIR"), "episode_metadata.riegeli");
  const std::string episode_index_filename =
      file::JoinPath(getenv("TEST_TMPDIR"), "episode_index.riegeli");

  // Create and write predictable data.
  const std::vector<Data> expected_steps = {
      ParseTextOrDie<Data>(R"pb(datum: { values: { float_values: 10.0 } })pb"),
      ParseTextOrDie<Data>(R"pb(datum: { values: { float_values: 11.0 } })pb"),
      ParseTextOrDie<Data>(R"pb(datum: { values: { float_values: 12.0 } })pb"),
      ParseTextOrDie<Data>(R"pb(datum: { values: { float_values: 13.0 } })pb"),
      ParseTextOrDie<Data>(R"pb(datum: { values: { float_values: 14.0 } })pb"),
      ParseTextOrDie<Data>(R"pb(datum: { values: { float_values: 15.0 } })pb"),
      ParseTextOrDie<Data>(R"pb(datum: { values: { float_values: 16.0 } })pb")};
  const Data expected_episode_metadata =
      ParseTextProtoOrDie(R"pb(datum: { values: { int32_values: 12345 } })pb");
  std::vector<int64_t> expected_step_offsets;
  {
    // Write steps and record their riegeli offsets.
    riegeli::RecordWriter steps_writer{RiegeliFileWriter(steps_filename)};
    ENVLOGGER_EXPECT_OK(steps_writer.status());

    for (const Data& data : expected_steps) {
      EXPECT_THAT(steps_writer.WriteRecord(data), IsTrue());
      expected_step_offsets.push_back(steps_writer.LastPos().get().numeric());
    }

    riegeli::RecordWriter step_offsets_writer{
        RiegeliFileWriter(step_offsets_filename)};
    ENVLOGGER_EXPECT_OK(step_offsets_writer.status());
    xt::xarray<int64_t> step_offsets =
        xt::adapt(expected_step_offsets, {expected_steps.size()});
    const Datum step_offsets_datum = Encode(step_offsets);
    EXPECT_THAT(step_offsets_writer.WriteRecord(step_offsets_datum), IsTrue());

    // Write episode metadata and record their riegeli offsets.
    riegeli::RecordWriter episode_metadata_writer{
        RiegeliFileWriter(episode_metadata_filename)};
    ENVLOGGER_EXPECT_OK(episode_metadata_writer.status());
    EXPECT_THAT(episode_metadata_writer.WriteRecord(expected_episode_metadata),
                IsTrue());

    xt::xarray<int64_t> episode_index = {
        {0, static_cast<int64_t>(
                episode_metadata_writer.LastPos().get().numeric())}};
    riegeli::RecordWriter episode_index_writer{
        RiegeliFileWriter(episode_index_filename)};
    ENVLOGGER_EXPECT_OK(episode_index_writer.status());
    const Datum episode_index_datum = Encode(episode_index);
    EXPECT_THAT(episode_index_writer.WriteRecord(episode_index_datum),
                IsTrue());

    EXPECT_THAT(steps_writer.Flush(riegeli::FlushType::kFromMachine), IsTrue());
    EXPECT_THAT(steps_writer.Close(), IsTrue());

    EXPECT_THAT(step_offsets_writer.Flush(riegeli::FlushType::kFromMachine),
                IsTrue());
    EXPECT_THAT(step_offsets_writer.Close(), IsTrue());

    EXPECT_THAT(episode_metadata_writer.Flush(riegeli::FlushType::kFromMachine),
                IsTrue());
    EXPECT_THAT(episode_metadata_writer.Close(), IsTrue());

    EXPECT_THAT(episode_index_writer.Flush(riegeli::FlushType::kFromMachine),
                IsTrue());
    EXPECT_THAT(episode_index_writer.Close(), IsTrue());
  }

  RiegeliShardReader reader;
  ENVLOGGER_EXPECT_OK(
      reader.Init(steps_filename, step_offsets_filename,
                  /*episode_metadata_filepath=*/episode_metadata_filename,
                  /*episode_index_filepath=*/episode_index_filename));
  EXPECT_THAT(reader.NumSteps(), Eq(7));
  EXPECT_THAT(reader.NumEpisodes(), Eq(1));

  // Check steps.
  for (int i = 0; i < reader.NumSteps(); ++i) {
    const auto step = reader.Step(i);
    EXPECT_THAT(step, Not(Eq(std::nullopt)));
    EXPECT_THAT(*step, EqualsProto(expected_steps[i]));
  }
  EXPECT_EQ(reader.NumSteps(), expected_steps.size());

  // Check episodes.

  // If `include_metadata==false`, we expect the `metadata` field to be empty.
  const auto episode0_no_metadata_opt =
      reader.Episode(0, /*include_metadata=*/false);
  EXPECT_THAT(episode0_no_metadata_opt, Not(Eq(std::nullopt)));
  EXPECT_THAT(*episode0_no_metadata_opt, EqualsEpisode(0, 7));
  EXPECT_THAT(episode0_no_metadata_opt->metadata, Eq(std::nullopt));

  // If `include_metadata==true`, we expect the `metadata` field to be filled
  // with the data we wrote above.
  const auto episode0_opt = reader.Episode(0, /*include_metadata=*/true);
  EXPECT_THAT(episode0_opt, Not(Eq(std::nullopt)));
  EXPECT_THAT(*episode0_opt, EqualsEpisode(0, 7));
  EXPECT_THAT(episode0_opt->metadata, Not(Eq(std::nullopt)));
  EXPECT_THAT(*episode0_opt->metadata, EqualsProto(expected_episode_metadata));
}

}  // namespace
}  // namespace envlogger
