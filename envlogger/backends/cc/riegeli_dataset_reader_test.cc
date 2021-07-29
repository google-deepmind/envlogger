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

#include "envlogger/backends/cc/riegeli_dataset_reader.h"

#include <algorithm>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "envlogger/backends/cc/episode_info.h"
#include "envlogger/backends/cc/riegeli_shard_writer.h"
#include "envlogger/backends/cc/riegeli_dataset_io_constants.h"
#include "envlogger/platform/filesystem.h"
#include "envlogger/platform/parse_text_proto.h"
#include "envlogger/platform/proto_testutil.h"
#include "envlogger/platform/riegeli_file_writer.h"
#include "envlogger/platform/test_macros.h"
#include "envlogger/proto/storage.pb.h"
#include "riegeli/base/base.h"
#include "riegeli/records/record_reader.h"
#include "riegeli/records/record_writer.h"

namespace envlogger {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::IsTrue;
using ::testing::Not;
using ::testing::Value;

// A simple matcher to compare the output of RiegeliDatasetReader::Episode().
MATCHER_P2(EqualsEpisode, start_index, num_steps, "") {
  return Value(arg.start, start_index) && Value(arg.num_steps, num_steps);
}

// A specification for creating directories with some fictitious data.
struct TimestampDirSpec {
  struct StepSpec {
    // A step-specific payload.
    float payload;
    // Whether this step is the first step of a new episode.
    bool is_new_episode;
    // Episodic metadata.
    absl::optional<Data> episode_metadata;
    // Whether to use a different payload type (other than Data).
    bool use_other_payload_type;
  };

  // The timestamp associated with this directory.
  // This influences the name of the directory.
  absl::Time timestamp;
  // List of steps to be inserted into this timestamp dir.
  std::vector<StepSpec> steps;
};

// Creates timestamp dirs in `data_dir` according to `specs`.
void CreateTimestampDirs(absl::string_view data_dir,
                         const std::vector<TimestampDirSpec>& specs) {
  for (const auto& spec : specs) {
    const std::string timestamp = absl::FormatTime(
        "%Y%m%dT%H%M%E6f", spec.timestamp, absl::UTCTimeZone());
    const std::string timestamp_dir = file::JoinPath(data_dir, timestamp);
    ENVLOGGER_EXPECT_OK(file::CreateDir(timestamp_dir));
    const std::string steps_file =
        file::JoinPath(timestamp_dir, internal::kStepsFilename);
    const std::string step_offsets_file =
        file::JoinPath(timestamp_dir, internal::kStepOffsetsFilename);
    const std::string episode_metadata_file =
        file::JoinPath(timestamp_dir, internal::kEpisodeMetadataFilename);
    const std::string episode_index_file =
        file::JoinPath(timestamp_dir, internal::kEpisodeIndexFilename);
    RiegeliShardWriter writer;
    ENVLOGGER_EXPECT_OK(writer.Init(steps_file, step_offsets_file,
                                    episode_metadata_file, episode_index_file,
                                    "transpose,brotli:6,chunk_size:1M"));
    for (const auto& step : spec.steps) {
      if (step.use_other_payload_type) {
        Datum::Shape::Dim data;
        data.set_size(step.payload);
        writer.AddStep(data, step.is_new_episode);
      } else {
        Data data;
        data.mutable_datum()->mutable_values()->add_float_values(step.payload);
        writer.AddStep(data, step.is_new_episode);
      }
      if (step.episode_metadata) {
        writer.SetEpisodeMetadata(*step.episode_metadata);
      }
    }
    writer.Flush();
  }
}

TEST(DataDirectoryIndex, BadRiegeliProtoFormatFile) {
  const std::string data_dir =
      file::JoinPath(getenv("TEST_TMPDIR"), "my_data_dir");
  ENVLOGGER_EXPECT_OK(file::CreateDir(data_dir));

  // Write metadata and specs in the wrong proto format (Datum instead of Data).
  Datum datum;
  datum.mutable_values()->add_float_values(123.456f);
  {
    riegeli::RecordWriter writer(
        RiegeliFileWriter<>(
            file::JoinPath(data_dir, internal::kMetadataFilename), "w"),
        riegeli::RecordWriterBase::Options().set_transpose(true));
    EXPECT_THAT(writer.WriteRecord(datum), IsTrue());
    writer.Flush(riegeli::FlushType::kFromMachine);
  }

  RiegeliDatasetReader reader;
  EXPECT_TRUE(absl::IsInternal(reader.Init(data_dir)));
  reader.Close();  // (optional) Close the reader and free up its resources.

  ENVLOGGER_EXPECT_OK(file::RecursivelyDelete(data_dir));
}

TEST(DataDirectoryIndex, SimpleMetadataAndSpec) {
  const std::string data_dir =
      file::JoinPath(getenv("TEST_TMPDIR"), "meta_spec_dir");
  ENVLOGGER_EXPECT_OK(file::CreateDir(data_dir));

  // Write metadata and specs.
  Data metadata;
  metadata.mutable_datum()->mutable_values()->add_float_values(123.456f);
  {
    riegeli::RecordWriter writer(
        RiegeliFileWriter<>(
            file::JoinPath(data_dir, internal::kMetadataFilename), "w"),
        riegeli::RecordWriterBase::Options().set_transpose(true));
    EXPECT_THAT(writer.WriteRecord(metadata), IsTrue());
    writer.Flush(riegeli::FlushType::kFromMachine);
  }

  RiegeliDatasetReader reader;
  ENVLOGGER_EXPECT_OK(reader.Init(data_dir));
  const auto actual_metadata = reader.Metadata();
  EXPECT_THAT(actual_metadata, Not(Eq(absl::nullopt)));
  EXPECT_THAT(*actual_metadata, EqualsProto(metadata));

  ENVLOGGER_EXPECT_OK(file::RecursivelyDelete(data_dir));
}

TEST(DataDirectoryIndex, OneShard) {
  const std::string data_dir =
      file::JoinPath(getenv("TEST_TMPDIR"), "my_data_dir");
  ENVLOGGER_EXPECT_OK(file::CreateDir(data_dir));

  // Write metadata and specs.
  Data episode0_metadata = ParseTextProtoOrDie(R"pb(
    datum: { values: { int32_values: 12345 } }
  )pb");
  Data dummy;
  {
    riegeli::RecordWriter writer(
        RiegeliFileWriter<>(
            file::JoinPath(data_dir, internal::kMetadataFilename), "w"),
        riegeli::RecordWriterBase::Options().set_transpose(true));
    EXPECT_THAT(writer.WriteRecord(dummy), IsTrue());
    writer.Flush(riegeli::FlushType::kFromMachine);
  }
  CreateTimestampDirs(data_dir, {{absl::Now() - absl::Minutes(60),
                                  {{1.0f, true},
                                   {2.0f, false},
                                   {3.0f, false, episode0_metadata},
                                   {4.0f, true},
                                   {5.0f, false}}}});
  RiegeliDatasetReader reader;
  ENVLOGGER_EXPECT_OK(reader.Init(data_dir));
  EXPECT_THAT(reader.NumSteps(), Eq(5));
  EXPECT_THAT(reader.NumEpisodes(), Eq(2));
  // Check for some invalid operations.
  EXPECT_THAT(reader.Step(-1), Eq(absl::nullopt));
  EXPECT_THAT(reader.Step(reader.NumSteps() + 1), Eq(absl::nullopt));
  EXPECT_THAT(reader.Episode(-1), Eq(absl::nullopt));
  EXPECT_THAT(reader.Episode(reader.NumEpisodes() + 1), Eq(absl::nullopt));

  // Check steps.
  std::vector<Data> steps;
  for (int i = 0; i < reader.NumSteps(); ++i) {
    const auto step = reader.Step(i);
    EXPECT_THAT(step, Not(Eq(absl::nullopt)));
    steps.push_back(*step);
  }
  EXPECT_THAT(
      steps,
      ElementsAre(EqualsProto("datum: { values: { float_values: 1.0}}"),
                  EqualsProto("datum: { values: { float_values: 2.0}}"),
                  EqualsProto("datum: { values: { float_values: 3.0}}"),
                  EqualsProto("datum: { values: { float_values: 4.0}}"),
                  EqualsProto("datum: { values: { float_values: 5.0}}")));

  // Check episodes.
  std::vector<EpisodeInfo> episodes;
  for (int i = 0; i < reader.NumEpisodes(); ++i) {
    const auto episode = reader.Episode(i, /*include_metadata=*/true);
    EXPECT_THAT(episode, Not(Eq(absl::nullopt)));
    episodes.push_back(*episode);
  }
  EXPECT_THAT(episodes, ElementsAre(EqualsEpisode(0, 3), EqualsEpisode(3, 2)));
  EXPECT_THAT(episodes[0].metadata, Not(Eq(absl::nullopt)));
  EXPECT_THAT(*episodes[0].metadata, EqualsProto(episode0_metadata));

  ENVLOGGER_EXPECT_OK(file::RecursivelyDelete(data_dir));
}

// Same as OneShard, but it stores a payload that's not envlogger::Data to
// ensure that the reader and writer can work with arbitrary proto messages.
TEST(DataDirectoryIndex, OneShardNonDmDataPayload) {
  const std::string data_dir =
      file::JoinPath(getenv("TEST_TMPDIR"), "my_data_dir");
  ENVLOGGER_EXPECT_OK(file::CreateDir(data_dir));

  // Write metadata and specs.
  Data episode0_metadata = ParseTextProtoOrDie(R"pb(
    datum: { values: { int32_values: 12345 } }
  )pb");
  Datum::Shape::Dim dummy = ParseTextProtoOrDie("size: 17");
  {
    riegeli::RecordWriter writer(
        RiegeliFileWriter<>(
            file::JoinPath(data_dir, internal::kMetadataFilename), "w"),
        riegeli::RecordWriterBase::Options().set_transpose(true));
    EXPECT_THAT(writer.WriteRecord(dummy), IsTrue());
    writer.Flush(riegeli::FlushType::kFromMachine);
  }
  const bool use_other_payload_type = true;
  CreateTimestampDirs(
      data_dir, {{absl::Now() - absl::Minutes(60),
                  {{1.0f, true, absl::nullopt, use_other_payload_type},
                   {2.0f, false, absl::nullopt, use_other_payload_type},
                   {3.0f, false, episode0_metadata, use_other_payload_type},
                   {4.0f, true, absl::nullopt, use_other_payload_type},
                   {5.0f, false, absl::nullopt, use_other_payload_type}}}});
  RiegeliDatasetReader reader;
  ENVLOGGER_EXPECT_OK(reader.Init(data_dir));
  EXPECT_THAT(reader.NumSteps(), Eq(5));
  EXPECT_THAT(reader.NumEpisodes(), Eq(2));
  // Check for some invalid operations.
  EXPECT_THAT(reader.Step<Datum::Shape::Dim>(-1), Eq(absl::nullopt));
  EXPECT_THAT(reader.Step(reader.NumSteps() + 1), Eq(absl::nullopt));
  EXPECT_THAT(reader.Episode(-1), Eq(absl::nullopt));
  EXPECT_THAT(reader.Episode(reader.NumEpisodes() + 1), Eq(absl::nullopt));

  // Check steps.
  std::vector<Datum::Shape::Dim> steps;
  for (int i = 0; i < reader.NumSteps(); ++i) {
    const auto step = reader.Step<Datum::Shape::Dim>(i);
    EXPECT_THAT(step, Not(Eq(absl::nullopt)));
    steps.push_back(*step);
  }
  EXPECT_THAT(steps, ElementsAre(EqualsProto("size: 1"), EqualsProto("size: 2"),
                                 EqualsProto("size: 3"), EqualsProto("size: 4"),
                                 EqualsProto("size: 5")));

  // Check episodes.
  std::vector<EpisodeInfo> episodes;
  for (int i = 0; i < reader.NumEpisodes(); ++i) {
    const auto episode = reader.Episode(i, /*include_metadata=*/true);
    EXPECT_THAT(episode, Not(Eq(absl::nullopt)));
    episodes.push_back(*episode);
  }
  EXPECT_THAT(episodes, ElementsAre(EqualsEpisode(0, 3), EqualsEpisode(3, 2)));
  EXPECT_THAT(episodes[0].metadata, Not(Eq(absl::nullopt)));
  EXPECT_THAT(*episodes[0].metadata, EqualsProto(episode0_metadata));

  ENVLOGGER_EXPECT_OK(file::RecursivelyDelete(data_dir));
}

TEST(DataDirectoryIndex, TwoShards) {
  const std::string data_dir =
      file::JoinPath(getenv("TEST_TMPDIR"), "my_data_dir");
  ENVLOGGER_EXPECT_OK(file::CreateDir(data_dir));

  // Write metadata and specs.
  Data dummy;
  {
    riegeli::RecordWriter writer(
        RiegeliFileWriter<>(
            file::JoinPath(data_dir, internal::kMetadataFilename), "w"),
        riegeli::RecordWriterBase::Options().set_transpose(true));
    EXPECT_THAT(writer.WriteRecord(dummy), IsTrue());
    writer.Flush(riegeli::FlushType::kFromMachine);
  }
  CreateTimestampDirs(
      data_dir,
      {{absl::Now() - absl::Minutes(60),
        {{1.0f, true},
         {2.0f, false},
         {3.0f, false},
         {4.0f, true},
         {5.0f, false}}},
       {absl::Now() - absl::Minutes(50),
        {{60.0f, true}, {70.0f, true}, {80.0f, false}, {90.0f, false}}}});
  RiegeliDatasetReader reader;
  ENVLOGGER_EXPECT_OK(reader.Init(data_dir));
  EXPECT_THAT(reader.NumSteps(), Eq(9));
  EXPECT_THAT(reader.NumEpisodes(), Eq(4));

  // Check steps.
  std::vector<Data> steps;
  for (int i = 0; i < reader.NumSteps(); ++i) {
    const auto step = reader.Step(i);
    EXPECT_THAT(step, Not(Eq(absl::nullopt)));
    steps.push_back(*step);
  }
  EXPECT_THAT(
      steps,
      ElementsAre(EqualsProto("datum: { values: { float_values: 1.0}}"),
                  EqualsProto("datum: { values: { float_values: 2.0}}"),
                  EqualsProto("datum: { values: { float_values: 3.0}}"),
                  EqualsProto("datum: { values: { float_values: 4.0}}"),
                  EqualsProto("datum: { values: { float_values: 5.0}}"),
                  EqualsProto("datum: { values: { float_values: 60.0}}"),
                  EqualsProto("datum: { values: { float_values: 70.0}}"),
                  EqualsProto("datum: { values: { float_values: 80.0}}"),
                  EqualsProto("datum: { values: { float_values: 90.0}}")));

  // Check episodes.
  std::vector<EpisodeInfo> episodes;
  for (int i = 0; i < reader.NumEpisodes(); ++i) {
    const auto episode = reader.Episode(i);
    EXPECT_THAT(episode, Not(Eq(absl::nullopt)));
    episodes.push_back(*episode);
  }
  EXPECT_THAT(episodes, ElementsAre(EqualsEpisode(0, 3), EqualsEpisode(3, 2),
                                    EqualsEpisode(5, 1), EqualsEpisode(6, 3)));

  ENVLOGGER_EXPECT_OK(file::RecursivelyDelete(data_dir));
}

}  // namespace
}  // namespace envlogger
