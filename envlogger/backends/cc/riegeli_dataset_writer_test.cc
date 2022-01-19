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
#include "envlogger/platform/riegeli_file_reader.h"
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
using ::testing::SizeIs;
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

TEST(RiegeliDatasetWriterTest, EmptyTagDir) {
  const Data metadata;
  const int max_episodes_per_shard = 1000;
  RiegeliDatasetWriter writer;
  EXPECT_TRUE(absl::IsNotFound(
      writer.Init(/*data_dir=*/"", metadata, max_episodes_per_shard)));
}

TEST(RiegeliDatasetWriterTest, NonPositiveMaxEpisodesPerShard) {
  const std::string data_dir =
      file::JoinPath(getenv("TEST_TMPDIR"), "single_shard");
  const Data metadata;
  const int max_episodes_per_shard = -1;

  ENVLOGGER_EXPECT_OK(file::CreateDir(data_dir));
  {
    RiegeliDatasetWriter writer;
    ENVLOGGER_EXPECT_OK(
        writer.Init(data_dir, metadata, max_episodes_per_shard));
    // Write "lots" of episodes.
    for (int i = 0; i < 1000; ++i) {
      Data data;
      data.mutable_datum()->mutable_values()->add_float_values(i);
      writer.AddStep(data, /*is_new_episode=*/true);
    }
    writer.Flush();
  }

  // Check that there's only a single shard.
  ENVLOGGER_ASSERT_OK_AND_ASSIGN(const auto match_results,
                                 file::GetSubdirectories(data_dir));
  EXPECT_THAT(match_results, SizeIs(1));

  ENVLOGGER_EXPECT_OK(file::RecursivelyDelete(data_dir));
}

TEST(RiegeliDatasetWriterTest, PositiveMaxEpisodesPerShard) {
  const std::string data_dir =
      file::JoinPath(getenv("TEST_TMPDIR"), "multi_shard");
  const Data metadata;
  const int max_episodes_per_shard = 5;

  ENVLOGGER_EXPECT_OK(file::CreateDir(data_dir));
  {
    RiegeliDatasetWriter writer;
    ENVLOGGER_EXPECT_OK(
        writer.Init(data_dir, metadata, max_episodes_per_shard));
    // Write "lots" of episodes.
    for (int i = 0; i < 1000; ++i) {
      Data data;
      data.mutable_datum()->mutable_values()->add_float_values(i);
      writer.AddStep(data, /*is_new_episode=*/true);
    }
    writer.Flush();
  }

  // Check that there are many shards.
  ENVLOGGER_ASSERT_OK_AND_ASSIGN(const auto match_results,
                                 file::GetSubdirectories(data_dir));
  const int expected_num_shards = 1000 / max_episodes_per_shard;
  EXPECT_THAT(match_results, SizeIs(expected_num_shards))
      << "Expecting 1000/max_episodes_per_shard == 200 shards.";

  ENVLOGGER_EXPECT_OK(file::RecursivelyDelete(data_dir));
}

TEST(RiegeliDatasetWriterTest, EpisodeMetadata) {
  const std::string data_dir =
      file::JoinPath(getenv("TEST_TMPDIR"), "single_shard");
  const Data metadata;
  const int max_episodes_per_shard = -1;  // Single shard for entire trajectory.

  ENVLOGGER_EXPECT_OK(file::CreateDir(data_dir));
  const Data episode0_metadata =
      ParseTextProtoOrDie(R"pb(datum: { values: { int64_values: 7 } })pb");
  const Data episode2_metadata =
      ParseTextProtoOrDie(R"pb(datum: { values: { int64_values: 8 } })pb");
  {
    RiegeliDatasetWriter writer;
    ENVLOGGER_EXPECT_OK(
        writer.Init(data_dir, metadata, max_episodes_per_shard));
    // Write 30 identical steps and 3 episodes of 10 steps each.
    // The metadata we write is:
    // episode 0: int64(7)
    // episode 1: nothing (should be left at its default nullopt).
    // episode 2: int64(8).
    const Data step_data =
        ParseTextProtoOrDie(R"pb(datum: { values: { int64_values: 321 } })pb");
    // Setting episodic metadata before an episode is available should be
    // ignored.
    writer.SetEpisodeMetadata(ParseTextProtoOrDie(
        R"pb(datum: { values: { float_values: 3.14 } })pb"));
    for (int i = 0; i < 30; ++i) {
      if (i < 10) {  // Call during all steps of episode 0.
        writer.SetEpisodeMetadata(episode0_metadata);
      } else if (i > 20) {  // Set multiple times during episode 2.
        writer.SetEpisodeMetadata(episode2_metadata);
      }
      writer.AddStep(step_data, /*is_new_episode=*/i % 10 == 0);
    }
    writer.Flush();
  }

  // Get single shard.
  ENVLOGGER_ASSERT_OK_AND_ASSIGN(const auto match_results,
                                 file::GetSubdirectories(data_dir));
  EXPECT_THAT(match_results, SizeIs(1));
  const auto& shard_dir = match_results[0];

  // Open episode metadata file and read contents into vector.
  const std::string episode_metadata_riegeli_file =
      file::JoinPath(shard_dir, internal::kEpisodeMetadataFilename);
  riegeli::RecordReader episode_metadata_reader(
      RiegeliFileReader<>(episode_metadata_riegeli_file, "r"));
  ENVLOGGER_EXPECT_OK(episode_metadata_reader.status());
  std::vector<Data> actual_episode_metadata;
  Data value;
  while (episode_metadata_reader.ReadRecord(value)) {
    actual_episode_metadata.push_back(value);
  }
  EXPECT_THAT(actual_episode_metadata,
              ElementsAre(EqualsProto(episode0_metadata),
                          // Second episode1 should have no metadata here.
                          EqualsProto(episode2_metadata)));

  ENVLOGGER_EXPECT_OK(file::RecursivelyDelete(data_dir));
}

// This test ensures that `writer_options` are not ignored and have an effect in
// the output (i.e. the file sizes of the output files are different).
TEST(RiegeliDatasetWriterTest, WriterOptionsAreForwarded) {
  // Write the same trajectory using two different writer options.
  auto write_trajectory = [](absl::string_view output_dir,
                             absl::string_view writer_options) {
    const Data metadata;
    const int max_episodes_per_shard = -1;

    ENVLOGGER_EXPECT_OK(file::CreateDir(output_dir));
    {
      RiegeliDatasetWriter writer;
      ENVLOGGER_EXPECT_OK(writer.Init(std::string(output_dir), metadata,
                                      max_episodes_per_shard,
                                      std::string(writer_options)));
      // Write some steps.
      for (int i = 0; i < 100; ++i) {
        Data data;
        data.mutable_datum()->mutable_values()->add_float_values(i);
        writer.AddStep(data, /*is_new_episode=*/true);
      }
      writer.Flush();
    }
  };
  const std::string writer_options1 =
      file::JoinPath(getenv("TEST_TMPDIR"), "writer_options1");
  const std::string writer_options2 =
      file::JoinPath(getenv("TEST_TMPDIR"), "writer_options2");
  write_trajectory(writer_options1, "brotli:9,chunk_size:10M");
  write_trajectory(writer_options2, "brotli:1,chunk_size:16K");

  // Read the sizes of kStepsFilename from both trajectories.
  ENVLOGGER_ASSERT_OK_AND_ASSIGN(const auto writer_options1_results,
                                 file::GetSubdirectories(writer_options1));
  ENVLOGGER_ASSERT_OK_AND_ASSIGN(const auto writer_options2_results,
                                 file::GetSubdirectories(writer_options2));
  EXPECT_THAT(writer_options1_results, SizeIs(1));  // Expecting a single shard.
  EXPECT_THAT(writer_options2_results, SizeIs(1));  // Expecting a single shard.
  const std::string writer_options1_fname =
      file::JoinPath(writer_options1_results[0], internal::kStepsFilename);
  const std::string writer_options2_fname =
      file::JoinPath(writer_options2_results[0], internal::kStepsFilename);
  const absl::StatusOr<int64_t> writer_options1_size =
      file::GetSize(writer_options1_fname);
  const absl::StatusOr<int64_t> writer_options2_size =
      file::GetSize(writer_options2_fname);

  // Verify that the sizes exist and that they're different.
  EXPECT_THAT(writer_options1_size.ok(), IsTrue());
  EXPECT_THAT(writer_options2_size.ok(), IsTrue());
  EXPECT_THAT(*writer_options1_size, Not(Eq(*writer_options2_size)));

  ENVLOGGER_EXPECT_OK(file::RecursivelyDelete(writer_options1));
  ENVLOGGER_EXPECT_OK(file::RecursivelyDelete(writer_options2));
}

}  // namespace
}  // namespace envlogger
