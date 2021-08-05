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

#include <iterator>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"
#include "absl/flags/flag.h"
#include "absl/random/random.h"
#include "absl/types/optional.h"
#include "envlogger/backends/cc/episode_info.h"
#include "envlogger/backends/cc/riegeli_shard_reader.h"
#include "envlogger/backends/cc/riegeli_shard_writer.h"
#include "envlogger/platform/filesystem.h"
#include "envlogger/platform/parse_text_proto.h"
#include "envlogger/platform/proto_testutil.h"
#include "envlogger/platform/test_macros.h"
#include "envlogger/proto/storage.pb.h"

namespace envlogger {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Not;
using ::testing::Value;

// A simple matcher to compare the output of TrajectoryReader::Episode().
MATCHER_P2(EqualsEpisode, start_index, num_steps, "") {
  return Value(arg.start, start_index) && Value(arg.num_steps, num_steps);
}

TEST(RiegeliShardIoTest, NonEmptyMultipleEpisodes) {
  const std::string steps_filename =
      file::JoinPath(getenv("TEST_TMPDIR"), "some_steps.riegeli");
  const std::string step_offsets_filename =
      file::JoinPath(getenv("TEST_TMPDIR"), "step_offsets.riegeli");
  const std::string episode_metadata_filename = file::JoinPath(
      getenv("TEST_TMPDIR"), "some_episodic_metadata.riegeli");
  const std::string episode_index_filename = file::JoinPath(
      getenv("TEST_TMPDIR"), "episodic_index.riegeli");
  {
    RiegeliShardWriter writer;
    ENVLOGGER_EXPECT_OK(writer.Init(
        steps_filename, step_offsets_filename, episode_metadata_filename,
        episode_index_filename, "transpose,brotli:6,chunk_size:1M"));
    writer.AddStep(ParseTextOrDie<Data>(R"PROTO(datum: {
                                                  values: { float_values: 10.0 }
                                                })PROTO"),
                   /*is_new_episode=*/true);
    writer.AddStep(ParseTextOrDie<Data>(
        R"PROTO(datum: { values: { float_values: 11.0 } })PROTO"));
    writer.AddStep(ParseTextOrDie<Data>(
        R"PROTO(datum: { values: { float_values: 12.0 } })PROTO"));
    writer.AddStep(ParseTextOrDie<Data>(R"PROTO(datum: {
                                                  values: { float_values: 13.0 }
                                                })PROTO"),
                   /*is_new_episode=*/true);
    writer.AddStep(ParseTextOrDie<Data>(
        R"PROTO(datum: { values: { float_values: 14.0 } })PROTO"));
    writer.AddStep(ParseTextOrDie<Data>(
        R"PROTO(datum: { values: { float_values: 15.0 } })PROTO"));
    writer.AddStep(ParseTextOrDie<Data>(R"PROTO(datum: {
                                                  values: { float_values: 16.0 }
                                                })PROTO"),
                   /*is_new_episode=*/true);
    writer.Flush();
  }

  RiegeliShardReader reader;
  ENVLOGGER_EXPECT_OK(reader.Init(steps_filename, step_offsets_filename,
                                  episode_metadata_filename,
                                  episode_index_filename));
  EXPECT_THAT(reader.NumSteps(), Eq(7));
  EXPECT_THAT(reader.NumEpisodes(), Eq(3));

  // Check steps.
  std::vector<Data> steps;
  for (int i = 0; i < reader.NumSteps(); ++i) {
    const auto step = reader.Step(i);
    EXPECT_THAT(step, Not(Eq(absl::nullopt)));
    steps.push_back(*step);
  }

  EXPECT_THAT(
      steps,
      ElementsAre(EqualsProto("datum: { values: { float_values: 10.0 } }"),
                  EqualsProto("datum: { values: { float_values: 11.0 } }"),
                  EqualsProto("datum: { values: { float_values: 12.0 } }"),
                  EqualsProto("datum: { values: { float_values: 13.0 } }"),
                  EqualsProto("datum: { values: { float_values: 14.0 } }"),
                  EqualsProto("datum: { values: { float_values: 15.0 } }"),
                  EqualsProto("datum: { values: { float_values: 16.0 } }")));

  // Check episodes.
  std::vector<EpisodeInfo> episodes;
  for (int i = 0; i < reader.NumEpisodes(); ++i) {
    const auto episode = reader.Episode(i);
    EXPECT_THAT(episode, Not(Eq(absl::nullopt)));
    episodes.push_back(*episode);
  }
  EXPECT_THAT(episodes, ElementsAre(EqualsEpisode(0, 3), EqualsEpisode(3, 3),
                                    EqualsEpisode(6, 1)));
}

// Checks that the index is written and read correctly even when multiple
// flushes occur between writes.
TEST(RiegeliShardIoTest, MultipleFlushes) {
  const std::string steps_filename = file::JoinPath(
      getenv("TEST_TMPDIR"), "my_trajectories.riegeli");
  const std::string step_offsets_filename =
      file::JoinPath(getenv("TEST_TMPDIR"), "my_index.riegeli");
  const std::string episode_metadata_filename = file::JoinPath(
      getenv("TEST_TMPDIR"), "my_episodic_metadata.riegeli");
  const std::string episode_index_filename = file::JoinPath(
      getenv("TEST_TMPDIR"), "my_episodic_index.riegeli");

  // Write some data with random interleaved Flush()es.
  absl::BitGen bitgen;
  const int num_steps = absl::Uniform(bitgen, 500, 2000);
  std::vector<int> episode_starts;
  for (int i = 0; i < 100; ++i) {  // About 0.3 * 100 episodes.
    if (absl::Bernoulli(bitgen, 0.3f)) episode_starts.push_back(i);
  }
  const Data payload = ParseTextProtoOrDie(
      R"PROTO(datum: { values: { float_values: 1.0 } })PROTO");
  {
    RiegeliShardWriter writer;
    ENVLOGGER_EXPECT_OK(writer.Init(
        steps_filename, step_offsets_filename, episode_metadata_filename,
        episode_index_filename, "transpose,brotli:6,chunk_size:1M"));

    for (int i = 0; i < num_steps; ++i) {
      const bool is_new_episode =
          absl::c_find(episode_starts, i) != episode_starts.end();
      writer.AddStep(payload, is_new_episode);

      // Flushes can be issued at any moment while writing trajectories and the
      // writer should still behave correctly in those cases.
      if (absl::Bernoulli(bitgen, 0.5f)) writer.Flush();
    }
  }

  // Check that the index contains the expected data.
  RiegeliShardReader reader;
  ENVLOGGER_EXPECT_OK(reader.Init(steps_filename, step_offsets_filename,
                                  episode_metadata_filename,
                                  episode_index_filename));
  EXPECT_THAT(reader.NumSteps(), Eq(num_steps));
  EXPECT_THAT(reader.NumEpisodes(), Eq(episode_starts.size()));
  for (size_t i = 0; i + 1 < episode_starts.size(); ++i) {
    const auto episode = reader.Episode(i);
    EXPECT_THAT(episode, Not(Eq(absl::nullopt)));
    const int expected_num_steps = episode_starts[i + 1] - episode_starts[i];
    EXPECT_THAT(*episode, EqualsEpisode(episode_starts[i], expected_num_steps));
  }
  const auto last_episode = reader.Episode(episode_starts.size() - 1);
  EXPECT_THAT(last_episode, Not(Eq(absl::nullopt)));
  EXPECT_THAT(*last_episode, EqualsEpisode(episode_starts.back(),
                                           num_steps - episode_starts.back()));

  // Check actual step data in trajectories.
  for (int i = 0; i < reader.NumSteps(); ++i) {
    const auto step = reader.Step(i);
    EXPECT_THAT(step, Not(Eq(absl::nullopt))) << "step " << i << " is nullopt";
    EXPECT_THAT(*step, EqualsProto(payload));
  }
}

void BM_WriteRead(benchmark::State& state) {
  const int num_steps = state.range(0);
  absl::BitGen bitgen;
  const std::string steps_filename = file::JoinPath(
      getenv("TEST_TMPDIR"), "my_trajectories.riegeli");
  const std::string step_offsets_filename =
      file::JoinPath(getenv("TEST_TMPDIR"), "my_index.riegeli");
  const std::string episode_metadata_filename = file::JoinPath(
      getenv("TEST_TMPDIR"), "my_episodic_metadata.riegeli");
  const std::string episode_index_filename = file::JoinPath(
      getenv("TEST_TMPDIR"), "my_episodic_index.riegeli");
  // Create the data once to avoid having malloc() measured in the main loop.
  std::vector<std::pair<Data, bool>> payloads;
  for (int i = 0; i < num_steps; ++i) {
    Data payload;
    payload.mutable_datum()->mutable_values()->add_float_values(
        absl::Bernoulli(bitgen, 0.3f));
    const bool is_new_episode = absl::Bernoulli(bitgen, 0.5f);
    payloads.push_back({payload, is_new_episode});
  }

  auto fn = [&steps_filename, &step_offsets_filename,
             &episode_metadata_filename, &episode_index_filename, &payloads]() {
    {
      RiegeliShardWriter writer;
      ENVLOGGER_EXPECT_OK(writer.Init(
          steps_filename, step_offsets_filename, episode_metadata_filename,
          episode_index_filename, "transpose,brotli:6,chunk_size:1M"));

      for (const auto& [payload, is_new_episode] : payloads) {
        writer.AddStep(payload, is_new_episode);
      }
    }

    RiegeliShardReader reader;
    ENVLOGGER_EXPECT_OK(reader.Init(steps_filename, step_offsets_filename,
                                    episode_metadata_filename,
                                    episode_index_filename));
    for (int64_t i = 0; i < reader.NumSteps(); ++i) {
      absl::optional<Data> payload = reader.Step(i);
    }
  };

  for (auto s : state) {
    fn();
  }
}

BENCHMARK(BM_WriteRead)->Range(1, 2 << 23);

}  // namespace
}  // namespace envlogger

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  return RUN_ALL_TESTS();
}
