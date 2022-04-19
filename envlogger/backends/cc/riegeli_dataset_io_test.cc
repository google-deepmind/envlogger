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

#include <algorithm>
#include <iterator>
#include <optional>
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
#include "envlogger/backends/cc/episode_info.h"
#include "envlogger/backends/cc/riegeli_dataset_reader.h"
#include "envlogger/backends/cc/riegeli_dataset_writer.h"
#include "envlogger/backends/cc/riegeli_shard_writer.h"
#include "envlogger/platform/filesystem.h"
#include "envlogger/platform/parse_text_proto.h"
#include "envlogger/platform/proto_testutil.h"
#include "envlogger/platform/test_macros.h"
#include "envlogger/proto/storage.pb.h"
#include "riegeli/base/base.h"
#include "riegeli/records/record_reader.h"
#include "riegeli/records/record_writer.h"

namespace envlogger {
namespace {

using ::testing::Eq;
using ::testing::Not;
using ::testing::Value;

// A simple matcher to compare the output of RiegeliDatasetReader::Episode().
MATCHER_P2(EqualsEpisode, start_index, num_steps, "") {
  return Value(arg.start, start_index) && Value(arg.num_steps, num_steps);
}

TEST(RiegeliDatasetTest, MetadataTest) {
  const std::string data_dir =
      file::JoinPath(getenv("TEST_TMPDIR"), "metadata");
  const Data metadata =
      ParseTextProtoOrDie("datum: { values: { int32_values: 1234 } }");
  const int max_episodes_per_shard = -1;

  ENVLOGGER_EXPECT_OK(file::CreateDir(data_dir));
  {
    RiegeliDatasetWriter writer;
    ENVLOGGER_EXPECT_OK(writer.Init(data_dir, metadata, max_episodes_per_shard,
                                    "transpose,brotli:6,chunk_size:1M"));
    // Write a single step to pass RiegeliDatasetReader::Init()'s strict checks.
    Data data;
    data.mutable_datum()->mutable_values()->add_float_values(1.234f);
    writer.AddStep(data, /*is_new_episode=*/true);
    writer.Flush();
  }

  RiegeliDatasetReader reader;
  ENVLOGGER_EXPECT_OK(reader.Init(data_dir));
  const auto actual_metadata = reader.Metadata();
  EXPECT_THAT(actual_metadata, Not(Eq(absl::nullopt)));
  EXPECT_THAT(*actual_metadata, EqualsProto(metadata));

  ENVLOGGER_EXPECT_OK(file::RecursivelyDelete(data_dir));
}

}  // namespace
}  // namespace envlogger
