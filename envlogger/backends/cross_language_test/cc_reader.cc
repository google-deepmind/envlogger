// Copyright 2025 DeepMind Technologies Limited..
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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/flags/parse.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/flags/flag.h"
#include <gmpxx.h>
#include "envlogger/backends/cc/riegeli_dataset_reader.h"
#include "envlogger/converters/xtensor_codec.h"
#include "envlogger/platform/proto_testutil.h"
#include "envlogger/proto/storage.pb.h"
#include "xtensor/xarray.hpp"

ABSL_FLAG(std::string, trajectories_dir, "", "Path to reader trajectory.");

using ::testing::DoubleEq;
using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::IsTrue;
using ::testing::SizeIs;

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  VLOG(0) << "Starting C++ Reader...";
  VLOG(0) << "--trajectories_dir: " << absl::GetFlag(FLAGS_trajectories_dir);
  envlogger::RiegeliDatasetReader reader;
  const absl::Status init_status =
      reader.Init(absl::GetFlag(FLAGS_trajectories_dir));
  VLOG(0) << "init_status: " << init_status;

  VLOG(0) << "reader.NumSteps(): " << reader.NumSteps();
  for (int64_t i = 0; i < reader.NumSteps(); ++i) {
    std::optional<envlogger::Data> step = reader.Step(i);
    EXPECT_THAT(step.has_value(), IsTrue())
        << "All steps should be readable. Step " << i << " is not available.";

    envlogger::DataView step_view(std::addressof(*step));
    EXPECT_THAT(step_view->value_case(), Eq(envlogger::Data::ValueCase::kTuple))
        << "Each step should be a tuple.";
    EXPECT_THAT(step_view, SizeIs(3))
        << "Each step should consist of (timestep, action, custom data)";
    const envlogger::Data& timestep = *step_view[0];
    const envlogger::Data& action = *step_view[1];
    const envlogger::Data& custom_data = *step_view[2];
    VLOG(1) << "timestep: " << timestep.ShortDebugString();
    VLOG(1) << "action: " << action.ShortDebugString();
    VLOG(1) << "custom_data: " << custom_data.ShortDebugString();
    envlogger::DataView timestep_view(&timestep);
    EXPECT_THAT(timestep_view->value_case(),
                Eq(envlogger::Data::ValueCase::kTuple))
        << "Each timestep should be a tuple.";
    EXPECT_THAT(timestep_view, SizeIs(4))
        << "Each timestep should consist of (step type, reward, discount, "
           "observation)";

    // Check timestep values.
    // Check step type.
    const envlogger::Data& step_type = *timestep_view[0];
    VLOG(2) << "step_type: " << step_type.ShortDebugString();
    std::optional<envlogger::BasicType> decoded_step_type =
        envlogger::Decode(step_type.datum());
    EXPECT_THAT(decoded_step_type.has_value(), IsTrue())
        << "Failed to decode step_type";
    EXPECT_THAT(absl::holds_alternative<mpz_class>(*decoded_step_type),
                IsTrue());
    const mpz_class step_type_decoded =
        absl::get<mpz_class>(*decoded_step_type);
    EXPECT_THAT(cmp(step_type_decoded, i ? 1 : 0), Eq(0));
    // Check reward.
    const envlogger::Data& reward = *timestep_view[1];
    VLOG(2) << "reward: " << reward.ShortDebugString();
    std::optional<envlogger::BasicType> decoded_reward =
        envlogger::Decode(reward.datum());
    EXPECT_THAT(decoded_reward.has_value(), IsTrue())
        << "Failed to decode reward";
    EXPECT_THAT(absl::holds_alternative<double>(*decoded_reward), IsTrue());
    const double r = absl::get<double>(*decoded_reward);
    EXPECT_THAT(r, DoubleEq(i / 100.0));
    // Check discount.
    const envlogger::Data& discount = *timestep_view[2];
    VLOG(2) << "discount: " << discount.ShortDebugString();
    std::optional<envlogger::BasicType> decoded_discount =
        envlogger::Decode(discount.datum());
    EXPECT_THAT(decoded_discount.has_value(), IsTrue())
        << "Failed to decode discount";
    EXPECT_THAT(absl::holds_alternative<double>(*decoded_discount), IsTrue());
    const double gamma = absl::get<double>(*decoded_discount);
    EXPECT_THAT(gamma, DoubleEq(0.99));
    // Check observation.
    const envlogger::Data& observation = *timestep_view[3];
    VLOG(2) << "observation: " << observation.ShortDebugString();
    std::optional<envlogger::BasicType> decoded_obs =
        envlogger::Decode(observation.datum());
    EXPECT_THAT(decoded_obs.has_value(), IsTrue())
        << "Failed to decode observation";
    EXPECT_THAT(absl::holds_alternative<xt::xarray<float>>(*decoded_obs),
                IsTrue());
    const xt::xarray<float>& obs = absl::get<xt::xarray<float>>(*decoded_obs);
    EXPECT_THAT(obs, SizeIs(1));
    EXPECT_THAT(obs(0), FloatEq(i));

    // Check action.
    std::optional<envlogger::BasicType> decoded_action =
        envlogger::Decode(action.datum());
    EXPECT_THAT(decoded_action.has_value(), IsTrue())
        << "Failed to decode action";
    EXPECT_THAT(absl::holds_alternative<int>(*decoded_action), IsTrue());
    const int a = absl::get<int>(*decoded_action);
    EXPECT_THAT(a, Eq(100 - i));

    // There should be no custom data, but it should still be a valid pointer.
    EXPECT_THAT(custom_data, EqualsProto(envlogger::Data()));
  }

  reader.Close();

  return 0;
}
