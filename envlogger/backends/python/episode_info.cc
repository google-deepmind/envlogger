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

#include "envlogger/backends/cc/episode_info.h"

#include <cstdint>
#include <optional>

#include "envlogger/proto/storage.pb.h"
#include "pybind11//pybind11.h"
#include "pybind11//stl.h"
#include "pybind11_protobuf/proto_casters.h"

PYBIND11_MODULE(episode_info, m) {
  pybind11::google::ImportProtoModule();

  m.doc() = "EpisodeInfo bindings.";

  pybind11::class_<envlogger::EpisodeInfo>(m, "EpisodeInfo")
      .def(pybind11::init<int64_t, int64_t, std::optional<envlogger::Data>>(),
           pybind11::arg("start") = 0, pybind11::arg("num_steps") = 0,
           pybind11::arg("metadata") = std::nullopt)
      .def_readwrite("start", &envlogger::EpisodeInfo::start)
      .def_readwrite("num_steps", &envlogger::EpisodeInfo::num_steps)
      .def_readwrite("metadata", &envlogger::EpisodeInfo::metadata);
}
