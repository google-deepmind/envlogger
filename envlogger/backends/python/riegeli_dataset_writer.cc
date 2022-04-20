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

#include "envlogger/proto/storage.pb.h"
#include "pybind11//pybind11.h"
#include "pybind11//stl.h"
#include "pybind11_protobuf/proto_casters.h"
#include "riegeli/bytes/string_writer.h"
#include "riegeli/endian/endian_writing.h"

namespace {

// This traverses the proto to convert repeated 'float_values' into big-endian
// byte arrays 'float_values_buffer'. In python this allows using
// np.frombuffer(float_values_buffer) which is overall 2.3x more efficient than
// building the np array from the repeated field (this includes the cost of
// doing the conversion in C++).
// In the future, we could do the same for other data types stored in repeated
// fields.
void OptimizeDataProto(envlogger::Data* data) {
  switch (data->value_case()) {
    case envlogger::Data::kDatum: {
      auto* datum = data->mutable_datum();
      if (!datum->values().float_values().empty()) {
        riegeli::StringWriter writer(
            datum->mutable_values()->mutable_float_values_buffer(),
            riegeli::StringWriterBase::Options().set_size_hint(
                datum->values().float_values_size() * sizeof(float)));
        for (float v : datum->values().float_values()) {
          // We need this endian transformation because codec.decode() expects
          // big endian values. Otherwise we could just have done a bit_cast<>
          // of datum->values().float_values().data() into the bytes buffer.
          riegeli::WriteBigEndian32(absl::bit_cast<uint32_t>(v), writer);
        }
        writer.Close();
        datum->mutable_values()->clear_float_values();
      }
      break;
    }
    case envlogger::Data::kArray:
      for (auto& value : *data->mutable_array()->mutable_values()) {
        OptimizeDataProto(&value);
      }
      break;
    case envlogger::Data::kTuple:
      for (auto& value : *data->mutable_tuple()->mutable_values()) {
        OptimizeDataProto(&value);
      }
      break;
    case envlogger::Data::kDict:
      for (auto& value : *data->mutable_dict()->mutable_values()) {
        OptimizeDataProto(&value.second);
      }
      break;
    case envlogger::Data::VALUE_NOT_SET:
      break;
  }
}
}  // namespace

PYBIND11_MODULE(riegeli_dataset_writer, m) {
  pybind11::google::ImportProtoModule();
  pybind11::module::import("envlogger.backends.python.episode_info");

  m.doc() = "RiegeliDatasetWriter bindings.";

  pybind11::class_<envlogger::RiegeliDatasetWriter>(m, "RiegeliDatasetWriter")
      .def(pybind11::init<>())
      // Initializes the writer with the given arguments.
      // If successful, `void` is returned with no side effects. Otherwise a
      // `RuntimeError` is raised with an accompanying message.
      // Note: `absl::Status` isn't used because there are incompatibilities
      // between slightly different versions of `pybind11_abseil` when used with
      // different projects. Please see
      // https://github.com/deepmind/envlogger/issues/3 for more details.
      .def(
          "init",
          [](envlogger::RiegeliDatasetWriter* self, std::string data_dir,
             const envlogger::Data& metadata = envlogger::Data(),
             int64_t max_episodes_per_shard = 0,
             std::string writer_options =
                 "transpose,brotli:6,chunk_size:1M") -> void {
            const absl::Status status = self->Init(
                data_dir, metadata, max_episodes_per_shard, writer_options);
            if (!status.ok()) throw std::runtime_error(status.ToString());
          },
          pybind11::arg("data_dir"),
          pybind11::arg("metadata") = envlogger::Data(),
          pybind11::arg("max_episodes_per_shard") = 0,
          pybind11::arg("writer_options") = "transpose,brotli:6,chunk_size:1M")
      .def("add_step", &envlogger::RiegeliDatasetWriter::AddStep,
           pybind11::arg("data"), pybind11::arg("is_new_episode") = false)
      .def("set_episode_metadata",
           &envlogger::RiegeliDatasetWriter::SetEpisodeMetadata,
           pybind11::arg("data"))
      .def("flush", &envlogger::RiegeliDatasetWriter::Flush)
      .def("close", &envlogger::RiegeliDatasetWriter::Close);
}
