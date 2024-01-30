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

#include "envlogger/backends/cc/riegeli_dataset_reader.h"

#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
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
            datum->mutable_values()->mutable_float_values_buffer());
        writer.SetWriteSizeHint(datum->values().float_values_size() *
                                sizeof(float));
        riegeli::WriteBigEndianFloats(datum->values().float_values(), writer);
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

PYBIND11_MODULE(riegeli_dataset_reader, m) {
  pybind11::google::ImportProtoModule();
  pybind11::module::import("envlogger.backends.python.episode_info");

  m.doc() = "RiegeliDatasetReader bindings.";

  pybind11::class_<envlogger::RiegeliDatasetReader>(m, "RiegeliDatasetReader")
      .def(pybind11::init<>())
      .def("clone",
           [](envlogger::RiegeliDatasetReader* self)
               -> std::optional<envlogger::RiegeliDatasetReader> {
             absl::StatusOr<envlogger::RiegeliDatasetReader> copy =
                 self->Clone();
             if (!copy.ok()) {
               VLOG(0) << "Error cloning reader: " << copy.status().ToString();
               return std::nullopt;
             }
             return std::move(*copy);
           })
      // Initializes the reader with the given `data_dir`.
      // If successful, `void` is returned with no side effects. Otherwise a
      // `RuntimeError` is raised with an accompanying message.
      // Note: `absl::Status` isn't used because there are incompatibilities
      // between slightly different versions of `pybind11_abseil` when used with
      // different projects. Please see
      // https://github.com/deepmind/envlogger/issues/3 for more details.
      .def(
          "init",
          [](envlogger::RiegeliDatasetReader* self,
             std::string data_dir) -> void {
            const absl::Status status = self->Init(data_dir);
            if (!status.ok()) throw std::runtime_error(status.ToString());
          },
          pybind11::arg("data_dir"))
      .def("metadata", &envlogger::RiegeliDatasetReader::Metadata)
      .def_property_readonly("num_steps",
                             &envlogger::RiegeliDatasetReader::NumSteps)
      .def_property_readonly("num_episodes",
                             &envlogger::RiegeliDatasetReader::NumEpisodes)
      .def("data_dir", &envlogger::RiegeliDatasetReader::DataDir)
      .def("step", &envlogger::RiegeliDatasetReader::Step<envlogger::Data>,
           pybind11::arg("step_index"))
      .def(
          "step",
          [](envlogger::RiegeliDatasetReader* self, int64_t step_index,
             pybind11::object* data) -> pybind11::object {
            if (!pybind11::hasattr(*data, "FromString")) {
              VLOG(0)
                  << "`data` must have a method `FromString()` which "
                     "parses bytes. For example, protobufs have this method.";
              return pybind11::none();
            }

            std::optional<std::string> record =
                self->Step<std::string>(step_index);
            if (!record) {
              VLOG(0) << "Failed to read record at step_index: " << step_index;
              return pybind11::none();
            }

            pybind11::memoryview view = pybind11::memoryview::from_memory(
                record->data(), record->size());
            return data->attr("FromString")(view);
          },
          pybind11::arg("step_index"), pybind11::arg("data"),
          "Same as step(), but allows passing an object that defines "
          "`FromString()` (e.g. protobufs) that will be used for parsing the "
          "(binary) payload.\nExample usage: reader.step(123, MyProtoMessage)")
      // Same as step() except that it returns a serialized envlogger::Data
      // proto. Return type is 'bytes' or None.
      .def("serialized_step",
           [](envlogger::RiegeliDatasetReader* self,
              int64_t step_index) -> std::optional<pybind11::bytes> {
             std::optional<envlogger::Data> data = self->Step(step_index);
             OptimizeDataProto(&*data);
             if (!data) return std::nullopt;
             return pybind11::bytes(data->SerializeAsString());
           })
      .def("episode", &envlogger::RiegeliDatasetReader::Episode,
           pybind11::arg("episode_index"),
           pybind11::arg("include_metadata") = false)
      .def("close", &envlogger::RiegeliDatasetReader::Close)
      // Pickling support.
      .def(pybind11::pickle(
          [](const envlogger::RiegeliDatasetReader& self) {  // __getstate__().
            pybind11::dict output;
            output["data_dir"] = self.DataDir();
            return output;
          },
          [](pybind11::dict d) {  // __setstate__().
            const std::string data_dir = d["data_dir"].cast<std::string>();
            auto reader = std::make_unique<envlogger::RiegeliDatasetReader>();
            const absl::Status status = reader->Init(data_dir);
            if (!status.ok()) {
              throw std::runtime_error(
                  "Failed to initialize RiegeliDatasetReader: " +
                  status.ToString());
            }
            return reader;
          }));
}
