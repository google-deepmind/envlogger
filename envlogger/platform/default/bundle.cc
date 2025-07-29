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

#include "envlogger/platform/default/bundle.h"

#include <functional>
#include <future>  // NOLINT(build/c++11)
#include <utility>

#include "glog/logging.h"

namespace envlogger {
namespace thread {

Bundle::Bundle() : finished_(false) {}
Bundle::~Bundle() {
  CHECK(finished_) << "JoinAll() should be called before releasing the bundle.";
}

void Bundle::Add(std::function<void()>&& function) {
  CHECK(!finished_) << "Add cannot be called after JoinAll is invoked.";
  futures_.push_back(std::async(std::launch::async, std::move(function)));
}

void Bundle::JoinAll() {
  CHECK(!finished_) << "JoinAll should be called only once.";
  finished_ = true;
  for (const auto& future : futures_) {
    future.wait();
  }
  futures_.clear();
}

}  // namespace thread
}  // namespace envlogger
