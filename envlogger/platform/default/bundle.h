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

#ifndef THIRD_PARTY_PY_ENVLOGGER_PLATFORM_DEFAULT_BUNDLE_H_
#define THIRD_PARTY_PY_ENVLOGGER_PLATFORM_DEFAULT_BUNDLE_H_

#include <functional>
#include <future>  // NOLINT(build/c++11)
#include <vector>

namespace envlogger {
namespace thread {

// Bundles a set of parallel function calls.
class Bundle {
 public:
  Bundle();
  ~Bundle();

  // Adds the function for asynchronous execution. It cannot be called if
  // JoinAll() has been invoked.
  void Add(std::function<void()>&& function);

  // Waits for the execution of the added asynchronous functions to terminate.
  // It must be called before releasing the Bundle object.
  void JoinAll();

 private:
  bool finished_ = false;
  std::vector<std::future<void>> futures_;
};

}  // namespace thread
}  // namespace envlogger

#endif  // THIRD_PARTY_PY_ENVLOGGER_PLATFORM_DEFAULT_BUNDLE_H_
