# coding=utf-8
# Copyright 2021 DeepMind Technologies Limited..
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Environment logger backend that stores all data in RAM.
"""

from typing import Any

from envlogger import step_data
from envlogger.backends import backend_reader
from envlogger.backends import backend_writer
from envlogger.backends.python import episode_info


class InMemoryBackendWriter(backend_writer.BackendWriter):
  """Backend that stores trajectory data in memory."""

  def __init__(self, **base_kwargs):
    super().__init__(**base_kwargs)
    self.steps = []
    self.episode_metadata = {}
    self.episode_start_indices = []

  def _record_step(self, data: step_data.StepData,
                   is_new_episode: bool) -> None:
    if is_new_episode:
      self.episode_start_indices.append(len(self.steps))
    self.steps.append(data)

  def set_episode_metadata(self, data: Any) -> None:
    current_episode = len(self.episode_start_indices)
    if current_episode > 0:
      self.episode_metadata[current_episode] = data

  def close(self) -> None:
    pass


class InMemoryBackendReader(backend_reader.BackendReader):
  """Reader that reads data from an InMemoryBackend."""

  def __init__(self, in_memory_backend_writer: InMemoryBackendWriter):
    self._backend = in_memory_backend_writer
    super().__init__()

  def close(self) -> None:
    pass

  def _get_nth_step(self, i: int) -> step_data.StepData:
    return self._backend.steps[i]

  def _get_nth_episode_info(self,
                            i: int,
                            include_metadata: bool = False
                           ) -> episode_info.EpisodeInfo:
    if i == len(self._backend.episode_start_indices) - 1:  # Last episode.
      length = len(self._backend.steps) - self._backend.episode_start_indices[i]
    else:
      length = (self._backend.episode_start_indices[i + 1] -
                self._backend.episode_start_indices[i])
    episode_metadata = self._backend.episode_metadata.get(i, None)
    return episode_info.EpisodeInfo(
        start=self._backend.episode_start_indices[i],
        num_steps=length,
        metadata=episode_metadata)

  def _get_num_steps(self) -> int:
    return len(self._backend.steps)

  def _get_num_episodes(self) -> int:
    return len(self._backend.episode_start_indices)

  def metadata(self):
    return self._backend.metadata()
