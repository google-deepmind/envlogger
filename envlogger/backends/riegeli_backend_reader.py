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

"""For reading trajectory data from riegeli files."""

from typing import Any, Dict, Tuple

from absl import logging
import dm_env
from envlogger import step_data
from envlogger.backends import backend_reader
from envlogger.backends.python import episode_info
from envlogger.backends.python import riegeli_dataset_reader
from envlogger.converters import codec
from envlogger.proto import storage_pb2

from pybind11_abseil import status




class RiegeliBackendReader(backend_reader.BackendReader):
  """A class that reads logs produced by an EnvironmentLoggerWrapper instance.

  Attributes:
    episodes: Traverse the data episode-wise in list-like fashion.
    steps: Traverse the data stepwise in list-like fashion.
  """

  def __init__(self, data_directory: str):
    self._reader = riegeli_dataset_reader.RiegeliDatasetReader()
    try:
      self._reader.init(data_directory)
    except status.StatusNotOk as e:
      if (e.status.code() == status.StatusCode.NOT_FOUND and
          e.status.message().startswith('Empty steps in ')):
        # This case happens frequently when clients abruptly kill the
        # EnvironmentLogger without calling its .close() method, which then
        # causes the last shard to be truncated. This can be because the client
        # exited successfully and "forgot" to call .close(), which is a bug, but
        # also because of a preempted work unit, which is expected to happen
        # under distributed settings.
        # We can't do much to fix the bad usages, but we can be a bit more
        # permissive and try to read the successful shards.
        logging.exception("""Ignoring error due to empty step offset file.
                      *********************************
                      ****   You likely forgot to   ***
                      **** call close() on your env ***
                      ****                          ***
                      *********************************""")
      else:
        raise ValueError(f'Reader init fails: {e}, {type(e)}')

    self._metadata = codec.decode(self._reader.metadata()) or {}
    super().__init__()

  def close(self):
    if self._reader is not None:
      self._reader.close()
    self._reader = None

  def _decode_step_data(self, data: Tuple[Any, Any, Any]) -> step_data.StepData:
    """Recovers dm_env.TimeStep from logged data (either dict or tuple)."""
    # Recover the TimeStep from the first tuple element.
    timestep = dm_env.TimeStep(
        dm_env.StepType(data[0][0]), data[0][1], data[0][2], data[0][3])
    return step_data.StepData(timestep, data[1], data[2])

  def _get_num_steps(self):
    return self._reader.num_steps

  def _get_num_episodes(self):
    return self._reader.num_episodes

  def _get_nth_step(self, i: int) -> step_data.StepData:
    """Returns the timestep given by offset `i` (0-based)."""
    serialized_data = self._reader.serialized_step(i)
    data = storage_pb2.Data.FromString(serialized_data)
    return self._decode_step_data(codec.decode(data))

  def _get_nth_episode_info(self,
                            i: int,
                            include_metadata: bool = False
                           ) -> episode_info.EpisodeInfo:
    """Returns the index of the start of nth episode, and its length."""
    return self._reader.episode(i, include_metadata)

  def metadata(self):
    return self._metadata
