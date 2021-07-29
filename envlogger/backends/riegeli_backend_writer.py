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

"""For writing trajectory data to riegeli files."""

from typing import Any, Optional

from absl import logging
from envlogger import step_data
from envlogger.backends import backend_writer
from envlogger.backends import schedulers
from envlogger.backends.python import riegeli_dataset_writer
from envlogger.converters import codec
from pybind11_abseil import status


class RiegeliBackendWriter(backend_writer.BackendWriter):
  """Backend that writes trajectory data to riegeli files."""

  def __init__(
      self,
      data_directory: str,
      max_episodes_per_file: int = 10000,
      writer_options: str = 'transpose,brotli:6,chunk_size:1M',
      flush_scheduler: Optional[schedulers.Scheduler] = None,
      **base_kwargs,
  ):
    """Constructor.

    Calling `close()` will flush the trajectories and the index to disk and will
    ensure that they can be read later on. If it isn't called, there is a large
    risk of losing data. This is particularly common in some RL frameworks that
    do not clean up their environments. If the environment runs for a very long
    time, this can happen only to the last shard, but if the instance is
    short-lived, then a large portion of the trajectories can disappear.

    Args:
      data_directory: Destination for the episode data.
      max_episodes_per_file: maximum number of episodes stored in one file.
      writer_options: Comma-separated list of options that are passed to the
        Riegeli RecordWriter as is.
      flush_scheduler: This controls when data is flushed to permanent storage.
        If `None`, it defaults to a step-wise Bernoulli scheduler with 1/5000
        chances of flushing.
      **base_kwargs: arguments for the base class.
    """
    super().__init__(**base_kwargs)
    self._data_directory = data_directory
    if flush_scheduler is None:
      self._flush_scheduler = schedulers.bernoulli_step_scheduler(
          1.0 / 5000)
    else:
      self._flush_scheduler = flush_scheduler
    self._data_writer = riegeli_dataset_writer.RiegeliDatasetWriter()
    logging.info('self._data_directory: %r', self._data_directory)

    metadata = self._metadata or {}

    try:
      self._data_writer.init(
          data_dir=data_directory,
          metadata=codec.encode(metadata),
          max_episodes_per_shard=max_episodes_per_file,
          writer_options=writer_options)
    except status.StatusNotOk as e:
      logging.exception('exception: %r', e)

  def _record_step(self, data: step_data.StepData,
                   is_new_episode: bool) -> None:
    encoded_data = codec.encode(data)
    self._data_writer.add_step(encoded_data, is_new_episode)
    if self._flush_scheduler is not None and not self._flush_scheduler(data):
      return
    self._data_writer.flush()

  def set_episode_metadata(self, data: Any) -> None:
    encoded_data = codec.encode(data)
    self._data_writer.set_episode_metadata(encoded_data)

  def close(self) -> None:
    logging.info('Deleting the backend with data_dir: %r', self._data_directory)
    self._data_writer.close()
    logging.info('Done deleting the backend with data_dir: %r',
                 self._data_directory)
