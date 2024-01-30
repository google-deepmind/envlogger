# coding=utf-8
# Copyright 2024 DeepMind Technologies Limited..
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

"""TFDS backend for Envlogger."""
import dataclasses
from typing import Any, Dict, List, Optional

from absl import logging
from envlogger import step_data
from envlogger.backends import backend_writer
from envlogger.backends import rlds_utils
import tensorflow_datasets as tfds


DatasetConfig = tfds.rlds.rlds_base.DatasetConfig


@dataclasses.dataclass
class Episode(object):
  """Episode that is being constructed."""
  prev_step: step_data.StepData
  steps: Optional[List[rlds_utils.Step]] = None
  metadata: Optional[Dict[str, Any]] = None

  def add_step(self, step: step_data.StepData) -> None:
    rlds_step = rlds_utils.to_rlds_step(self.prev_step, step)
    if self.steps is None:
      self.steps = []
    self.steps.append(rlds_step)
    self.prev_step = step

  def get_rlds_episode(self) -> Dict[str, Any]:
    last_step = rlds_utils.to_rlds_step(self.prev_step, None)
    if self.steps is None:
      self.steps = []
    if self.metadata is None:
      self.metadata = {}

    return {'steps': self.steps + [last_step], **self.metadata}


class TFDSBackendWriter(backend_writer.BackendWriter):
  """Backend that writes trajectory data in TFDS format (and RLDS structure)."""


  def __init__(self,
               data_directory: str,
               ds_config: tfds.rlds.rlds_base.DatasetConfig,
               max_episodes_per_file: int = 1000,
               split_name: Optional[str] = None,
               version: str = '0.0.1',
               store_ds_metadata: bool = False,
               **base_kwargs):
    """Constructor.

    Args:
      data_directory: Directory to store the data
      ds_config: Dataset Configuration.
      max_episodes_per_file: Number of episodes to store per shard.
      split_name: Name to be used by the split. If None, 'train' will be used.
      version: version (major.minor.patch) of the dataset.
      store_ds_metadata: if False, it won't store the dataset level
        metadata.
      **base_kwargs: arguments for the base class.
    """
    super().__init__(**base_kwargs)
    if not split_name:
      split_name = 'train'
    ds_identity = tfds.core.dataset_info.DatasetIdentity(
        name=ds_config.name,
        version=tfds.core.Version(version),
        data_dir=data_directory,
        module_name='')
    if store_ds_metadata:
      metadata = self._metadata
    else:
      metadata = None
    self._data_directory = data_directory
    self._ds_info = tfds.rlds.rlds_base.build_info(ds_config, ds_identity,
                                                   metadata)
    self._ds_info.set_file_format('tfrecord')

    self._current_episode = None

    self._sequential_writer = tfds.core.SequentialWriter(
        self._ds_info, max_episodes_per_file)
    self._split_name = split_name
    self._sequential_writer.initialize_splits([split_name])
    logging.info('self._data_directory: %r', self._data_directory)

  def _write_and_reset_episode(self):
    if self._current_episode is not None:
      self._sequential_writer.add_examples(
          {self._split_name: [self._current_episode.get_rlds_episode()]})
      self._current_episode = None

  def _record_step(self, data: step_data.StepData,
                   is_new_episode: bool) -> None:
    """Stores RLDS steps in TFDS format."""

    if is_new_episode:
      self._write_and_reset_episode()

    if self._current_episode is None:
      self._current_episode = Episode(prev_step=data)
    else:
      self._current_episode.add_step(data)

  def set_episode_metadata(self, data: Dict[str, Any]) -> None:
    self._current_episode.metadata = data

  def close(self) -> None:
    logging.info('Deleting the backend with data_dir: %r', self._data_directory)
    self._write_and_reset_episode()
    self._sequential_writer.close_all()
    logging.info('Done deleting the backend with data_dir: %r',
                 self._data_directory)
