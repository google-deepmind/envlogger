# coding=utf-8
# Copyright 2022 DeepMind Technologies Limited..
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
import os
from typing import Any, Dict, List, Optional

from absl import logging
from envlogger import step_data
from envlogger.backends import backend_writer
from envlogger.backends import rlds_utils
import tensorflow as tf
import tensorflow_datasets as tfds


DatasetConfig = tfds.rlds.rlds_base.DatasetConfig


class RLDSEnvloggerBuilder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for RLDS datasets generated with TFDS BackendWriter."""

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.rlds.rlds_base.build_info(self.builder_config, self)

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # This builder is only used to write data from envlogger. Examples are
    # generated with the tfds_backend_writer module.
    raise NotImplementedError

  def _generate_examples(self, path):
    """Yields examples."""
    # This builder is only used to write data from envlogger. Examples are
    # generated with the tfds_backend_writer module.
    raise NotImplementedError


@dataclasses.dataclass
class Shard(object):
  """Shard represents a shard that is being written."""
  writer: tf.io.TFRecordWriter
  num_episodes: int = 0
  num_bytes: int = 0

  def add_episode(self, serialized_episode: str) -> None:
    self.writer.write(serialized_episode)
    self.num_episodes += 1
    self.num_bytes += len(serialized_episode)

  def close_writer(self) -> None:
    self.writer.flush()
    self.writer.close()


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

  def serialize_episode(
      self, features: tfds.features.FeaturesDict,
      serializer: tfds.core.example_serializer.ExampleSerializer) -> str:
    """Serializes an episode."""
    last_step = rlds_utils.to_rlds_step(self.prev_step, None)
    if self.steps is None:
      self.steps = []
    if self.metadata is None:
      self.metadata = {}

    episode = {'steps': self.steps + [last_step], **self.metadata}
    try:
      example = features.encode_example(episode)
    except Exception as e:
      tfds.core.utils.reraise(
          e, prefix='Failed to encode episode:\n', suffix=f'{episode}\n')

    return serializer.serialize_example(example)


@dataclasses.dataclass
class Split(object):
  """Information of a split that is being created."""
  info: tfds.core.splits.SplitInfo
  complete_shards: int = 0
  # The dataset name is taken from the builder class.
  ds_name: str = 'rlds_envlogger_builder'

  def update(self, shard: Shard) -> None:
    self.info = tfds.core.SplitInfo(
        name=self.info.name,
        shard_lengths=self.info.shard_lengths + [shard.num_episodes],
        num_bytes=self.info.num_bytes + shard.num_bytes,
        filename_template=self.info.filename_template)
    self.complete_shards += 1

  def get_shard_path(self) -> str:
    # The original names are: <ds_name>-<split>.<file extension>-xxxxx-of-yyyyy
    # At this point we don't know the number of shards, so '-of-yyyyy' is
    # not part of the name.
    filename_prefix = f'{self.ds_name}-{self.info.name}.tfrecord'
    filename = f'{filename_prefix}-{self.complete_shards:05d}'
    return filename

  def get_split_dict(self) -> tfds.core.splits.SplitDict:
    return tfds.core.splits.SplitDict([self.info])


def initialize_split(split_name: Optional[str], data_directory: Any,
                     builder_name: str) -> Split:
  """Initializes a split.

  Args:
    split_name: name of the split. If None, it uses `train`.
    data_directory: directory where the split data will be located.
    builder_name: name of the TFDS builder.

  Returns:
    A Split.

  """
  if not split_name:
    split_name = 'train'

  filename_template = tfds.core.ShardedFileTemplate(
      dataset_name=builder_name,
      data_dir=data_directory,
      split=split_name,
      filetype_suffix='tfrecord',
      template='{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_INDEX}',
  )
  return Split(
      info=tfds.core.splits.SplitInfo(
          name=split_name,
          shard_lengths=[],
          num_bytes=0,
          filename_template=filename_template),
      ds_name=builder_name)


class TFDSBackendWriter(backend_writer.BackendWriter):
  """Backend that writes trajectory data in TFDS format (and RLDS structure)."""


  def __init__(self,
               data_directory: str,
               ds_config: tfds.rlds.rlds_base.DatasetConfig,
               max_episodes_per_file: int = 1000,
               split_name: Optional[str] = None,
               version: str = '0.0.1',
               **base_kwargs):
    """Constructor.

    Args:
      data_directory: Directory to store the data
      ds_config: Dataset Configuration.
      max_episodes_per_file: Number of episodes to store per shard.
      split_name: Name to be used by the split. If None, the name of the parent
        directory will be used.
      version: version (major.minor.patch) of the dataset.
      **base_kwargs: arguments for the base class.
    """
    super().__init__(**base_kwargs)
    self._data_directory = data_directory
    builder_cls = RLDSEnvloggerBuilder
    builder_cls.VERSION = version
    builder_cls.BUILDER_CONFIGS = [ds_config]
    self._builder = builder_cls(data_dir=data_directory, config=ds_config.name)
    self._ds_info = tfds.rlds.rlds_base.build_info(ds_config, self._builder)
    self._ds_info.set_file_format('tfrecord')

    self._serializer = tfds.core.example_serializer.ExampleSerializer(
        self._ds_info.features.get_serialized_info())

    self._max_episodes_per_shard = max_episodes_per_file
    self._current_shard = None
    self._current_episode = None

    self._split = initialize_split(split_name, data_directory,
                                   self._builder.name)
    # We write the empty metadata so, when reading, we can always build the
    # dataset even if it's empty.
    self._write_split_metadata()
    logging.info('self._data_directory: %r', self._data_directory)

  def _write_split_metadata(self) -> None:
    self._ds_info.set_splits(self._split.get_split_dict())
    # Either writes the first metadata, or overwrites it.
    self._ds_info.write_to_directory(self._data_directory)

  def _finalize_shard_and_update_split(self) -> None:
    self._current_shard.close_writer()
    self._split.update(self._current_shard)
    self._current_shard = None
    self._write_split_metadata()

  def _finalize_episode_and_update_shard(self) -> None:
    serialized_example = self._current_episode.serialize_episode(
        self._ds_info.features, self._serializer)
    self._current_episode = None

    if self._current_shard is None:
      path = os.path.join(self._data_directory, self._split.get_shard_path())
      self._current_shard = Shard(writer=tf.io.TFRecordWriter(os.fspath(path)))

    self._current_shard.add_episode(serialized_example)

  def _write_and_reset_episode(self, is_last_episode: bool = False) -> None:
    if self._current_episode is None and self._current_shard is None:
      return

    if self._current_episode is not None:
      self._finalize_episode_and_update_shard()

    if is_last_episode or self._current_shard.num_episodes >= self._max_episodes_per_shard:
      self._finalize_shard_and_update_split()

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
    self._write_and_reset_episode(is_last_episode=True)
    logging.info('Done deleting the backend with data_dir: %r',
                 self._data_directory)
