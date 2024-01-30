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

"""Utils to test the backends."""
import time
from typing import Any, Dict, List, Optional

from absl import logging
from envlogger import step_data
from envlogger.backends import backend_writer
from envlogger.backends import tfds_backend_writer
from envlogger.testing import catch_env
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def generate_episode_data(
    backend: backend_writer.BackendWriter,
    num_episodes: int = 2,
) -> List[List[step_data.StepData]]:
  """Runs a Catch environment for `num_episodes` and logs them.

  Args:
    backend: environment logger writer.
    num_episodes: number of episodes to generate.

  Returns:
    List of generated episodes.

  """
  env = catch_env.Catch()

  logging.info('Training a random agent for %r episodes...', num_episodes)
  episodes_data = []
  for index in range(num_episodes):
    episode = []
    timestep = env.reset()
    data = step_data.StepData(timestep, None, {'timestamp': int(time.time())})
    episode.append(data)
    backend.record_step(data, is_new_episode=True)

    while not timestep.last():
      action = np.random.randint(low=0, high=3)
      timestep = env.step(action)
      data = step_data.StepData(timestep, action,
                                {'timestamp': int(time.time())})
      episode.append(data)
      backend.record_step(data, is_new_episode=False)
    backend.set_episode_metadata({'episode_id': index})
    episodes_data.append(episode)

  logging.info('Done training a random agent for %r episodes.', num_episodes)
  env.close()
  backend.close()
  return episodes_data


def catch_env_tfds_config(
    name: str = 'catch_example') -> tfds.rlds.rlds_base.DatasetConfig:
  """Creates a TFDS DatasetConfig for the Catch environment."""
  return tfds.rlds.rlds_base.DatasetConfig(
      name=name,
      observation_info=tfds.features.Tensor(
          shape=(10, 5), dtype=tf.float32,
          encoding=tfds.features.Encoding.ZLIB),
      action_info=tf.int64,
      reward_info=tf.float64,
      discount_info=tf.float64,
      step_metadata_info={'timestamp': tf.int64},
      episode_metadata_info={'episode_id': tf.int64})


def tfds_backend_catch_env(
    data_directory: str,
    max_episodes_per_file: int = 1,
    split_name: Optional[str] = None,
    ds_metadata: Optional[Dict[Any, Any]] = None,
    store_ds_metadata: bool = True,
) -> tfds_backend_writer.TFDSBackendWriter:
  """Creates a TFDS Backend Writer for the Catch Environment.

  Args:
    data_directory: directory where the data will be created (it has to exist).
    max_episodes_per_file: maximum number of episodes per file.
    split_name: number of the TFDS split to create.
    ds_metadata: metadata of the dataset.
    store_ds_metadata: if the metadata should be stored.
  Returns:
    TFDS backend writer.
  """
  return tfds_backend_writer.TFDSBackendWriter(
      data_directory=data_directory,
      split_name=split_name,
      ds_config=catch_env_tfds_config(),
      max_episodes_per_file=max_episodes_per_file,
      metadata=ds_metadata,
      store_ds_metadata=store_ds_metadata)
