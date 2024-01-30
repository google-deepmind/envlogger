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

"""Tests for tfds_backend_writer."""

from typing import List

from absl.testing import absltest
import dm_env
from envlogger import step_data
from envlogger.backends import rlds_utils
from envlogger.backends import tfds_backend_testlib
from envlogger.backends import tfds_backend_writer
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def _create_step(value: int, step_type: dm_env.StepType) -> step_data.StepData:
  return step_data.StepData(
      action=value, timestep=dm_env.TimeStep(step_type, value, value, value))


def _tfds_features() -> tfds.features.FeaturesDict:
  return tfds.features.FeaturesDict({
      'steps':
          tfds.features.Dataset({
              'observation': tf.int64,
              'action': tf.int64,
              'reward': tf.int64,
              'is_terminal': tf.bool,
              'is_first': tf.bool,
              'is_last': tf.bool,
              'discount': tf.int64,
          }),
  })


class TfdsBackendWriterEpisodeTest(absltest.TestCase):

  def test_add_step(self):
    episode = tfds_backend_writer.Episode(
        _create_step(0, dm_env.StepType.FIRST))
    step = _create_step(1, dm_env.StepType.MID)
    episode.add_step(step)

    self.assertEqual(episode.prev_step, step)
    self.assertLen(episode.steps, 1)

    expected_rlds_step = {
        'observation': 0,
        'action': 1,
        'reward': 1,
        'discount': 1,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
    }

    self.assertEqual(episode.steps[0], expected_rlds_step)

  def test_get_rlds_episode(self):
    episode = tfds_backend_writer.Episode(
        _create_step(0, dm_env.StepType.FIRST))
    episode.add_step(_create_step(1, dm_env.StepType.MID))
    episode.add_step(_create_step(2, dm_env.StepType.LAST))

    rlds_episode = episode.get_rlds_episode()

    self.assertIsInstance(rlds_episode, dict)
    self.assertIn('steps', rlds_episode)

    steps_counter = 0
    for index, step in enumerate(rlds_episode['steps']):
      self.assertEqual(index, step['observation'])
      self.assertFalse(step['is_terminal'])
      self.assertEqual(index == 0, step['is_first'])
      self.assertEqual(index == 2, step['is_last'])
      next_value = 0 if index == 2 else index + 1
      for key in ['action', 'reward', 'discount']:
        self.assertEqual(next_value, step[key])
      steps_counter += 1
    self.assertEqual(steps_counter, 3)


class TfdsBackendWriterTest(absltest.TestCase):

  def _assert_steps(self, expected_steps: List[step_data.StepData],
                    steps: tf.data.Dataset):
    steps = steps.as_numpy_iterator()
    for idx, rlds_step in enumerate(steps):
      step = expected_steps[idx + 1] if idx < len(expected_steps) - 1 else None
      expected_step = rlds_utils.to_rlds_step(expected_steps[idx], step)
      np.testing.assert_equal(expected_step, rlds_step)

  def test_backend_writer(self):
    num_episodes = 5
    max_episodes_per_file = 3
    data_dir = self.create_tempdir(name='my_data_dir').full_path
    expected_episodes = tfds_backend_testlib.generate_episode_data(
        backend=tfds_backend_testlib.tfds_backend_catch_env(
            data_directory=data_dir,
            max_episodes_per_file=max_episodes_per_file),
        num_episodes=num_episodes)

    builder = tfds.builder_from_directory(data_dir)
    ds = builder.as_dataset(split='train')

    num_episodes = 0
    for index, episode in enumerate(ds):
      self._assert_steps(expected_episodes[index], episode['steps'])
      self.assertEqual(episode['episode_id'], index)
      num_episodes += 1

    self.assertLen(expected_episodes, num_episodes)

  def test_backend_writer_with_split_name(self):
    num_episodes = 1
    max_episodes_per_file = 1
    data_dir = self.create_tempdir(name='my_data_dir').full_path
    expected_episodes = tfds_backend_testlib.generate_episode_data(
        backend=tfds_backend_testlib.tfds_backend_catch_env(
            data_directory=data_dir,
            max_episodes_per_file=max_episodes_per_file,
            split_name='split'),
        num_episodes=num_episodes)

    builder = tfds.builder_from_directory(data_dir)
    ds = builder.as_dataset(split='split')

    num_episodes = 0
    for index, episode in enumerate(ds):
      self._assert_steps(expected_episodes[index], episode['steps'])
      self.assertEqual(episode['episode_id'], index)
      num_episodes += 1

    self.assertLen(expected_episodes, num_episodes)

  def test_backend_writer_with_dataset_metadata(self):
    num_episodes = 5
    max_episodes_per_file = 3
    data_dir = self.create_tempdir(name='my_data_dir').full_path
    _ = tfds_backend_testlib.generate_episode_data(
        backend=tfds_backend_testlib.tfds_backend_catch_env(
            data_directory=data_dir,
            max_episodes_per_file=max_episodes_per_file,
            ds_metadata={'env_name': 'catch'},
            store_ds_metadata=True),
        num_episodes=num_episodes)

    builder = tfds.builder_from_directory(data_dir)
    info = builder.info
    self.assertDictEqual(info.metadata, {'env_name': 'catch'})

  def test_backend_writer_without_dataset_metadata(self):
    num_episodes = 5
    max_episodes_per_file = 3
    data_dir = self.create_tempdir(name='my_data_dir').full_path
    _ = tfds_backend_testlib.generate_episode_data(
        backend=tfds_backend_testlib.tfds_backend_catch_env(
            data_directory=data_dir,
            max_episodes_per_file=max_episodes_per_file,
            ds_metadata=None,
            store_ds_metadata=True),
        num_episodes=num_episodes)

    builder = tfds.builder_from_directory(data_dir)
    info = builder.info
    self.assertIsNone(info.metadata)

  def test_backend_writer_ignore_dataset_metadata(self):
    num_episodes = 5
    max_episodes_per_file = 3
    data_dir = self.create_tempdir(name='my_data_dir').full_path
    _ = tfds_backend_testlib.generate_episode_data(
        backend=tfds_backend_testlib.tfds_backend_catch_env(
            data_directory=data_dir,
            max_episodes_per_file=max_episodes_per_file,
            ds_metadata={'env_name': 'catch'},
            store_ds_metadata=False),
        num_episodes=num_episodes)

    builder = tfds.builder_from_directory(data_dir)
    info = builder.info
    self.assertIsNone(info.metadata)


if __name__ == '__main__':
  absltest.main()
