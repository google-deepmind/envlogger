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

"""Tests for rlds_utils."""

from absl.testing import absltest
import dm_env
from envlogger import step_data
from envlogger.backends import rlds_utils
from envlogger.backends import tfds_backend_testlib
import numpy as np
import tensorflow_datasets as tfds


class RldsUtilsTest(absltest.TestCase):

  def test_build_step(self):
    prev_step = step_data.StepData(
        timestep=dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=1,
            observation=2,
            discount=3),
        action=4)
    step = step_data.StepData(
        timestep=dm_env.TimeStep(
            step_type=dm_env.StepType.LAST, reward=5, observation=6,
            discount=7),
        action=8)

    expected_step = {
        'observation': 2,
        'action': 8,
        'reward': 5,
        'discount': 7,
        'is_terminal': False,
        'is_first': True,
        'is_last': False,
    }

    rlds_step = rlds_utils.to_rlds_step(prev_step, step)

    self.assertEqual(rlds_step, expected_step)

  def test_build_last_step(self):
    prev_step = step_data.StepData(
        timestep=dm_env.TimeStep(
            step_type=dm_env.StepType.LAST, reward=1, observation=2,
            discount=1),
        action=4)
    expected_step = {
        'observation': 2,
        'action': 0,
        'reward': 0,
        'discount': 0,
        'is_terminal': False,
        'is_first': False,
        'is_last': True,
    }

    rlds_step = rlds_utils.to_rlds_step(prev_step, None)

    self.assertEqual(rlds_step, expected_step)

  def test_build_nested_last_step(self):

    def gen_oar(mode):
      gen_fn = np.ones if mode == 'random' else np.zeros
      obs = {'0': gen_fn((1, 2)), '1': gen_fn((3, 4))}
      action = {'0': gen_fn((2, 3)), '1': gen_fn((4, 5))}
      reward = {'0': gen_fn((1, 1)), '1': gen_fn((2, 2))}
      return obs, action, reward

    prev_obs, prev_action, prev_reward = gen_oar('ones')
    prev_step = step_data.StepData(
        timestep=dm_env.TimeStep(
            step_type=dm_env.StepType.LAST,
            reward=prev_reward,
            observation=prev_obs,
            discount=1),
        action=prev_action)

    _, zero_action, zero_reward = gen_oar('zeros')
    expected_step = {
        'observation': prev_obs,
        'action': zero_action,
        'reward': zero_reward,
        'discount': 0,
        'is_terminal': False,
        'is_first': False,
        'is_last': True,
    }

    rlds_step = rlds_utils.to_rlds_step(prev_step, None)
    for key in rlds_step.keys():
      if isinstance(rlds_step[key], dict):  # obs, action, reward dicts
        for rv, ev in zip(rlds_step[key].values(), expected_step[key].values()):
          np.testing.assert_equal(rv, ev)
      else:
        self.assertEqual(rlds_step[key], expected_step[key])

  def test_build_terminal_step(self):
    prev_step = step_data.StepData(
        timestep=dm_env.TimeStep(
            step_type=dm_env.StepType.LAST, reward=1, observation=2,
            discount=0),
        action=4)
    expected_step = {
        'observation': 2,
        'action': 0,
        'reward': 0,
        'discount': 0,
        'is_terminal': True,
        'is_first': False,
        'is_last': True,
    }

    rlds_step = rlds_utils.to_rlds_step(prev_step, None)

    self.assertEqual(rlds_step, expected_step)

  def test_build_step_with_metadata(self):
    prev_step = step_data.StepData(
        timestep=dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=1,
            observation=2,
            discount=3),
        action=4,
        custom_data={'extra_data': 10})
    step = step_data.StepData(
        timestep=dm_env.TimeStep(
            step_type=dm_env.StepType.LAST, reward=5, observation=6,
            discount=7),
        action=8)

    expected_step = {
        'observation': 2,
        'action': 8,
        'reward': 5,
        'discount': 7,
        'is_terminal': False,
        'is_first': True,
        'is_last': False,
        'extra_data': 10,
    }

    rlds_step = rlds_utils.to_rlds_step(prev_step, step)

    self.assertEqual(rlds_step, expected_step)

  def test_regenerate_splits_noop(self):
    num_episodes = 3
    max_episodes_per_file = 2
    data_dir = self.create_tempdir(name='my_data_dir').full_path
    _ = tfds_backend_testlib.generate_episode_data(
        backend=tfds_backend_testlib.tfds_backend_catch_env(
            data_directory=data_dir,
            max_episodes_per_file=max_episodes_per_file,
            split_name='split'),
        num_episodes=num_episodes)

    builder = tfds.builder_from_directory(data_dir)

    self.assertEqual(list(builder.info.splits.keys()), ['split'])
    self.assertEqual(builder.info.splits['split'].num_examples, 3)
    self.assertEqual(builder.info.splits['split'].num_shards, 2)
    self.assertEqual(builder.info.splits['split'].shard_lengths, [2, 1])

    expected_splits = builder.info.splits

    new_builder = rlds_utils.maybe_recover_last_shard(builder)

    self.assertEqual(new_builder.info.splits, expected_splits)

  def test_regenerate_ds_with_one_split(self):
    num_episodes = 3
    max_episodes_per_file = 5
    data_dir = self.create_tempdir(name='my_data_dir').full_path
    _ = tfds_backend_testlib.generate_episode_data(
        backend=tfds_backend_testlib.tfds_backend_catch_env(
            data_directory=data_dir,
            max_episodes_per_file=max_episodes_per_file,
            split_name='split'),
        num_episodes=num_episodes)

    builder = tfds.builder_from_directory(data_dir)
    expected_splits = builder.info.splits
    # Remove info from the metadata
    builder.info.set_splits(
        tfds.core.splits.SplitDict([
            tfds.core.SplitInfo(
                name='split',
                shard_lengths=[],
                num_bytes=0,
                filename_template=tfds.core.ShardedFileTemplate(
                    dataset_name=builder.name,
                    split='split',
                    filetype_suffix='tfrecord',
                    data_dir=data_dir,
                    template='{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_INDEX}',
                ))
        ]))
    builder.info.write_to_directory(data_dir)

    new_builder = rlds_utils.maybe_recover_last_shard(builder)

    self.assertEqual(
        list(new_builder.info.splits.keys()), list(expected_splits.keys()))
    self.assertEqual(new_builder.info.splits['split'].num_examples,
                     expected_splits['split'].num_examples)
    self.assertEqual(new_builder.info.splits['split'].num_shards,
                     expected_splits['split'].num_shards)
    self.assertEqual(new_builder.info.splits['split'].shard_lengths,
                     expected_splits['split'].shard_lengths)
    self.assertEqual(new_builder.info.splits['split'].num_bytes,
                     expected_splits['split'].num_bytes)

  def test_regenerate_ds_last_split(self):
    num_episodes = 3
    max_episodes_per_file = 2
    data_dir = self.create_tempdir(name='my_data_dir').full_path
    _ = tfds_backend_testlib.generate_episode_data(
        backend=tfds_backend_testlib.tfds_backend_catch_env(
            data_directory=data_dir,
            max_episodes_per_file=max_episodes_per_file,
            split_name='split'),
        num_episodes=num_episodes)

    builder = tfds.builder_from_directory(data_dir)
    expected_splits = builder.info.splits
    # Remove info from the metadata
    # Since we don't know how many bytes each shard has, we let it as it was.
    # We check later that the number of bytes increased.
    builder.info.set_splits(
        tfds.core.splits.SplitDict([
            tfds.core.SplitInfo(
                name='split',
                shard_lengths=[expected_splits['split'].shard_lengths[0]],
                num_bytes=expected_splits['split'].num_bytes,
                filename_template=tfds.core.ShardedFileTemplate(
                    dataset_name=builder.name,
                    split='split',
                    filetype_suffix='tfrecord',
                    data_dir=data_dir,
                    template='{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_INDEX}',
                ),
            )
        ]))
    builder.info.write_to_directory(data_dir)

    new_builder = rlds_utils.maybe_recover_last_shard(builder)

    self.assertEqual(
        list(new_builder.info.splits.keys()), list(expected_splits.keys()))
    self.assertEqual(new_builder.info.splits['split'].num_examples,
                     expected_splits['split'].num_examples)
    self.assertEqual(new_builder.info.splits['split'].num_shards,
                     expected_splits['split'].num_shards)
    self.assertEqual(new_builder.info.splits['split'].shard_lengths,
                     expected_splits['split'].shard_lengths)
    # We don't know how many bytes are accounted to each episode, so we check
    # that the new number of bytes is larger.
    self.assertLess(expected_splits['split'].num_bytes,
                    new_builder.info.splits['split'].num_bytes)


if __name__ == '__main__':
  absltest.main()
