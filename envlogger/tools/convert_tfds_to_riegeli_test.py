# coding=utf-8
# Copyright 2026 DeepMind Technologies Limited..
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

"""Tests for converting TFDS datasets to EnvLogger Riegeli format."""

from unittest import mock

from absl import flags
from absl.testing import absltest
import dm_env
from envlogger import reader
from envlogger.tools import convert_tfds_to_riegeli
import numpy as np
import tensorflow as tf


def _create_dummy_rlds_dataset(
    is_terminal: bool = True, include_custom_data: bool = False
) -> tf.data.Dataset:
  """Creates a small dummy RLDS episode dataset."""

  def episode_generator():
    steps_dict = {
        'observation': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        'action': np.array([[0], [1]], dtype=np.int32),
        'reward': np.array([0.0, 1.0], dtype=np.float32),
        'discount': np.array([1.0, 0.5], dtype=np.float32),
        'is_first': np.array([True, False]),
        'is_last': np.array([False, True]),
        'is_terminal': np.array([False, is_terminal]),
    }
    if include_custom_data:
      steps_dict['custom_feature'] = np.array([10, 20], dtype=np.int32)

    steps_ds = tf.data.Dataset.from_tensor_slices(steps_dict)
    yield {'steps': steps_ds, 'episode_id': np.int64(42)}

  spec_dict = {
      'observation': tf.TensorSpec(shape=(2,), dtype=tf.float32),
      'action': tf.TensorSpec(shape=(1,), dtype=tf.int32),
      'reward': tf.TensorSpec(shape=(), dtype=tf.float32),
      'discount': tf.TensorSpec(shape=(), dtype=tf.float32),
      'is_first': tf.TensorSpec(shape=(), dtype=tf.bool),
      'is_last': tf.TensorSpec(shape=(), dtype=tf.bool),
      'is_terminal': tf.TensorSpec(shape=(), dtype=tf.bool),
  }
  if include_custom_data:
    spec_dict['custom_feature'] = tf.TensorSpec(shape=(), dtype=tf.int32)

  spec = {
      'steps': tf.data.DatasetSpec(spec_dict),
      'episode_id': tf.TensorSpec(shape=(), dtype=tf.int64),
  }
  return tf.data.Dataset.from_generator(
      episode_generator, output_signature=spec
  )


class ConvertTfdsToRiegeliTest(absltest.TestCase):

  def test_convert_tf_dataset_to_riegeli(self):
    output_dir = self.create_tempdir().full_path
    episodes_ds = _create_dummy_rlds_dataset()

    convert_tfds_to_riegeli.convert_tf_dataset_to_riegeli(
        episodes_dataset=episodes_ds,
        output_dir=output_dir,
    )

    with reader.Reader(data_directory=output_dir) as r:
      episodes = list(r.episodes)
      self.assertLen(episodes, 1)
      self.assertLen(episodes[0], 2)

  def test_convert_tf_dataset_with_custom_data_and_truncation(self):
    output_dir = self.create_tempdir().full_path
    episodes_ds = _create_dummy_rlds_dataset(
        is_terminal=False, include_custom_data=True
    )

    convert_tfds_to_riegeli.convert_tf_dataset_to_riegeli(
        episodes_dataset=episodes_ds,
        output_dir=output_dir,
    )

    with reader.Reader(data_directory=output_dir) as r:
      episodes = list(r.episodes)
      self.assertLen(episodes, 1)
      self.assertLen(episodes[0], 2)
      self.assertEqual(r.episode_metadata()[0], {'episode_id': 42})
      step_custom_data = episodes[0][0].custom_data
      self.assertIn('custom_feature', step_custom_data)
      self.assertEqual(step_custom_data['custom_feature'], 10)

  def test_empty_dataset_raises_value_error(self):
    empty_ds = tf.data.Dataset.from_tensor_slices([])
    with self.assertRaises(ValueError):
      convert_tfds_to_riegeli.convert_tf_dataset_to_riegeli(
          episodes_dataset=empty_ds,
          output_dir=self.create_tempdir().full_path,
      )

  @mock.patch.object(convert_tfds_to_riegeli, 'tfds')
  def test_main_cli(self, mock_tfds):
    output_dir = self.create_tempdir().full_path
    episodes_ds = _create_dummy_rlds_dataset()
    mock_tfds.load.return_value = episodes_ds

    flags.FLAGS([
        'convert_tfds_to_riegeli',
        f'--output_dir={output_dir}',
        '--tfds_name=dummy_dataset',
    ])
    convert_tfds_to_riegeli.main(['convert_tfds_to_riegeli'])

    with reader.Reader(data_directory=output_dir) as r:
      self.assertLen(list(r.episodes), 1)


class ReplayEnvironmentTest(absltest.TestCase):

  def test_replay_environment_lifecycle(self):
    spec = dm_env.specs.Array(shape=(2,), dtype=np.float32)
    env = convert_tfds_to_riegeli.ReplayEnvironment(
        observation_spec=spec,
        reward_spec=spec,
        discount_spec=spec,
        action_spec=spec,
    )
    self.assertEqual(env.observation_spec(), spec)
    self.assertEqual(env.reward_spec(), spec)
    self.assertEqual(env.discount_spec(), spec)
    self.assertEqual(env.action_spec(), spec)

    with self.assertRaises(ValueError):
      env.reset()
    with self.assertRaises(ValueError):
      env.step(action=0)

    t0 = dm_env.restart(np.array([1.0, 2.0], dtype=np.float32))
    t1 = dm_env.termination(
        np.float32(1.0), np.array([3.0, 4.0], dtype=np.float32)
    )
    env.set_episode(timesteps=[t0, t1], custom_data=[{'a': 1}, {'a': 2}])

    res_t0 = env.reset()
    self.assertIs(res_t0, t0)
    self.assertEqual(env.get_custom_data(), {'a': 1})

    res_t1 = env.step(action=0)
    self.assertIs(res_t1, t1)
    self.assertEqual(env.get_custom_data(), {'a': 2})

    with self.assertRaises(ValueError):
      env.step(action=0)

    env.close()


class CreateTimestepTest(absltest.TestCase):

  def test_create_timestep_first_step(self):
    steps = {
        'observation': np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        'reward': np.array([10.0, 20.0, 30.0], dtype=np.float32),
        'discount': np.array([1.0, 1.0, 1.0], dtype=np.float32),
        'is_terminal': np.array([False, False, True]),
    }
    ts = convert_tfds_to_riegeli._create_timestep(
        steps, step_index=0, total_steps=3, discount_dtype=np.float32
    )
    self.assertEqual(ts.step_type, dm_env.StepType.FIRST)
    self.assertIsNone(ts.reward)
    self.assertIsNone(ts.discount)
    np.testing.assert_array_equal(ts.observation, [1.0])

  def test_create_timestep_middle_step(self):
    steps = {
        'observation': np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        'reward': np.array([10.0, 20.0, 30.0], dtype=np.float32),
        'discount': np.array([0.9, 0.8, 0.7], dtype=np.float32),
        'is_terminal': np.array([False, False, True]),
    }
    ts = convert_tfds_to_riegeli._create_timestep(
        steps, step_index=1, total_steps=3, discount_dtype=np.float32
    )
    self.assertEqual(ts.step_type, dm_env.StepType.MID)
    self.assertEqual(ts.reward, 10.0)
    self.assertEqual(ts.discount, 0.9)
    np.testing.assert_array_equal(ts.observation, [2.0])

  def test_create_timestep_last_step_terminal(self):
    steps = {
        'observation': np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        'reward': np.array([10.0, 20.0, 30.0], dtype=np.float32),
        'discount': np.array([0.9, 0.8, 0.7], dtype=np.float32),
        'is_terminal': np.array([False, False, True]),
    }
    ts = convert_tfds_to_riegeli._create_timestep(
        steps, step_index=2, total_steps=3, discount_dtype=np.float32
    )
    self.assertEqual(ts.step_type, dm_env.StepType.LAST)
    self.assertEqual(ts.reward, 20.0)
    self.assertEqual(ts.discount, 0.0)
    np.testing.assert_array_equal(ts.observation, [3.0])

  def test_create_timestep_last_step_truncation(self):
    steps = {
        'observation': np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        'reward': np.array([10.0, 20.0, 30.0], dtype=np.float32),
        'discount': np.array([0.9, 0.8, 0.7], dtype=np.float32),
        'is_terminal': np.array([False, False, False]),
    }
    ts = convert_tfds_to_riegeli._create_timestep(
        steps, step_index=2, total_steps=3, discount_dtype=np.float32
    )
    self.assertEqual(ts.step_type, dm_env.StepType.LAST)
    self.assertEqual(ts.reward, 20.0)
    self.assertEqual(ts.discount, 0.8)
    np.testing.assert_array_equal(ts.observation, [3.0])


if __name__ == '__main__':
  absltest.main()
