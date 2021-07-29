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

"""EnvLogger Tests."""

import glob
import os
import tempfile
import threading
from typing import List, Optional
import uuid

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from dm_env import specs
from envlogger import environment_logger
from envlogger import reader
from envlogger import step_data
from envlogger.backends import backend_type
from envlogger.backends import in_memory_backend
from envlogger.backends import schedulers
from envlogger.converters import codec
from envlogger.converters import spec_codec
from envlogger.proto import storage_pb2
from envlogger.testing import catch_env
import mock
import numpy as np
import riegeli



class CustomSpecsEnvironment(dm_env.Environment):
  """An Environment that allows us to customize its specs."""

  def __init__(self,
               observation_spec,
               action_spec,
               reward_spec,
               discount_spec,
               episode_length=10):
    self._observation_spec = observation_spec
    self._action_spec = action_spec
    self._reward_spec = reward_spec
    self._discount_spec = discount_spec
    self._episode_length = episode_length
    self._step_counter = 0

  def reset(self):
    self._step_counter = 0
    return dm_env.restart(123)  # Return whatever, we won't check it.

  def step(self, actions):
    self._step_counter += 1
    if self._step_counter >= self._episode_length:
      return dm_env.termination(-1.0, 987)
    return dm_env.transition(1.0, 321)  # Return whatever, we won't check it.

  def discount_spec(self):
    return self._discount_spec

  def reward_spec(self):
    return self._reward_spec

  def observation_spec(self):
    return self._observation_spec

  def action_spec(self):
    return self._action_spec


class RandomDataEnvironment(dm_env.Environment):
  """An Environment that produces random data of a particular shape."""

  def __init__(self, data_size=1000, prob_episode_end=0.01):
    self._data_size = data_size
    self._prob_episode_end = prob_episode_end

  def reset(self):
    return dm_env.restart(self._obs())

  def step(self, actions):
    if np.random.rand() < self._prob_episode_end:
      return dm_env.termination(1.0, self._obs())
    return dm_env.transition(1.0, self._obs())

  def observation_spec(self):
    return specs.Array(shape=(self._data_size,), dtype=np.float32)

  def action_spec(self):
    return specs.Array(shape=(), dtype=np.int32)

  def _obs(self):
    return np.random.rand(self._data_size,)


def _train(env: dm_env.Environment,
           num_episodes: int) -> List[step_data.StepData]:
  logging.info('Training a random agent for %r episodes...', num_episodes)
  num_actions = 3
  episodes_data = []
  for _ in range(num_episodes):
    timestep = env.reset()
    episodes_data.append(step_data.StepData(timestep, None, None))

    while not timestep.last():
      action = np.random.choice(num_actions)
      timestep = env.step(action)
      episodes_data.append(step_data.StepData(timestep, action, None))
  logging.info('Done training a random agent for %r episodes.', num_episodes)
  env.close()
  return episodes_data


class EnvLoggerTest(parameterized.TestCase):

  def setUp(self):
    super(EnvLoggerTest, self).setUp()
    self._temp_dir = tempfile.TemporaryDirectory(
        dir=absltest.get_default_test_tmpdir())
    self.dataset_path = os.path.join(self._temp_dir.name,
                                     'environment_logger_test')
    os.makedirs(self.dataset_path, exist_ok=True)

  def tearDown(self):
    self._temp_dir.cleanup()
    super(EnvLoggerTest, self).tearDown()

  @parameterized.named_parameters(
      ('bare_spec', specs.Array(shape=(1, 2, 3), dtype=np.int8)),
      ('bounded_spec',
       specs.BoundedArray(shape=(4, 5), dtype=np.int8, minimum=10, maximum=50)),
      ('discrete_array_spec', specs.DiscreteArray(num_values=73)),
      ('list_spec', [
          specs.Array(shape=(1, 2, 3), dtype=np.int8),
          specs.Array(shape=(), dtype=np.float64)
      ]), ('tuple_spec',
           (specs.Array(shape=(1,), dtype=np.float32),
            specs.Array(shape=(1, 2, 3), dtype=np.int8))), ('dict_spec', {
                'my_float': specs.Array(shape=(1,), dtype=np.float32),
                'integers': specs.Array(shape=(1, 2, 3), dtype=np.int8),
            }))
  def test_specs_of_different_types_are_supported(self, spec):
    """Ensures that different spec types are supported."""
    env = CustomSpecsEnvironment(
        observation_spec=spec,
        action_spec=spec,
        reward_spec=spec,
        discount_spec=spec)
    env = environment_logger.EnvLogger(
        env,
        data_directory=self.dataset_path,
        backend=backend_type.BackendType.RIEGELI)
    _train(env, num_episodes=1)
    with reader.Reader(self.dataset_path) as data_reader:
      self.assertEqual(data_reader.observation_spec(), spec)
      self.assertEqual(type(data_reader.observation_spec()), type(spec))
      self.assertEqual(data_reader.action_spec(), spec)
      self.assertEqual(type(data_reader.action_spec()), type(spec))
      self.assertEqual(data_reader.reward_spec(), spec)
      self.assertEqual(type(data_reader.reward_spec()), type(spec))
      self.assertEqual(data_reader.discount_spec(), spec)
      self.assertEqual(type(data_reader.discount_spec()), type(spec))

  def test_different_specs_are_actually_different(self):
    """Ensures that different spec types are maintained."""
    spec1 = specs.Array(shape=(1, 2, 3), dtype=np.int8)
    spec2 = specs.Array(shape=(1,), dtype=np.int64)
    spec3 = specs.BoundedArray(
        shape=(4, 5, 6), dtype=np.float32, minimum=10.0, maximum=11.0)
    spec4 = specs.DiscreteArray(num_values=321, dtype=np.int16)
    env = CustomSpecsEnvironment(
        observation_spec=spec1,
        action_spec=spec2,
        reward_spec=spec3,
        discount_spec=spec4)
    env = environment_logger.EnvLogger(
        env,
        data_directory=self.dataset_path,
        backend=backend_type.BackendType.RIEGELI)
    _train(env, num_episodes=1)
    with reader.Reader(self.dataset_path) as data_reader:
      self.assertEqual(data_reader.observation_spec(), spec1)
      self.assertEqual(data_reader.action_spec(), spec2)
      self.assertEqual(data_reader.reward_spec(), spec3)
      self.assertEqual(data_reader.discount_spec(), spec4)
      self.assertNotEqual(data_reader.observation_spec(),
                          data_reader.action_spec())
      self.assertNotEqual(data_reader.observation_spec(),
                          data_reader.reward_spec())
      self.assertNotEqual(data_reader.observation_spec(),
                          data_reader.discount_spec())
      self.assertNotEqual(data_reader.action_spec(), data_reader.reward_spec())
      self.assertNotEqual(data_reader.action_spec(),
                          data_reader.discount_spec())
      self.assertNotEqual(data_reader.reward_spec(),
                          data_reader.discount_spec())

  def test_metadata_is_available(self):
    """Ensures that if `metadata` is passed, it can be read."""
    env = catch_env.Catch()
    env = environment_logger.EnvLogger(
        env,
        data_directory=self.dataset_path,
        metadata={'do_not_forget_me': 'i am important!'},
        max_episodes_per_file=973,
        writer_options='transpose,brotli:1,chunk_size:50M',
        backend=backend_type.BackendType.RIEGELI)
    _train(env, num_episodes=1)
    with reader.Reader(data_directory=self.dataset_path) as data_reader:
      metadata = data_reader.metadata()
      environment_specs = metadata.pop('environment_specs')
      for k, v in spec_codec.encode_environment_specs(env).items():
        for spec_name, spec_value in v.items():
          if isinstance(spec_value, np.ndarray):
            np.testing.assert_array_equal(
                environment_specs[k][spec_name], spec_value)
          else:
            self.assertEqual(environment_specs[k][spec_name], spec_value)
      self.assertDictEqual(data_reader.metadata(),
                           {'do_not_forget_me': 'i am important!'})

  def test_data_reader_get_timestep(self):
    """Ensures that we can fetch single timesteps from a Reader."""
    num_episodes = 13
    num_steps_per_episode = 10
    num_steps = num_episodes * num_steps_per_episode
    env = catch_env.Catch()
    backend = in_memory_backend.InMemoryBackendWriter()
    env = environment_logger.EnvLogger(
        env, data_directory=self.dataset_path, backend=backend)
    expected_data = _train(env, num_episodes=num_episodes)
    self.assertLen(
        expected_data,
        num_steps,
        msg=(f'We expect {num_steps} steps when running an actor for '
             f'{num_episodes} episodes of {num_steps_per_episode} steps each.'))

    data_reader = in_memory_backend.InMemoryBackendReader(backend)
    self.assertLen(
        data_reader.steps,
        num_steps,
        msg=(f'We expect {num_steps} steps when running an actor for '
             f'{num_episodes} episodes of {num_steps_per_episode} steps each.'))
    # All 130 steps should be accessible with __getitem__().
    for i in range(num_steps):
      np.testing.assert_equal(data_reader.steps[i], expected_data[i])

    # All 130 steps should be accessible with __iter__().
    step_index = 0
    for step_index, (actual, expected) in enumerate(
        zip(data_reader.steps, expected_data)):
      np.testing.assert_equal(actual, expected)
    self.assertEqual(step_index, num_steps - 1)  # step_index is 0-based

  def test_step_fn(self):
    """Checks that `step_fn` produces expected custom data."""
    v = np.random.randint(1000)
    expected_custom_data = list(range(v + 1, v + 1 + 20))

    def increment_fn(unused_timestep, unused_action, unused_env):
      """A function that increments `v` then returns it."""
      nonlocal v
      v += 1
      return v

    env = catch_env.Catch()
    env = environment_logger.EnvLogger(
        env,
        data_directory=self.dataset_path,
        step_fn=increment_fn,
        backend=backend_type.BackendType.RIEGELI)
    _train(env, num_episodes=2)

    actual_data = []
    with reader.Reader(self.dataset_path) as data_reader:
      tag_data = list(data_reader.steps)
      actual_data += tag_data
      self.assertLen(
          data_reader.steps,
          20,
          msg='Expected 20 steps in total from an actor running 2 episodes '
          'of 10 steps each.')
      self.assertLen(
          data_reader.episodes,
          2,
          msg='Expected 2 episodes in total from an actor running 2 '
          'episodes.')
    np.testing.assert_equal([x.custom_data for x in actual_data],
                            expected_custom_data)

  def test_episode_fn(self):
    """Checks that `episode_fn` produces expected custom data."""
    v = 100

    def increment_fn(timestep, unused_action, unused_env) -> Optional[int]:
      """Increments `v` on the last timestep and returns it in even episodes."""
      nonlocal v
      if timestep.first():
        v += 1
      return np.int32(v) if v % 2 == 0 else None

    env = catch_env.Catch()
    env = environment_logger.EnvLogger(
        env,
        data_directory=self.dataset_path,
        episode_fn=increment_fn,
        backend=backend_type.BackendType.RIEGELI)
    _train(env, num_episodes=11)

    actual_metadata = []
    with reader.Reader(self.dataset_path) as data_reader:
      for metadata in data_reader.episode_metadata():
        actual_metadata.append(metadata)
    self.assertEqual(
        actual_metadata,
        [None, 102, None, 104, None, 106, None, 108, None, 110, None])

  def test_truncated_trajectory(self):
    """Ensures that the reader handles a truncated trajectory."""
    env = catch_env.Catch()
    env = environment_logger.EnvLogger(
        env,
        data_directory=self.dataset_path,
        max_episodes_per_file=2,
        backend=backend_type.BackendType.RIEGELI)
    expected_data = _train(env, num_episodes=5)
    # Remove the last 10 steps from the last episode to simulate a truncated
    # trajectory.
    expected_data = expected_data[:-10]

    # Truncate the last timestamp dir of the first (and only) actor.
    first_actor = self.dataset_path
    dir_contents = os.listdir(first_actor)
    dir_contents = [
        d for d in dir_contents if os.path.isdir(os.path.join(first_actor, d))
    ]
    dir_contents = sorted(dir_contents)
    last_timestamp_dir = os.path.join(first_actor, dir_contents[-1])
    for fname in [
        'steps.riegeli', 'step_offsets.riegeli', 'episode_metadata.riegeli',
        'episode_index.riegeli'
    ]:
      with open(os.path.join(last_timestamp_dir, fname), 'w') as f:
        f.truncate()

    actual_data = []
    with reader.Reader(first_actor) as data_reader:
      tag_data = list(data_reader.steps)
      actual_data += tag_data
      self.assertLen(
          data_reader.steps,
          40,
          msg='Expected 40 steps in total from an actor running 4 episodes '
          'of 10 steps each (last shard should not be included).')
      self.assertLen(
          data_reader.episodes,
          4,
          msg='Expected 4 episodes in total from an actor running 4 '
          'episodes (last shard should not be included).')
    np.testing.assert_equal(actual_data, expected_data)

  def test_episode_starts_monotonically_increasing(self):
    """Ensures that all episode starts form an increasing sequence."""
    env = RandomDataEnvironment(data_size=100, prob_episode_end=0.01)
    env = environment_logger.EnvLogger(
        env,
        data_directory=self.dataset_path,
        max_episodes_per_file=10_000_000_000,
        flush_scheduler=schedulers.bernoulli_step_scheduler(1.0 / 13),
        backend=backend_type.BackendType.RIEGELI)
    _train(env, num_episodes=100)

    actor = self.dataset_path
    dir_contents = os.listdir(actor)
    dir_contents = [
        d for d in dir_contents if os.path.isdir(os.path.join(actor, d))
    ]
    for d in dir_contents:
      timestamp_dir = os.path.join(actor, d)
      episode_index_file = os.path.join(timestamp_dir,
                                        'episode_index.riegeli')
      with riegeli.RecordReader(open(episode_index_file,
                                     'rb')) as riegeli_reader:
        previous = None
        for record in riegeli_reader.read_messages(storage_pb2.Datum):
          decoded = codec.decode_datum(record)
          for episode_start, _ in decoded:
            if previous is None:
              continue

            self.assertGreater(episode_start, previous)
            previous = episode_start
    with reader.Reader(actor) as tagreader:
      for episode in tagreader.episodes:
        self.assertGreaterEqual(len(episode), 0)
      for episode_metadata in tagreader.episode_metadata():
        self.assertIsNone(episode_metadata)

  def test_logger_close(self):
    """Ensures that `.close()` is idempotent and can be called multiple times."""
    env = catch_env.Catch()
    env = environment_logger.EnvLogger(
        env,
        data_directory=self.dataset_path,
        backend=backend_type.BackendType.RIEGELI,
    )
    _train(env, num_episodes=1)

    for _ in range(10):
      env.close()  # Can be called multiple times.

  def test_logger_as_context(self):
    """Ensures that EnvLogger can be used as a context."""
    env = catch_env.Catch()
    with environment_logger.EnvLogger(
        env,
        data_directory=self.dataset_path,
        backend=backend_type.BackendType.RIEGELI) as env:
      _train(env, num_episodes=1)



if __name__ == '__main__':
  absltest.main()
