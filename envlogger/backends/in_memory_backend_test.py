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

"""Tests for in_memory_backend."""

import operator
from typing import List, Optional, Tuple

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from envlogger import step_data
from envlogger.backends import in_memory_backend
from envlogger.backends import schedulers
from envlogger.testing import catch_env
import numpy as np


def _collect_episode_data(
    num_episodes: int,
    scheduler: Optional[schedulers.Scheduler] = None
) -> Tuple[List[step_data.StepData], in_memory_backend.InMemoryBackendWriter]:

  logging.info('Creating Catch environment...')
  env = catch_env.Catch()
  logging.info('Done creating Catch environment.')
  backend = in_memory_backend.InMemoryBackendWriter(scheduler=scheduler)
  logging.info('Training a random agent for %r episodes...', num_episodes)
  num_actions = 3
  episodes_data = []
  for _ in range(num_episodes):
    timestep = env.reset()
    data = step_data.StepData(timestep, None, None)
    episodes_data.append(data)
    backend.record_step(data, is_new_episode=True)

    while not timestep.last():
      action = np.random.choice(num_actions)
      timestep = env.step(action)
      data = step_data.StepData(timestep, action, None)
      episodes_data.append(data)
      backend.record_step(data, is_new_episode=False)
  logging.info('Done training a random agent for %r episodes.', num_episodes)
  env.close()
  return episodes_data, backend


class InMemoryBackendTest(parameterized.TestCase):

  def _validate_steps(self,
                      actual_steps,
                      expected_steps,
                      num_episodes,
                      num_steps_per_episode=10):
    num_steps = num_episodes * num_steps_per_episode
    self.assertLen(actual_steps, num_steps)
    self.assertLen(expected_steps, num_steps)
    for actual_step, expected_step in zip(actual_steps, expected_steps):
      np.testing.assert_equal(actual_step, expected_step)

  def test_steps_round_trip(self):
    num_episodes = 3
    expected_steps, backend = _collect_episode_data(num_episodes)
    actual_steps = list(in_memory_backend.InMemoryBackendReader(backend).steps)
    self._validate_steps(actual_steps, expected_steps, num_episodes)

  def test_episodes_round_trip(self):
    num_episodes = 3
    num_steps_per_episode = 10
    expected_steps, backend = _collect_episode_data(num_episodes)
    for episode_index, episode in enumerate(
        in_memory_backend.InMemoryBackendReader(backend).episodes):
      episode_actual_steps = list(episode)
      episode_expected_steps = expected_steps[episode_index *
                                              num_steps_per_episode:
                                              (episode_index + 1) *
                                              num_steps_per_episode]
      self._validate_steps(
          episode_actual_steps, episode_expected_steps, num_episodes=1)

  def test_scheduler(self):
    num_episodes = 2
    step_interval = 2
    scheduler = schedulers.n_step_scheduler(step_interval=step_interval)
    expected_steps, backend = _collect_episode_data(num_episodes, scheduler)
    expected_steps = [
        step for i, step in enumerate(expected_steps) if i % step_interval == 0
    ]
    actual_steps = list(in_memory_backend.InMemoryBackendReader(backend).steps)
    self._validate_steps(
        actual_steps,
        expected_steps,
        num_episodes,
        num_steps_per_episode=10 / step_interval)

  def test_step_negative_indices(self):
    """Ensures that negative step indices are handled correctly."""
    _, backend = _collect_episode_data(num_episodes=6)
    data_reader = in_memory_backend.InMemoryBackendReader(backend)
    np.testing.assert_equal(data_reader.steps[-1],
                            data_reader.steps[len(data_reader.steps) - 1])
    np.testing.assert_equal(data_reader.steps[-len(data_reader.steps)],
                            data_reader.steps[0])

  def test_step_out_of_bounds_indices(self):
    """Ensures that out of bounds step indices are handled correctly."""
    _, backend = _collect_episode_data(num_episodes=6)
    data_reader = in_memory_backend.InMemoryBackendReader(backend)
    self.assertRaises(IndexError, operator.getitem, data_reader.steps,
                      len(data_reader.steps))
    self.assertRaises(IndexError, operator.getitem, data_reader.steps,
                      -len(data_reader.steps) - 1)

  def test_episode_negative_indices(self):
    """Ensures that negative episode indices are handled correctly."""
    _, backend = _collect_episode_data(num_episodes=6)
    data_reader = in_memory_backend.InMemoryBackendReader(backend)
    np.testing.assert_equal(
        data_reader.episodes[-1][:],
        data_reader.episodes[len(data_reader.episodes) - 1][:])
    np.testing.assert_equal(data_reader.episodes[-len(data_reader.episodes)][:],
                            data_reader.episodes[0][:])

  def test_episode_out_of_bounds_indices(self):
    """Ensures that out of bounds episode indices are handled correctly."""
    _, backend = _collect_episode_data(num_episodes=6)
    data_reader = in_memory_backend.InMemoryBackendReader(backend)
    self.assertRaises(IndexError, operator.getitem, data_reader.episodes,
                      len(data_reader.episodes))
    self.assertRaises(IndexError, operator.getitem, data_reader.episodes,
                      -len(data_reader.episodes) - 1)

  def test_episode_step_negative_indices(self):
    """Ensures that negative episode step indices are handled correctly."""
    _, backend = _collect_episode_data(num_episodes=6)
    data_reader = in_memory_backend.InMemoryBackendReader(backend)
    for episode in data_reader.episodes:
      np.testing.assert_equal(episode[-1], episode[len(episode) - 1])
      np.testing.assert_equal(episode[-len(episode)], episode[0])

  def test_episode_step_out_of_bounds_indices(self):
    """Ensures that out of bounds episode step indices are handled correctly."""
    _, backend = _collect_episode_data(num_episodes=6)
    data_reader = in_memory_backend.InMemoryBackendReader(backend)
    for episode in data_reader.episodes:
      self.assertRaises(IndexError, operator.getitem, episode, len(episode))
      self.assertRaises(IndexError, operator.getitem, episode,
                        -len(episode) - 1)


if __name__ == '__main__':
  absltest.main()
