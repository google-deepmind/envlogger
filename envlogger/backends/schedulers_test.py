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

"""Tests for schedulers."""

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from envlogger import step_data
from envlogger.backends import schedulers
import numpy as np


def _create_episode(num_transitions: int):
  """Creates an episode with `num_transition` transitions."""
  episode = [step_data.StepData(dm_env.restart(observation=None), action=None)]
  for _ in range(num_transitions):
    episode.append(
        step_data.StepData(
            dm_env.transition(observation=None, reward=None), action=None))
  episode.append(
      step_data.StepData(
          dm_env.termination(observation=None, reward=None), action=None))
  return episode


class DefaultSchedulersTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('negative_interval', -1),
      ('zero_interval', 0),
  )
  def test_n_steps_invalid_args(self, step_interval):
    """n_step_scheduler should raise an error if given invalid intervals."""
    self.assertRaises(
        ValueError,
        schedulers.n_step_scheduler,
        step_interval=step_interval)

  def test_n_steps_interval_1(self):
    """n_step_scheduler should always return True if interval is 1."""
    scheduler = schedulers.n_step_scheduler(step_interval=1)
    for _ in range(100):
      self.assertTrue(scheduler(None))

  def test_n_steps_interval_3(self):
    """n_step_scheduler should return True only every 3 steps."""
    n = 3
    scheduler = schedulers.n_step_scheduler(step_interval=n)
    self.assertTrue(scheduler(None))
    self.assertFalse(scheduler(None))
    self.assertFalse(scheduler(None))
    self.assertTrue(scheduler(None))
    self.assertFalse(scheduler(None))
    self.assertFalse(scheduler(None))

  def test_n_steps_interval_n(self):
    """n_step_scheduler should return True only every n steps."""
    for _ in range(10):
      n = np.random.randint(1, 50)
      scheduler = schedulers.n_step_scheduler(step_interval=n)
      for i in range(0, n * 10):
        if i % n == 0:
          self.assertTrue(scheduler(None))
        else:
          self.assertFalse(scheduler(None))

  @parameterized.named_parameters(
      ('negative_probability', -1),
      ('greater_than_1_probability', 1.01),
  )
  def test_bernoulli_steps_invalid_args(self, keep_probability):
    """bernoulli_scheduler should raise an error with negative probabilities."""
    self.assertRaises(
        ValueError,
        schedulers.bernoulli_step_scheduler,
        keep_probability=keep_probability)

  def test_bernoulli_steps_probability_0(self):
    """bernoulli_step_scheduler should always False if given probability 0.0."""
    scheduler = schedulers.bernoulli_step_scheduler(keep_probability=0)
    for _ in range(100):
      self.assertFalse(scheduler(None))

  def test_bernoulli_steps_probability_1(self):
    """bernoulli_step_scheduler should always False if given probability 1.0."""
    scheduler = schedulers.bernoulli_step_scheduler(keep_probability=1)
    for _ in range(100):
      self.assertTrue(scheduler(None))

  def test_bernoulli_steps_probability_1pct(self):
    """bernoulli_step_scheduler should return more False than True with p=0.01."""
    scheduler = schedulers.bernoulli_step_scheduler(
        keep_probability=0.01)
    num_true = 0
    for _ in range(1000):
      num_true += scheduler(None)
    num_false = 1000 - num_true
    self.assertGreater(num_false, num_true)

  def test_bernoulli_steps_probability_99pct(self):
    """bernoulli_step_scheduler should return more True than False with p=0.99."""
    scheduler = schedulers.bernoulli_step_scheduler(
        keep_probability=0.99)
    num_true = 0
    for _ in range(1000):
      num_true += scheduler(None)
    num_false = 1000 - num_true
    self.assertGreater(num_true, num_false)

  @parameterized.named_parameters(
      ('negative_interval', -1),
      ('zero_interval', 0),
  )
  def test_n_episode_invalid_args(self, episode_interval):
    """n_episode_scheduler should raise an error if given invalid intervals."""
    self.assertRaises(
        ValueError,
        schedulers.n_episode_scheduler,
        episode_interval=episode_interval)

  def test_n_episode_interval_1(self):
    """n_episode_scheduler should always return True if interval is 1."""
    scheduler = schedulers.n_episode_scheduler(episode_interval=1)
    for _ in range(100):
      for timestep in _create_episode(num_transitions=np.random.randint(100)):
        self.assertTrue(scheduler(timestep))

  def test_n_episode_interval_2(self):
    """n_episode_scheduler should return True every other episode."""
    scheduler = schedulers.n_episode_scheduler(episode_interval=2)
    for _ in range(100):
      for timestep in _create_episode(num_transitions=np.random.randint(100)):
        self.assertTrue(scheduler(timestep))
      for timestep in _create_episode(num_transitions=np.random.randint(100)):
        self.assertFalse(scheduler(timestep))

  def test_n_episode_interval_n(self):
    """n_episode_scheduler should return True only every n episodes."""
    for _ in range(10):
      n = np.random.randint(1, 50)
      scheduler = schedulers.n_episode_scheduler(episode_interval=n)
      for i in range(0, n * 10):
        if i % n == 0:
          for timestep in _create_episode(
              num_transitions=np.random.randint(100)):
            self.assertTrue(scheduler(timestep))
        else:
          for timestep in _create_episode(
              num_transitions=np.random.randint(100)):
            self.assertFalse(scheduler(timestep))

  @parameterized.named_parameters(
      ('negative_probability', -1),
      ('greater_than_1_probability', 1.01),
  )
  def test_bernoulli_episodes_invalid_args(self, keep_probability):
    """bernoulli_scheduler should raise an error with negative probabilities."""
    self.assertRaises(
        ValueError,
        schedulers.bernoulli_episode_scheduler,
        keep_probability=keep_probability)

  def test_bernoulli_episodes_probability_0(self):
    """bernoulli_episode_scheduler should always False if given probability 0.0."""
    scheduler = schedulers.bernoulli_episode_scheduler(
        keep_probability=0.0)
    for _ in range(100):
      for timestep in _create_episode(num_transitions=np.random.randint(100)):
        self.assertFalse(scheduler(timestep))

  def test_bernoulli_episodes_probability_1(self):
    """bernoulli_episode_scheduler should always False if given probability 1.0."""
    scheduler = schedulers.bernoulli_episode_scheduler(
        keep_probability=1.0)
    for _ in range(100):
      for timestep in _create_episode(num_transitions=np.random.randint(100)):
        self.assertTrue(scheduler(timestep))

  def test_bernoulli_episodes_probability_1pct(self):
    """bernoulli_episode_scheduler should return more False with p=0.01."""
    scheduler = schedulers.bernoulli_episode_scheduler(
        keep_probability=0.01)
    num_true = 0
    num_false = 0
    for _ in range(1000):
      for timestep in _create_episode(num_transitions=np.random.randint(100)):
        outcome = scheduler(timestep)
        if outcome:
          num_true += 1
        else:
          num_false += 1
    self.assertGreater(num_false, num_true)

  def test_bernoulli_episodes_probability_99pct(self):
    """bernoulli_episode_scheduler should return more True with p=0.99."""
    scheduler = schedulers.bernoulli_episode_scheduler(
        keep_probability=0.99)
    num_true = 0
    num_false = 0
    for _ in range(1000):
      for timestep in _create_episode(num_transitions=np.random.randint(100)):
        outcome = scheduler(timestep)
        if outcome:
          num_true += 1
        else:
          num_false += 1
    self.assertGreater(num_true, num_false)

  @parameterized.named_parameters(
      ('empty_list', []),
      ('empty_ndarray', np.array([], dtype=np.int64)),
  )
  def test_list_steps_empty_steps(self, desired_steps):
    """list_step_scheduler should raise an error if given invalid steps."""
    self.assertRaises(
        ValueError,
        schedulers.list_step_scheduler,
        desired_steps=desired_steps)

  def test_list_np_array_wrong_type(self):
    """list_step_scheduler should raise an error if given invalid steps."""
    self.assertRaises(
        TypeError,
        schedulers.list_step_scheduler,
        desired_steps=np.array([1.0, 10.0, 100.0], dtype=np.float32))

  def test_list_steps_single_item(self):
    """list_step_scheduler should return True if step is in `desired_steps`."""
    scheduler = schedulers.list_step_scheduler(desired_steps=[3])
    self.assertFalse(scheduler(None))
    self.assertFalse(scheduler(None))
    self.assertFalse(scheduler(None))
    self.assertTrue(scheduler(None))  # 4th step should be True.
    for _ in range(100):
      self.assertFalse(scheduler(None))

  def test_list_steps_first_10(self):
    """list_step_scheduler should return True if step is in `desired_steps`."""
    scheduler = schedulers.list_step_scheduler(
        desired_steps=list(range(10)))
    for _ in range(10):  # First 10 steps should be True.
      self.assertTrue(scheduler(None))
    for _ in range(100):
      self.assertFalse(scheduler(None))

  def test_list_steps_logspace(self):
    """list_step_scheduler should return True if step is in `desired_steps`."""
    desired_steps = np.logspace(
        start=0, stop=3, num=10, base=10.0).astype(np.int32) - 1
    # At this point: desired_steps = [0, 1, 3, 9, 20, 45, 99, 214, 463, 999]
    scheduler = schedulers.list_step_scheduler(
        desired_steps=desired_steps)
    for i in range(1000):
      if i in [0, 1, 3, 9, 20, 45, 99, 214, 463, 999]:
        self.assertTrue(scheduler(None))
      else:
        self.assertFalse(scheduler(None))

  @parameterized.named_parameters(
      ('empty_list', []),
      ('empty_ndarray', np.array([], dtype=np.int64)),
  )
  def test_list_empty_episodes(self, desired_episodes):
    """list_episode_scheduler should raise an error if given invalid episodes."""
    self.assertRaises(
        ValueError,
        schedulers.list_episode_scheduler,
        desired_episodes=desired_episodes)

  def test_list_episodes_np_array_wrong_type(self):
    """list_episode_scheduler should raise an error if given invalid episodes."""
    self.assertRaises(
        TypeError,
        schedulers.list_episode_scheduler,
        desired_episodes=np.array([1.0, 10.0, 100.0], dtype=np.float32))

  def test_list_episodes_single_item(self):
    """Scheduler should return True if episode is in `desired_episodes`."""
    scheduler = schedulers.list_episode_scheduler(desired_episodes=[3])
    for timestep in _create_episode(num_transitions=np.random.randint(100)):
      self.assertFalse(scheduler(timestep))
    for timestep in _create_episode(num_transitions=np.random.randint(100)):
      self.assertFalse(scheduler(timestep))
    for timestep in _create_episode(num_transitions=np.random.randint(100)):
      self.assertFalse(scheduler(timestep))
    for timestep in _create_episode(
        num_transitions=np.random.randint(100)):  # 4th episode should be True.
      self.assertTrue(scheduler(timestep))
    for _ in range(100):
      for timestep in _create_episode(num_transitions=np.random.randint(100)):
        self.assertFalse(scheduler(timestep))

  def test_list_episodes_first_10(self):
    """Scheduler should return True if episode is in `desired_episodes`."""
    scheduler = schedulers.list_episode_scheduler(
        desired_episodes=list(range(10)))
    for _ in range(10):  # First 10 episodes should be True.
      for timestep in _create_episode(num_transitions=np.random.randint(100)):
        self.assertTrue(scheduler(timestep))
    for _ in range(100):
      for timestep in _create_episode(num_transitions=np.random.randint(100)):
        self.assertFalse(scheduler(timestep))

  def test_list_episodes_logspace(self):
    """Scheduler should return True if episode is in `desired_episodes`."""
    desired_episodes = np.logspace(
        start=0, stop=3, num=10, base=10.0).astype(np.int32) - 1
    # At this point: desired_episodes = [0, 1, 3, 9, 20, 45, 99, 214, 463, 999]
    scheduler = schedulers.list_episode_scheduler(
        desired_episodes=desired_episodes)
    for i in range(1000):
      if i in [0, 1, 3, 9, 20, 45, 99, 214, 463, 999]:
        for timestep in _create_episode(num_transitions=np.random.randint(100)):
          self.assertTrue(scheduler(timestep))
      else:
        for timestep in _create_episode(num_transitions=np.random.randint(100)):
          self.assertFalse(scheduler(timestep))


if __name__ == '__main__':
  absltest.main()
