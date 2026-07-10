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

import datetime

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
    """NStepScheduler should raise an error if given invalid intervals."""
    self.assertRaises(
        ValueError, schedulers.NStepScheduler, step_interval=step_interval)

  def test_n_steps_interval_1(self):
    """NStepScheduler should always return True if interval is 1."""
    scheduler = schedulers.NStepScheduler(step_interval=1)
    for _ in range(100):
      self.assertTrue(scheduler(None))

  def test_n_steps_interval_3(self):
    """NStepScheduler should return True only every 3 steps."""
    n = 3
    scheduler = schedulers.NStepScheduler(step_interval=n)
    self.assertTrue(scheduler(None))
    self.assertFalse(scheduler(None))
    self.assertFalse(scheduler(None))
    self.assertTrue(scheduler(None))
    self.assertFalse(scheduler(None))
    self.assertFalse(scheduler(None))

  def test_n_steps_interval_n(self):
    """NStepScheduler should return True only every n steps."""
    for _ in range(10):
      n = np.random.randint(1, 50)
      scheduler = schedulers.NStepScheduler(step_interval=n)
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
    """The scheduler should raise an error with negative probabilities."""
    self.assertRaises(
        ValueError,
        schedulers.BernoulliStepScheduler,
        keep_probability=keep_probability)

  def test_bernoulli_steps_probability_0(self):
    """BernoulliStepScheduler should return False if given probability 0.0."""
    scheduler = schedulers.BernoulliStepScheduler(keep_probability=0)
    for _ in range(100):
      self.assertFalse(scheduler(None))

  def test_bernoulli_steps_probability_1(self):
    """BernoulliStepScheduler should return True if given probability 1.0."""
    scheduler = schedulers.BernoulliStepScheduler(keep_probability=1)
    for _ in range(100):
      self.assertTrue(scheduler(None))

  def test_bernoulli_steps_probability_1pct(self):
    """BernoulliStepScheduler should return more False than True with p=0.01."""
    scheduler = schedulers.BernoulliStepScheduler(keep_probability=0.01)
    num_true = 0
    for _ in range(1000):
      num_true += scheduler(None)
    num_false = 1000 - num_true
    self.assertGreater(num_false, num_true)

  def test_bernoulli_steps_probability_99pct(self):
    """BernoulliStepScheduler should return more True than False with p=0.99."""
    scheduler = schedulers.BernoulliStepScheduler(keep_probability=0.99)
    num_true = 0
    for _ in range(1000):
      num_true += scheduler(None)
    num_false = 1000 - num_true
    self.assertGreater(num_true, num_false)

  def test_bernoulli_step_fixed_seed(self):
    """BernoulliStepScheduler should return deterministic outcomes."""

    seed = np.random.default_rng().integers(10000)

    # Run one trial with `seed`.
    scheduler = schedulers.BernoulliStepScheduler(
        keep_probability=0.5, seed=seed)
    outcomes = [scheduler(None) for _ in range(100)]

    # Repeat the trial with the same `seed`.
    other_scheduler = schedulers.BernoulliStepScheduler(
        keep_probability=0.5, seed=seed)
    other_outcomes = [other_scheduler(None) for _ in range(100)]

    # Assert that the outcomes are exactly the same.
    self.assertEqual(outcomes, other_outcomes)

  @parameterized.named_parameters(
      ('negative_interval', -1),
      ('zero_interval', 0),
  )
  def test_n_episode_invalid_args(self, episode_interval):
    """NEpisodeScheduler should raise an error if given invalid intervals."""
    self.assertRaises(
        ValueError,
        schedulers.NEpisodeScheduler,
        episode_interval=episode_interval)

  def test_n_episode_interval_1(self):
    """NEpisodeScheduler should always return True if interval is 1."""
    scheduler = schedulers.NEpisodeScheduler(episode_interval=1)
    for _ in range(100):
      for timestep in _create_episode(num_transitions=np.random.randint(100)):
        self.assertTrue(scheduler(timestep))

  def test_n_episode_interval_2(self):
    """NEpisodeScheduler should return True every other episode."""
    scheduler = schedulers.NEpisodeScheduler(episode_interval=2)
    for _ in range(100):
      for timestep in _create_episode(num_transitions=np.random.randint(100)):
        self.assertTrue(scheduler(timestep))
      for timestep in _create_episode(num_transitions=np.random.randint(100)):
        self.assertFalse(scheduler(timestep))

  def test_n_episode_interval_n(self):
    """NEpisodeScheduler should return True only every n episodes."""
    for _ in range(10):
      n = np.random.randint(1, 50)
      scheduler = schedulers.NEpisodeScheduler(episode_interval=n)
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
    """The scheduler should raise an error with negative probabilities."""
    self.assertRaises(
        ValueError,
        schedulers.BernoulliEpisodeScheduler,
        keep_probability=keep_probability)

  def test_bernoulli_episodes_probability_0(self):
    """BernoulliEpisodeScheduler should return False if given probability 0.0."""
    scheduler = schedulers.BernoulliEpisodeScheduler(keep_probability=0.0)
    for _ in range(100):
      for timestep in _create_episode(num_transitions=np.random.randint(100)):
        self.assertFalse(scheduler(timestep))

  def test_bernoulli_episodes_probability_1(self):
    """BernoulliEpisodeScheduler should return True if given probability 1.0."""
    scheduler = schedulers.BernoulliEpisodeScheduler(keep_probability=1.0)
    for _ in range(100):
      for timestep in _create_episode(num_transitions=np.random.randint(100)):
        self.assertTrue(scheduler(timestep))

  def test_bernoulli_episodes_probability_1pct(self):
    """BernoulliEpisodeScheduler should return more False with p=0.01."""
    scheduler = schedulers.BernoulliEpisodeScheduler(keep_probability=0.01)
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
    """BernoulliEpisodeScheduler should return more True with p=0.99."""
    scheduler = schedulers.BernoulliEpisodeScheduler(keep_probability=0.99)
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

  def test_bernoulli_episode_fixed_seed(self):
    """BernoulliEpisodeScheduler should return deterministic outcomes."""

    seed = np.random.default_rng().integers(10000)
    episodes = [
        _create_episode(num_transitions=np.random.randint(100))
        for _ in range(1000)
    ]

    # Run one trial with `seed`.
    scheduler = schedulers.BernoulliEpisodeScheduler(
        keep_probability=0.5, seed=seed)
    outcomes = []
    for episode in episodes:
      for timestep in episode:
        outcomes.append(scheduler(timestep))

    # Repeat the trial with the same `seed`.
    other_scheduler = schedulers.BernoulliEpisodeScheduler(
        keep_probability=0.5, seed=seed)
    other_outcomes = []
    for episode in episodes:
      for timestep in episode:
        other_outcomes.append(other_scheduler(timestep))

    # Assert that the outcomes are exactly the same.
    self.assertEqual(outcomes, other_outcomes)

  @parameterized.named_parameters(
      ('empty_list', []),
      ('empty_ndarray', np.array([], dtype=np.int64)),
  )
  def test_list_steps_empty_steps(self, desired_steps):
    """ListStepScheduler should raise an error if given invalid steps."""
    self.assertRaises(
        ValueError, schedulers.ListStepScheduler, desired_steps=desired_steps)

  def test_list_np_array_wrong_type(self):
    """ListStepScheduler should raise an error if given invalid steps."""
    self.assertRaises(
        TypeError,
        schedulers.ListStepScheduler,
        desired_steps=np.array([1.0, 10.0, 100.0], dtype=np.float32))

  def test_list_steps_single_item(self):
    """ListStepScheduler should return True if step is in `desired_steps`."""
    scheduler = schedulers.ListStepScheduler(desired_steps=[3])
    self.assertFalse(scheduler(None))
    self.assertFalse(scheduler(None))
    self.assertFalse(scheduler(None))
    self.assertTrue(scheduler(None))  # 4th step should be True.
    for _ in range(100):
      self.assertFalse(scheduler(None))

  def test_list_steps_first_10(self):
    """ListStepScheduler should return True if step is in `desired_steps`."""
    scheduler = schedulers.ListStepScheduler(desired_steps=list(range(10)))
    for _ in range(10):  # First 10 steps should be True.
      self.assertTrue(scheduler(None))
    for _ in range(100):
      self.assertFalse(scheduler(None))

  def test_list_steps_logspace(self):
    """ListStepScheduler should return True if step is in `desired_steps`."""
    desired_steps = np.logspace(
        start=0, stop=3, num=10, base=10.0).astype(np.int32) - 1
    # At this point: desired_steps = [0, 1, 3, 9, 20, 45, 99, 214, 463, 999]
    scheduler = schedulers.ListStepScheduler(desired_steps=desired_steps)
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
    """ListEpisodeScheduler should raise an error if given invalid episodes."""
    self.assertRaises(
        ValueError,
        schedulers.ListEpisodeScheduler,
        desired_episodes=desired_episodes)

  def test_list_episodes_np_array_wrong_type(self):
    """ListEpisodeScheduler should raise an error if given invalid episodes."""
    self.assertRaises(
        TypeError,
        schedulers.ListEpisodeScheduler,
        desired_episodes=np.array([1.0, 10.0, 100.0], dtype=np.float32))

  def test_list_episodes_single_item(self):
    """Scheduler should return True if episode is in `desired_episodes`."""
    scheduler = schedulers.ListEpisodeScheduler(desired_episodes=[3])
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
    """The scheduler should return True if episode is in `desired_episodes`."""
    scheduler = schedulers.ListEpisodeScheduler(
        desired_episodes=list(range(10)))
    for _ in range(10):  # First 10 episodes should be True.
      for timestep in _create_episode(num_transitions=np.random.randint(100)):
        self.assertTrue(scheduler(timestep))
    for _ in range(100):
      for timestep in _create_episode(num_transitions=np.random.randint(100)):
        self.assertFalse(scheduler(timestep))

  def test_list_episodes_logspace(self):
    """The scheduler should return True if episode is in `desired_episodes`."""
    desired_episodes = np.logspace(
        start=0, stop=3, num=10, base=10.0).astype(np.int32) - 1
    # At this point: desired_episodes = [0, 1, 3, 9, 20, 45, 99, 214, 463, 999]
    scheduler = schedulers.ListEpisodeScheduler(
        desired_episodes=desired_episodes)
    for i in range(1000):
      if i in [0, 1, 3, 9, 20, 45, 99, 214, 463, 999]:
        for timestep in _create_episode(num_transitions=np.random.randint(100)):
          self.assertTrue(scheduler(timestep))
      else:
        for timestep in _create_episode(num_transitions=np.random.randint(100)):
          self.assertFalse(scheduler(timestep))

  @parameterized.named_parameters(
      ('negative_seconds', -5),
      ('zero_seconds', 0),
      ('negative_timedelta', datetime.timedelta(seconds=-1)),
      ('zero_timedelta', datetime.timedelta(seconds=0)),
  )
  def test_time_scheduler_invalid_args(self, interval):
    """TimeScheduler should raise an error if given invalid intervals."""
    self.assertRaises(ValueError, schedulers.TimeScheduler, interval=interval)

  def test_time_scheduler_logic(self):
    """TimeScheduler should return True only after the interval has elapsed."""
    current_time = 100.0

    def mock_clock() -> float:
      return current_time

    scheduler = schedulers.TimeScheduler(interval=5, clock=mock_clock)

    self.assertFalse(scheduler(None))

    current_time = 104.0
    self.assertFalse(scheduler(None))

    current_time = 105.0
    self.assertTrue(scheduler(None))

    self.assertFalse(scheduler(None))

    current_time = 111.0
    self.assertTrue(scheduler(None))

  def test_time_scheduler_timedelta_logic(self):
    current_time = 100.0

    def mock_clock() -> float:
      return current_time

    scheduler = schedulers.TimeScheduler(
        interval=datetime.timedelta(seconds=5), clock=mock_clock
    )

    self.assertFalse(scheduler(None))

    current_time = 105.0
    self.assertTrue(scheduler(None))

  def test_time_scheduler_boundary_conditions(self):
    """TimeScheduler should trigger exactly at the boundary."""
    current_time = 100.0

    def mock_clock() -> float:
      return current_time

    scheduler = schedulers.TimeScheduler(interval=5.0, clock=mock_clock)
    epsilon = 1e-6

    # Verify T_0
    self.assertFalse(scheduler(None))

    # T_0 + interval - epsilon -> should not trigger
    current_time = 100.0 + 5.0 - epsilon
    self.assertFalse(scheduler(None))

    # T_0 + interval -> should trigger
    current_time = 100.0 + 5.0
    self.assertTrue(scheduler(None))

    # Immediately after -> should not trigger
    self.assertFalse(scheduler(None))

    # T_1 = 105.0
    # T_1 + interval - epsilon -> should not trigger
    current_time = 105.0 + 5.0 - epsilon
    self.assertFalse(scheduler(None))

    # T_1 + interval + epsilon -> should trigger
    current_time = 105.0 + 5.0 + epsilon
    self.assertTrue(scheduler(None))

  def test_time_scheduler_timedelta_boundary_conditions(self):
    """TimeScheduler should preserve microsecond precision with timedelta."""
    current_time = 100.0

    def mock_clock() -> float:
      return current_time

    interval = datetime.timedelta(microseconds=10)
    scheduler = schedulers.TimeScheduler(interval=interval, clock=mock_clock)

    # interval in seconds is 0.00001
    interval_seconds = 0.00001
    epsilon = 1e-6  # 1 microsecond

    # Verify T_0
    self.assertFalse(scheduler(None))

    # T_0 + interval - epsilon -> should not trigger
    current_time = 100.0 + interval_seconds - epsilon
    self.assertFalse(scheduler(None))

    # T_0 + interval -> should trigger
    current_time = 100.0 + interval_seconds
    self.assertTrue(scheduler(None))

  def test_time_scheduler_large_time_jump(self):
    """TimeScheduler should trigger once and reset baseline on large jump."""
    current_time = 100.0

    def mock_clock() -> float:
      return current_time

    scheduler = schedulers.TimeScheduler(interval=5.0, clock=mock_clock)

    # Verify T_0
    self.assertFalse(scheduler(None))

    # Jump to 112.0 (elapsed 12.0)
    current_time = 112.0
    self.assertTrue(scheduler(None))  # Should trigger

    # Subsequent call at same time should not trigger
    self.assertFalse(scheduler(None))

    # T_1 is now 112.0
    # T_1 + 1.0 -> should not trigger
    current_time = 113.0
    self.assertFalse(scheduler(None))

    # T_1 + 5.0 -> should trigger
    current_time = 117.0
    self.assertTrue(scheduler(None))

  def test_time_scheduler_backward_clock(self):
    """TimeScheduler should handle backward clock jumps gracefully."""
    current_time = 100.0

    def mock_clock() -> float:
      return current_time

    scheduler = schedulers.TimeScheduler(interval=5.0, clock=mock_clock)

    # Verify T_0
    self.assertFalse(scheduler(None))

    # Move clock backward to 95.0
    current_time = 95.0
    self.assertFalse(scheduler(None))

    # Move clock back to 100.0
    current_time = 100.0
    self.assertFalse(scheduler(None))

    # Move clock to 105.0 (T_0 + 5.0) -> should trigger
    current_time = 105.0
    self.assertTrue(scheduler(None))


if __name__ == '__main__':
  absltest.main()
