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

"""Common logging scheduling strategies."""

from typing import Callable, List, Optional, Union

from envlogger import step_data
import numpy as np


# A Scheduler returns True when something should be activated and False
# otherwise.
Scheduler = Callable[[step_data.StepData], bool]


class NStepScheduler:
  """Returns `True` every N times it is called."""

  def __init__(self, step_interval: int):
    if step_interval <= 0:
      raise ValueError(f'step_interval must be positive, got {step_interval}')

    self._step_interval = step_interval
    self._step_counter = 0

  def __call__(self, unused_data: step_data.StepData):
    """Returns `True` every N times it is called."""

    should_log = self._step_counter % self._step_interval == 0
    self._step_counter += 1
    return should_log


class BernoulliStepScheduler:
  """Returns `True` with a given probability."""

  def __init__(self, keep_probability: float, seed: Optional[int] = None):
    if keep_probability < 0.0 or keep_probability > 1.0:
      raise ValueError(
          f'keep_probability must be in [0,1], got: {keep_probability}')

    self._keep_probability = keep_probability
    self._rng = np.random.default_rng(seed)

  def __call__(self, unused_data: step_data.StepData):
    """Returns `True` with probability `self._keep_probability`."""

    return self._rng.random() < self._keep_probability


class NEpisodeScheduler:
  """Returns `True` every N episodes."""

  def __init__(self, episode_interval: int):
    if episode_interval <= 0:
      raise ValueError(
          f'episode_interval must be positive, got {episode_interval}')

    self._episode_interval = episode_interval
    self._episode_counter = 0

  def __call__(self, data: step_data.StepData):
    """Returns `True` every N episodes."""

    should_log = self._episode_counter % self._episode_interval == 0
    if data.timestep.last():
      self._episode_counter += 1
    return should_log


class BernoulliEpisodeScheduler:
  """Returns `True` with a given probability at every episode."""

  def __init__(self, keep_probability: float, seed: Optional[int] = None):
    if keep_probability < 0.0 or keep_probability > 1.0:
      raise ValueError(
          f'keep_probability must be in [0,1], got: {keep_probability}')

    self._keep_probability = keep_probability
    self._rng = np.random.default_rng(seed)
    self._current_p = self._rng.random()

  def __call__(self, data: step_data.StepData):
    """Returns `True` with probability `self._keep_probability`."""

    if data.timestep.last():
      self._current_p = self._rng.random()
    return self._current_p < self._keep_probability


class ListStepScheduler:
  """Returns `True` for steps in `desired_steps`.

  Please see unit tests for examples of using this scheduler. In particular,
  you can use Numpy's functions such as logspace() to generate non-linear steps.
  """

  def __init__(self, desired_steps: Union[List[int], np.ndarray]):
    if (isinstance(desired_steps, np.ndarray) and
        not (desired_steps.dtype == np.int32 or
             desired_steps.dtype == np.int64)):
      raise TypeError(
          f'desired_steps.dtype must be np.in32 or np.int64: {desired_steps} '
          f'(dtype: {desired_steps.dtype})')
    if len(desired_steps) <= 0:
      raise ValueError(f'desired_steps cannot be empty: {desired_steps}')

    self._desired_steps = set(desired_steps)
    self._step_counter = 0

  def __call__(self, data: step_data.StepData):
    """Returns `True` every N episodes."""

    should_log = self._step_counter in self._desired_steps
    self._step_counter += 1
    return should_log


class ListEpisodeScheduler:
  """Returns `True` for episodes in `desired_episodes`.

  Please see unit tests for examples of using this scheduler. In particular,
  you can use Numpy's functions such as logspace() to generate non-linear steps.
  """

  def __init__(self, desired_episodes: Union[List[int], np.ndarray]):
    if (isinstance(desired_episodes, np.ndarray) and
        not (desired_episodes.dtype == np.int32 or
             desired_episodes.dtype == np.int64)):
      raise TypeError('desired_episodes.dtype must be np.in32 or np.int64: '
                      f'{desired_episodes} (dtype: {desired_episodes.dtype})')
    if len(desired_episodes) <= 0:
      raise ValueError(f'desired_episodes cannot be empty: {desired_episodes}')

    self._desired_episodes = set(desired_episodes)
    self._episode_counter = 0

  def __call__(self, data: step_data.StepData):
    """Returns `True` every N episodes."""

    should_log = self._episode_counter in self._desired_episodes
    if data.timestep.last():
      self._episode_counter += 1
    return should_log
