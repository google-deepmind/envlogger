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

"""Common logging scheduling strategies."""

from typing import Callable, List, Union

from envlogger import step_data
import numpy as np


# A Scheduler returns True when something should be activated and False
# otherwise.
Scheduler = Callable[[step_data.StepData], bool]


def n_step_scheduler(step_interval: int) -> Scheduler:
  """Returns a closure that returns True every N times it is called.

  Args:
    step_interval: Must be a positive integer.
  Raises:
    ValueError: When `step_interval` is not a positive integer.
  """
  if step_interval <= 0:
    raise ValueError(f'step_interval must be positive, got {step_interval}')

  def f(unused_data: step_data.StepData) -> bool:
    should_log = f.step_counter % step_interval == 0
    f.step_counter += 1
    return should_log

  f.step_counter = 0
  return f


def bernoulli_step_scheduler(keep_probability: float) -> Scheduler:
  """Returns a closure that returns True with probability `keep_probability`.

  Args:
    keep_probability: Must be within [0,1].
  Raises:
    ValueError: When `keep_probability` is not in [0,1].
  """
  if keep_probability < 0.0 or keep_probability > 1.0:
    raise ValueError(
        f'keep_probability must be in [0,1], got: {keep_probability}')

  def f(unused_data: step_data.StepData) -> bool:
    return np.random.random() < keep_probability

  return f


def n_episode_scheduler(episode_interval: int) -> Scheduler:
  """Returns a closure that returns True every N episodes.

  Args:
    episode_interval: Must be a positive integer.
  Raises:
    ValueError: When `episode_interval` is not a positive integer.
  """
  if episode_interval <= 0:
    raise ValueError(
        f'episode_interval must be positive, got {episode_interval}')

  def f(data: step_data.StepData) -> bool:
    should_log = f.episode_counter % episode_interval == 0
    if data.timestep.last():
      f.episode_counter += 1
    return should_log

  f.episode_counter = 0
  return f


def bernoulli_episode_scheduler(keep_probability: float) -> Scheduler:
  """Returns a closure that keeps episodes with probability `keep_probability`.

  Args:
    keep_probability: Must be within [0,1].
  Raises:
    ValueError: When `keep_probability` is not in [0,1].
  """
  if keep_probability < 0.0 or keep_probability > 1.0:
    raise ValueError(
        f'keep_probability must be in [0,1], got: {keep_probability}')

  def f(data: step_data.StepData) -> bool:
    if data.timestep.last():
      f.current_p = np.random.random()
    return f.current_p < keep_probability

  f.current_p = np.random.random()

  return f


def list_step_scheduler(
    desired_steps: Union[List[int], np.ndarray]) -> Scheduler:
  """Returns a closure that keeps steps in `desired_steps`.

  Please see unit tests for examples of using this scheduler. In particular,
  you can use Numpy's functions such as logspace() to generate non-linear steps.

  Args:
    desired_steps: Step indices (0-based) that indicate desired steps to log.
  Raises:
    ValueError: When `desired_steps` is empty.
  """
  if (isinstance(desired_steps, np.ndarray) and
      not (desired_steps.dtype == np.int32 or desired_steps.dtype == np.int64)):
    raise TypeError(
        f'desired_steps.dtype must be np.in32 or np.int64: {desired_steps} '
        f'(dtype: {desired_steps.dtype})')
  if len(desired_steps) <= 0:
    raise ValueError(f'desired_steps cannot be empty: {desired_steps}')

  def f(unused_data: step_data.StepData) -> bool:
    should_log = f.step_counter in f.desired_steps
    f.step_counter += 1
    return should_log

  f.desired_steps = set(desired_steps)
  f.step_counter = 0
  return f


def list_episode_scheduler(
    desired_episodes: Union[List[int], np.ndarray]) -> Scheduler:
  """Returns a closure that keeps episodes in `desired_episodes`.

  Please see unit tests for examples of using this scheduler. In particular,
  you can use Numpy's functions such as logspace() to generate non-linear
  episodes.

  Args:
    desired_episodes: Episode indices (0-based) that indicate desired espiodes
        to log.
  Raises:
    ValueError: When `desired_episodes` is empty.
  """
  if (isinstance(desired_episodes, np.ndarray) and
      not (desired_episodes.dtype == np.int32 or
           desired_episodes.dtype == np.int64)):
    raise TypeError('desired_episodes.dtype must be np.in32 or np.int64: '
                    f'{desired_episodes} (dtype: {desired_episodes.dtype})')
  if len(desired_episodes) <= 0:
    raise ValueError(f'desired_episodes cannot be empty: {desired_episodes}')

  def f(data: step_data.StepData) -> bool:
    should_log = f.episode_counter in f.desired_episodes
    if data.timestep.last():
      f.episode_counter += 1
    return should_log

  f.desired_episodes = set(desired_episodes)
  f.episode_counter = 0
  return f
