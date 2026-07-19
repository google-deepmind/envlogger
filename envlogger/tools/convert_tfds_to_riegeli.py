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

"""Tool to convert TFDS/RLDS datasets to EnvLogger Riegeli format."""

from collections.abc import Sequence
import dataclasses
from typing import Any

from absl import app
from absl import flags
from absl import logging
import dm_env
import dm_env.specs
from envlogger import environment_logger
from envlogger.backends import riegeli_backend_writer
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tree

# RLDS dataset string key constants (eliminates dependency on rlds package).
_RLDS_STEPS = 'steps'
_RLDS_OBSERVATION = 'observation'
_RLDS_ACTION = 'action'
_RLDS_REWARD = 'reward'
_RLDS_DISCOUNT = 'discount'
_RLDS_IS_FIRST = 'is_first'
_RLDS_IS_LAST = 'is_last'
_RLDS_IS_TERMINAL = 'is_terminal'

_TFDS_DIR = flags.DEFINE_string(
    'tfds_dir', None, 'Directory of the TFDS dataset.'
)
_TFDS_NAME = flags.DEFINE_string(
    'tfds_name', None, 'Name of registered TFDS dataset.'
)
_SPLIT = flags.DEFINE_string('split', 'train', 'Dataset split to convert.')
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None, 'Directory to write Riegeli dataset.'
)
_MAX_EPISODE_STEPS = flags.DEFINE_integer(
    'max_episode_steps', 10000, 'Max steps per episode batch.'
)


class ReplayEnvironment(dm_env.Environment):
  """Environment which replays an episode."""

  def __init__(
      self,
      observation_spec: Any,
      reward_spec: Any,
      discount_spec: Any,
      action_spec: Any,
  ):
    self._observation_spec = observation_spec
    self._reward_spec = reward_spec
    self._discount_spec = discount_spec
    self._action_spec = action_spec

    self._timesteps: Sequence[dm_env.TimeStep] | None = None
    self._timesteps_index: int = 0
    self._custom_data: Sequence[Any] | None = None
    self._custom_data_index: int = 0

  def set_episode(
      self,
      timesteps: Sequence[dm_env.TimeStep],
      custom_data: Sequence[Any],
  ) -> None:
    """Set the active episode to replay, must be called right before reset."""
    self._timesteps = timesteps
    self._custom_data = custom_data

  def reset(self) -> dm_env.TimeStep:
    self._timesteps_index = 0
    self._custom_data_index = 0
    if self._timesteps is None:
      raise ValueError('No timesteps set.')
    timestep = self._timesteps[self._timesteps_index]
    self._timesteps_index += 1
    return timestep

  def step(self, action: Any) -> dm_env.TimeStep:
    if self._timesteps is None:
      raise ValueError('No timesteps set.')
    timestep = self._timesteps[self._timesteps_index]
    self._timesteps_index += 1
    if self._timesteps_index >= len(self._timesteps):
      self._timesteps = None
    return timestep

  def close(self) -> None:
    pass

  def observation_spec(self) -> Any:
    return self._observation_spec

  def reward_spec(self) -> Any:
    return self._reward_spec

  def discount_spec(self) -> Any:
    return self._discount_spec

  def action_spec(self) -> Any:
    return self._action_spec

  def get_custom_data(self) -> Any:
    if self._custom_data is None:
      return {}
    custom_data = self._custom_data[self._custom_data_index]
    self._custom_data_index += 1
    return custom_data


@dataclasses.dataclass
class EpisodeMetadataHolder:
  """Placeholder for episode metadata."""

  episode_metadata: dict[str, Any] | None = None

  def get_episode_metadata(
      self,
      timestep: dm_env.TimeStep,
      action: Any,
      env: dm_env.Environment,
  ) -> dict[str, Any] | None:
    del timestep
    del action
    del env
    return self.episode_metadata


def _dm_env_spec(x: Any) -> Any:
  """Converts a TF spec to a DMEnv spec."""

  def dm_env_spec_single(t: Any) -> dm_env.specs.Array:
    return dm_env.specs.Array(shape=t.shape, dtype=t.dtype.as_numpy_dtype)

  return tree.map_structure(dm_env_spec_single, x)


def _get_step_value(struct: Any, index: int) -> Any:
  """Extracts the value at a specific index from a nested structure of arrays."""
  return tree.map_structure(lambda x, idx=index: x[idx], struct)


def _create_timestep(
    steps: Any, step_index: int, total_steps: int, discount_dtype: Any
) -> dm_env.TimeStep:
  """Creates a dm_env.TimeStep for a given step index in an RLDS episode."""
  if step_index == 0:
    return dm_env.restart(_get_step_value(steps[_RLDS_OBSERVATION], step_index))

  prev_reward = _get_step_value(steps[_RLDS_REWARD], step_index - 1)
  curr_obs = _get_step_value(steps[_RLDS_OBSERVATION], step_index)

  if step_index == total_steps - 1:
    if steps[_RLDS_IS_TERMINAL][step_index]:
      discount = np.array(0.0, dtype=discount_dtype)
      return dm_env.TimeStep(
          dm_env.StepType.LAST, prev_reward, discount, curr_obs
      )
    prev_discount = _get_step_value(steps[_RLDS_DISCOUNT], step_index - 1)
    return dm_env.truncation(prev_reward, curr_obs, prev_discount)

  prev_discount = _get_step_value(steps[_RLDS_DISCOUNT], step_index - 1)
  return dm_env.transition(prev_reward, curr_obs, prev_discount)


def _extract_custom_data(
    steps: Any, step_index: int, custom_data_keys: Sequence[str]
) -> dict[str, Any]:
  """Extracts custom step metadata for a given step index."""
  current_custom_data = {}
  for key in custom_data_keys:
    current_custom_data[key] = _get_step_value(steps[key], step_index)
  return current_custom_data


def convert_episode_to_dmenv(
    steps: Any,
) -> tuple[Sequence[dm_env.TimeStep], Sequence[Any], Sequence[dict[str, Any]]]:
  """Converts steps of an RLDS episode to DMEnv timesteps and actions."""
  standard_keys = {
      _RLDS_ACTION,
      _RLDS_REWARD,
      _RLDS_IS_TERMINAL,
      _RLDS_OBSERVATION,
      _RLDS_DISCOUNT,
      _RLDS_IS_FIRST,
      _RLDS_IS_LAST,
  }
  custom_data_keys = [k for k in steps if k not in standard_keys]

  num_steps = steps[_RLDS_IS_TERMINAL].shape[0]
  discount_dtype = steps[_RLDS_DISCOUNT].dtype
  actions = []
  timesteps = []
  custom_data = []

  for k in range(num_steps):
    if k > 0:
      actions.append(_get_step_value(steps[_RLDS_ACTION], k - 1))

    timesteps.append(_create_timestep(steps, k, num_steps, discount_dtype))
    custom_data.append(_extract_custom_data(steps, k, custom_data_keys))

  return timesteps, actions, custom_data


def _infer_replay_environment(
    episodes_dataset: tf.data.Dataset,
) -> ReplayEnvironment:
  """Infers spec and creates a ReplayEnvironment from the first dataset step."""
  for episode in episodes_dataset:
    for step in episode[_RLDS_STEPS]:
      return ReplayEnvironment(
          observation_spec=_dm_env_spec(step[_RLDS_OBSERVATION]),
          reward_spec=_dm_env_spec(step[_RLDS_REWARD]),
          discount_spec=_dm_env_spec(step[_RLDS_DISCOUNT]),
          action_spec=_dm_env_spec(step[_RLDS_ACTION]),
      )
  raise ValueError('Empty dataset or empty first episode.')


def _extract_episode_metadata(episode: Any) -> dict[str, Any]:
  """Extracts metadata dictionary for an episode."""
  metadata = {}
  for key, val in episode.items():
    if key != _RLDS_STEPS:
      metadata[key] = val.numpy()
  return metadata


def _log_episode(
    episode: Any,
    env: ReplayEnvironment,
    wrapped_env: environment_logger.EnvLogger,
    metadata_holder: EpisodeMetadataHolder,
    max_episode_steps: int,
) -> None:
  """Logs a single episode to Riegeli via EnvLogger."""
  timesteps: Sequence[dm_env.TimeStep] = []
  actions: Sequence[Any] = []
  custom_data: Sequence[Any] = []

  batched_steps = episode[_RLDS_STEPS].batch(max_episode_steps)
  for e in batched_steps.as_numpy_iterator():
    timesteps, actions, custom_data = convert_episode_to_dmenv(e)

  metadata_holder.episode_metadata = _extract_episode_metadata(episode)
  env.set_episode(timesteps, custom_data)

  wrapped_env.reset()
  for action in actions:
    wrapped_env.step(action)


def convert_tf_dataset_to_riegeli(
    episodes_dataset: tf.data.Dataset,
    output_dir: str,
    max_episode_steps: int = 10000,
) -> None:
  """Converts a TF episode dataset (RLDS format) to EnvLogger Riegeli format."""
  env = _infer_replay_environment(episodes_dataset)
  episode_metadata_holder = EpisodeMetadataHolder()

  riegeli_backend = riegeli_backend_writer.RiegeliBackendWriter(
      data_directory=output_dir,
  )

  def step_fn(
      timestep: dm_env.TimeStep, action: Any, unused_env: dm_env.Environment
  ) -> Any:
    del timestep
    del action
    del unused_env
    return env.get_custom_data()

  wrapped_env = environment_logger.EnvLogger(
      env,
      step_fn=step_fn,
      backend=riegeli_backend,
      episode_fn=episode_metadata_holder.get_episode_metadata,
  )

  for episode in episodes_dataset:
    _log_episode(
        episode,
        env,
        wrapped_env,
        episode_metadata_holder,
        max_episode_steps,
    )

  wrapped_env.close()


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if not _OUTPUT_DIR.value:
    raise app.UsageError('--output_dir must be specified.')

  if _TFDS_DIR.value:
    builder = tfds.builder_from_directory(_TFDS_DIR.value)
    ds = builder.as_dataset(split=_SPLIT.value)
  elif _TFDS_NAME.value:
    ds = tfds.load(_TFDS_NAME.value, split=_SPLIT.value)
  else:
    raise ValueError('Either --tfds_dir or --tfds_name must be specified.')

  logging.info(
      'Converting TFDS dataset to Riegeli format at %s...', _OUTPUT_DIR.value
  )
  convert_tf_dataset_to_riegeli(
      episodes_dataset=ds,
      output_dir=_OUTPUT_DIR.value,
      max_episode_steps=_MAX_EPISODE_STEPS.value,
  )
  logging.info('Conversion completed successfully.')


if __name__ == '__main__':
  app.run(main)
