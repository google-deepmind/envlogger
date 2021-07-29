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

"""Wrapper that logs `TimeStep`s and actions, metadata and episodes.

Data can be read back using:

- The python reader API (environment_logger.reader).

"""

from typing import Any, Callable, Dict, Optional, Union

import dm_env
from envlogger import environment_wrapper
from envlogger import step_data
from envlogger.backends import backend_type
from envlogger.backends import backend_writer
from envlogger.backends import in_memory_backend
from envlogger.backends import riegeli_backend_writer
from envlogger.converters import spec_codec

_DEFAULT_BACKEND = backend_type.BackendType.RIEGELI


class EnvLogger(environment_wrapper.EnvironmentWrapper):
  """Wrapper that logs timestep and actions."""

  def __init__(
      self,
      env: dm_env.Environment,
      step_fn: Optional[Callable[[dm_env.TimeStep, Any, dm_env.Environment],
                                 Any]] = None,
      episode_fn: Optional[Callable[[dm_env.TimeStep, Any, dm_env.Environment],
                                    Any]] = None,
      metadata: Optional[Dict[str, Any]] = None,
      backend: Union[backend_writer.BackendWriter, backend_type.BackendType,
                     Callable[...,
                              backend_writer.BackendWriter]] = _DEFAULT_BACKEND,
      **backend_kwargs):
    """Constructor.

    Usage:
      my_env = MyDmEnvironment()
        with EnvLogger(my_env, data_directory='/some/path/', ...) as env:
        # Use `env` just like `my_env`.
        # `.close()` is automatically called when the context is over.

    Calling `close()` will flush the trajectories and the index to disk and will
    ensure that they can be read later on. If it isn't called, there is a large
    risk of losing data. This is particularly common in some RL frameworks that
    do not clean up their environments. If the environment runs for a very long
    time, this can happen only to the last shard, but if the instance is
    short-lived, then a large portion of the trajectories can disappear.

    Args:
      env: The wrapped environment.
      step_fn: A function that takes the current timestep, current action, the
        environment itself and returns custom data that's written at every
        step() if it's not None.
      episode_fn: A function that takes the current timestep, current action,
        the environment itself and returns custom episodic data that's written
        when the current episode is over. If it is None or if it returns None,
        nothing is written.  This function is called at every step during the
        course of an episode, but only the last value it returns will actually
        be stored (all intermediate return values are ignored).
      metadata: Any dataset-level custom data to be written.
      backend: One of the following:
        * A `LoggingBackend` instance: `EnvLogger` will simply use this instance
          as is.
        * A `BackendType` enum indicating the backend to use: `EnvLogger` will
          construct a `LoggingBackend` from a list of predefined backends
          passing `backend_kwargs`.
        * A `Callable`: `EnvLogger` will call the given function passing
          `backend_kwargs`. The function _must_ return a `LoggingBackend`
          instance.
      **backend_kwargs: Extra arguments use to construct the backend. These will
        be handed to `backend` without any modification.
    """
    super().__init__(env)

    self._step_fn = step_fn
    self._episode_fn = episode_fn
    self._reset_next_step = True
    metadata = metadata or {}
    metadata['environment_specs'] = spec_codec.encode_environment_specs(env)
    backend_kwargs['metadata'] = metadata

    # Set backend.
    if isinstance(backend, backend_writer.BackendWriter):
      self._backend = backend
    elif isinstance(backend, backend_type.BackendType):
      self._backend = {
          backend_type.BackendType.RIEGELI:
              riegeli_backend_writer.RiegeliBackendWriter,
          backend_type.BackendType.IN_MEMORY:
              in_memory_backend.InMemoryBackendWriter,
      }[backend](**backend_kwargs)
    else:
      self._backend = backend(**backend_kwargs)

  def _transform_step(self,
                      timestep: dm_env.TimeStep,
                      action: Optional[Any] = None) -> step_data.StepData:
    """Puts all data into a StepData named tuple."""
    custom_data = None
    if self._step_fn is not None:
      custom_data = self._step_fn(timestep, action, self._environment)
    return step_data.StepData(timestep, action, custom_data)

  def reset(self):
    self._reset_next_step = False
    timestep = self._environment.reset()
    data = self._transform_step(timestep, None)
    self._backend.record_step(data, is_new_episode=True)
    if self._episode_fn is not None:
      episode_metadata = self._episode_fn(timestep, None, self._environment)
      if episode_metadata is not None:
        self._backend.set_episode_metadata(episode_metadata)
    return timestep

  def step(self, action):
    if self._reset_next_step:
      return self.reset()

    timestep = self._environment.step(action)
    self._reset_next_step = timestep.last()
    data = self._transform_step(timestep, action)
    self._backend.record_step(data, is_new_episode=False)
    if self._episode_fn is not None:
      episode_metadata = self._episode_fn(timestep, action, self._environment)
      if episode_metadata is not None:
        self._backend.set_episode_metadata(episode_metadata)
    return timestep

  def close(self):
    self._environment.close()
    self._backend.close()
