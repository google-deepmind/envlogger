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

"""Base class for implementing environment wrappers.."""

import dm_env


class EnvironmentWrapper(dm_env.Environment):
  """An Environment which delegates calls to another environment.

  Subclasses should override one or more methods to modify the behavior of the
  backing environment as desired per the Decorator Pattern.

  This exposes the wrapped environment to subclasses with the `._environment`
  property and also defines `__getattr__` so that attributes are invisibly
  forwarded to the wrapped environment (and hence enabling duck-typing).
  """

  def __init__(self, environment: dm_env.Environment):
    self._environment = environment

  def __getattr__(self, name):
    return getattr(self._environment, name)

  def step(self, action) -> dm_env.TimeStep:
    return self._environment.step(action)

  def reset(self) -> dm_env.TimeStep:
    return self._environment.reset()

  def action_spec(self):
    return self._environment.action_spec()

  def discount_spec(self):
    return self._environment.discount_spec()

  def observation_spec(self):
    return self._environment.observation_spec()

  def reward_spec(self):
    return self._environment.reward_spec()

  def close(self):
    return self._environment.close()
