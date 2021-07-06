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

"""Abstract trajectory logging interface."""

import abc
from typing import Any, Dict, Optional

from envlogger import step_data
from envlogger.backends import schedulers


class BackendWriter(metaclass=abc.ABCMeta):
  """Abstract trajectory logging interface."""

  def __init__(self,
               metadata: Optional[Dict[str, Any]] = None,
               scheduler: Optional[schedulers.Scheduler] = None):
    """BackendWriter base class.

    Args:
      metadata: Any dataset-level custom data to be written.
      scheduler: A callable that takes the current timestep, current
        action, the environment itself and returns True if the current step
        should be logged, False otherwise. This function is called _before_
        `step_fn`, meaning that if it returns False, `step_fn` will not be
        called at all.
    """
    self._scheduler = scheduler
    self._metadata = metadata

  def record_step(self, data: step_data.StepData, is_new_episode: bool) -> None:
    if (self._scheduler is not None and not self._scheduler(data)):
      return
    self._record_step(data, is_new_episode)

  @abc.abstractmethod
  def set_episode_metadata(self, data: Any) -> None:
    pass

  @abc.abstractmethod
  def _record_step(self, data: step_data.StepData,
                   is_new_episode: bool) -> None:
    pass

  @abc.abstractmethod
  def close(self) -> None:
    pass

  def __del__(self):
    self.close()

  def metadata(self):
    return self._metadata
