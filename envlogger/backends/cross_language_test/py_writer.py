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

"""A simple python binary that creates a simple RL trajectory."""

from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import dm_env
import envlogger
import numpy as np

_TRAJECTORIES_DIR = flags.DEFINE_string(
    'trajectories_dir', None, 'Path to write trajectory.', required=True)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('Starting Python-based writer...')
  logging.info('--trajectories_dir: %r', _TRAJECTORIES_DIR.value)

  writer = envlogger.RiegeliBackendWriter(
      data_directory=_TRAJECTORIES_DIR.value, metadata={'my_data': [1, 2, 3]})
  writer.record_step(
      envlogger.StepData(
          timestep=dm_env.TimeStep(
              observation=np.array([0.0], dtype=np.float32),
              reward=0.0,
              discount=0.99,
              step_type=dm_env.StepType.FIRST),
          action=np.int32(100)),
      is_new_episode=True)
  for i in range(1, 100):
    writer.record_step(
        envlogger.StepData(
            timestep=dm_env.TimeStep(
                observation=np.array([float(i)], dtype=np.float32),
                reward=i / 100.0,
                discount=0.99,
                step_type=dm_env.StepType.MID),
            action=np.int32(100 - i)),
        is_new_episode=False)
  writer.close()


if __name__ == '__main__':
  app.run(main)
