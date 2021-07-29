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

"""A simple binary to run catch for a while and record its trajectories.
"""

import time

from absl import app
from absl import flags
from absl import logging
from envlogger import environment_logger
from envlogger.testing import catch_env
import numpy as np


FLAGS = flags.FLAGS

flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes to log.')
flags.DEFINE_string('trajectories_dir', '/tmp/catch_data/',
                    'Path in a filesystem to record trajectories.')


def main(unused_argv):
  logging.info('Creating Catch environment...')
  env = catch_env.Catch()
  logging.info('Done creating Catch environment.')

  def step_fn(unused_timestep, unused_action, unused_env):
    return {'timestamp': time.time()}

  logging.info('Wrapping environment with EnvironmentLogger...')
  with environment_logger.EnvLogger(
      env,
      data_directory=FLAGS.trajectories_dir,
      max_episodes_per_file=1000,
      metadata={
          'agent_type': 'random',
          'env_type': type(env).__name__,
          'num_episodes': FLAGS.num_episodes,
      },
      step_fn=step_fn) as env:
    logging.info('Done wrapping environment with EnvironmentLogger.')

    logging.info('Training a random agent for %r episodes...',
                 FLAGS.num_episodes)
    for i in range(FLAGS.num_episodes):
      logging.info('episode %r', i)
      timestep = env.reset()
      while not timestep.last():
        action = np.random.randint(low=0, high=3)
        timestep = env.step(action)
    logging.info('Done training a random agent for %r episodes.',
                 FLAGS.num_episodes)


if __name__ == '__main__':
  app.run(main)
