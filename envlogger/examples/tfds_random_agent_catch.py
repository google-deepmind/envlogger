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

"""A simple binary to run catch for a while and record its trajectories.
"""

import time

from absl import app
from absl import flags
from absl import logging
import envlogger
from envlogger.backends import tfds_backend_writer

from envlogger.testing import catch_env
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds



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

  dataset_config = tfds.rlds.rlds_base.DatasetConfig(
      name='catch_example',
      observation_info=tfds.features.Tensor(
          shape=(10, 5), dtype=tf.float32,
          encoding=tfds.features.Encoding.ZLIB),
      action_info=tf.int64,
      reward_info=tf.float64,
      discount_info=tf.float64,
      step_metadata_info={'timestamp': tf.float32})

  logging.info('Wrapping environment with EnvironmentLogger...')
  with envlogger.EnvLogger(
      env,
      step_fn=step_fn,
      backend = tfds_backend_writer.TFDSBackendWriter(
       data_directory=FLAGS.trajectories_dir,
       split_name='train',
       max_episodes_per_file=500,
       ds_config=dataset_config),
  ) as env:
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
