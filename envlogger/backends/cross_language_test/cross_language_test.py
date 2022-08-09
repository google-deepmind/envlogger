# coding=utf-8
# Copyright 2022 DeepMind Technologies Limited..
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

"""Tests writing trajectories in one language and reading from another."""

import os
import shutil
import subprocess
from typing import Sequence

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from rules_python.python.runfiles import runfiles


def _execute_binary(rel_path: str, args: Sequence[str]) -> bytes:
  r = runfiles.Create()
  path = r.Rlocation(os.path.join('__main__', 'envlogger', rel_path))
  cmd = [path] + args
  return subprocess.check_output(cmd, env=r.EnvVars())


class CrossLanguageTest(parameterized.TestCase):

  def test_py_writer_cc_reader(self):
    # Set up a trajectory directory.
    trajectories_dir = os.path.join(absltest.TEST_TMPDIR.value, 'my_trajectory')
    logging.info('trajectories_dir: %r', trajectories_dir)
    os.makedirs(trajectories_dir)

    # Find Python writer and run it.
    py_writer_output = _execute_binary(
        'backends/cross_language_test/py_writer',
        args=[f'--trajectories_dir={trajectories_dir}'])
    logging.info('py_writer_output: %r', py_writer_output)

    # Find C++ reader and run it.
    cc_reader_output = _execute_binary(
        'backends/cross_language_test/cc_reader',
        args=[f'--trajectories_dir={trajectories_dir}'])
    logging.info('cc_reader_output: %r', cc_reader_output)

    # If everything went well, there should be no
    # `subprocess.CalledProcessError`.

    logging.info('Cleaning up trajectories_dir %r', trajectories_dir)
    shutil.rmtree(trajectories_dir)


if __name__ == '__main__':
  absltest.main()
