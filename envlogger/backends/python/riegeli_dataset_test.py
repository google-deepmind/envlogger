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

"""Writes and reads data using RiegeliDataset{Writer, Reader}."""

import os
import shutil

from absl import logging
from absl.testing import absltest
from envlogger.backends.python import riegeli_dataset_reader
from envlogger.backends.python import riegeli_dataset_writer
from envlogger.converters import codec
from envlogger.proto import storage_pb2

from google.protobuf import descriptor_pool
from google.protobuf import message_factory


class RiegeliDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._directory = os.path.join(absltest.get_default_test_tmpdir(), 'blah')
    os.makedirs(self._directory)

  def tearDown(self):
    shutil.rmtree(self._directory)
    super().tearDown()

  def test_reader_non_existent_data_dir(self):
    """Checks that an exception is raised when a `data_dir` does not exist."""

    reader = riegeli_dataset_reader.RiegeliDatasetReader()
    self.assertRaises(RuntimeError, reader.init, data_dir='/i/do/not/exist/')

  def test_writer_non_existent_data_dir(self):
    """Checks that an exception is raised when a `data_dir` does not exist."""

    writer = riegeli_dataset_writer.RiegeliDatasetWriter()
    self.assertRaises(RuntimeError, writer.init, data_dir='/i/do/not/exist/')

  def test_storage_data_payload(self):
    """Ensures that we can read and write `Data` proto messages."""
    writer = riegeli_dataset_writer.RiegeliDatasetWriter()
    try:
      writer.init(data_dir=self._directory)
    except RuntimeError:
      logging.exception('Failed to initialize writer')

    for i in range(10):
      writer.add_step(codec.encode(i))
    writer.close()

    reader = riegeli_dataset_reader.RiegeliDatasetReader()
    try:
      reader.init(data_dir=self._directory)
    except RuntimeError:
      logging.exception('Failed to initialize reader')

    for i in range(reader.num_steps):
      step = reader.step(i)
      self.assertEqual(codec.decode(step), i)
    reader.close()

  def test_non_storage_data_payload(self):
    """Ensures that we can read and write proto messages other than `Data`."""
    writer = riegeli_dataset_writer.RiegeliDatasetWriter()
    try:
      writer.init(data_dir=self._directory)
    except RuntimeError:
      logging.exception('Failed to initialize writer')

    for i in range(10):
      dim = storage_pb2.Datum.Shape.Dim()
      dim.size = i
      writer.add_step(dim)
    writer.close()

    reader = riegeli_dataset_reader.RiegeliDatasetReader()
    try:
      reader.init(data_dir=self._directory)
    except RuntimeError:
      logging.exception('Failed to initialize reader')

    for i in range(reader.num_steps):
      step = reader.step(i, storage_pb2.Datum.Shape.Dim)
      self.assertEqual(step.size, i)
    reader.close()

  def test_dynamic_data_payload(self):
    """Checks that we can read and write dynamically obtained proto messages."""
    pool = descriptor_pool.Default()
    factory = message_factory.MessageFactory(pool)
    prototype = factory.GetPrototype(
        pool.FindMessageTypeByName('envlogger.Datum.Values'))

    writer = riegeli_dataset_writer.RiegeliDatasetWriter()
    try:
      writer.init(data_dir=self._directory)
    except RuntimeError:
      logging.exception('Failed to initialize writer')

    for i in range(10):
      values = prototype(float_values=[3.14 + i])
      writer.add_step(values)
    writer.close()

    reader = riegeli_dataset_reader.RiegeliDatasetReader()
    try:
      reader.init(data_dir=self._directory)
    except RuntimeError:
      logging.exception('Failed to initialize reader')

    for i in range(reader.num_steps):
      step = reader.step(i, prototype)
      self.assertAlmostEqual(step.float_values[0], 3.14 + i, places=3)
      # Protobuf message _objects_ also define `FromString()` and should work.
      step2 = reader.step(i, prototype())
      self.assertAlmostEqual(step2.float_values[0], 3.14 + i, places=3)
    reader.close()

  def test_clone(self):
    """Ensures that we can read the same data with a cloned reader."""
    writer = riegeli_dataset_writer.RiegeliDatasetWriter()
    try:
      writer.init(data_dir=self._directory)
    except RuntimeError:
      logging.exception('Failed to initialize writer')

    for i in range(10):
      writer.add_step(codec.encode(i))
    writer.close()

    reader = riegeli_dataset_reader.RiegeliDatasetReader()
    try:
      reader.init(data_dir=self._directory)
    except RuntimeError:
      logging.exception('Failed to initialize reader')

    cloned = reader.clone()
    self.assertEqual(cloned.num_steps, reader.num_steps)

    for i in range(reader.num_steps):
      step = reader.step(i)
      self.assertEqual(codec.decode(step), i)
    reader.close()

    # Even after closing the original `reader`, the cloned reader should still
    # work just like it.
    for i in range(cloned.num_steps):
      step = cloned.step(i)
      self.assertEqual(codec.decode(step), i)
    cloned.close()

if __name__ == '__main__':
  absltest.main()
