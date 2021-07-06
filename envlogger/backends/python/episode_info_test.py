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

"""Tests for episode_info.cc."""

import random

from absl.testing import absltest
from absl.testing import parameterized
from envlogger.backends.python import episode_info
from envlogger.proto import storage_pb2


class EpisodeInfoTest(parameterized.TestCase):

  def test_empty_episode_info(self):
    episode = episode_info.EpisodeInfo()
    self.assertEqual(episode.start, 0)
    self.assertEqual(episode.num_steps, 0)
    self.assertIsNone(episode.metadata)

  def test_episode_info_init_with_random_kwargs(self):
    random_starts = [random.randint(-1, 10000) for _ in range(100)]
    random_num_steps = [random.randint(-1, 10000) for _ in range(100)]
    random_metadata = []

    dimension = storage_pb2.Datum.Shape.Dim()
    dimension.size = -438
    for _ in range(100):
      metadata = storage_pb2.Data()
      metadata.datum.shape.dim.append(dimension)
      metadata.datum.values.int32_values.append(random.randint(-1, 10000))
      random_metadata.append(metadata)

    for start, num_steps, metadata in zip(random_starts, random_num_steps,
                                          random_metadata):
      episode = episode_info.EpisodeInfo(
          start=start, num_steps=num_steps, metadata=metadata)
      self.assertEqual(episode.start, start)
      self.assertEqual(episode.num_steps, num_steps)
      self.assertSequenceEqual(episode.metadata.datum.values.int32_values,
                               metadata.datum.values.int32_values)


if __name__ == '__main__':
  absltest.main()
