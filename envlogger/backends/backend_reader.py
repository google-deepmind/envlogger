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

"""Abstract interface for reading trajectories."""

import abc
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Sequence, TypeVar, Union

from absl import logging
from envlogger import step_data
from envlogger.backends.python import episode_info
from envlogger.converters import codec


T = TypeVar('T')


class _SequenceAdapter(Generic[T], Sequence[T]):
  """Convenient visitor for episodes/steps."""

  def __init__(self, count: int, get_nth_item: Callable[[int], T]):
    """Constructor.

    Args:
      count: Total number of items.
      get_nth_item: Function to get the nth item.
    """
    self._count = count
    self._index = 0
    self._get_nth_item = get_nth_item

  def __getitem__(self, index: Union[int, slice]) -> Union[T, List[T]]:
    """Retrieves items from this sequence.

    Args:
      index: item index or slice of indices.

    Returns:
      The item at `index` if index is of type `int`, or a list of items if
      `index` is a slice.  If `index` is a negative integer, then it is
      equivalent to index + len(self).

    Raises:
      IndexError: if index is an integer outside of the bounds [-length,
      length - 1].
    """
    if isinstance(index, slice):
      indices = index.indices(len(self))
      return [self._get_nth_item(i) for i in range(*indices)]
    if index >= self._count or index < -self._count:
      raise IndexError(f'`index`=={index} is out of the range [{-self._count}, '
                       f'{self._count - 1}].')
    index = index if index >= 0 else index + self._count
    return self._get_nth_item(index)

  def __len__(self) -> int:
    return self._count

  def __iter__(self) -> Iterator[T]:
    while self._index < len(self):
      yield self[self._index]
      self._index += 1
    self._index = 0

  def __next__(self) -> T:
    if self._index < len(self):
      index = self._index
      self._index += 1
      return self[index]
    else:
      raise StopIteration()


class BackendReader(metaclass=abc.ABCMeta):
  """Base class for trajectory readers."""

  def __init__(self):
    logging.info('Creating visitors.')
    self._steps = _SequenceAdapter(
        count=self._get_num_steps(), get_nth_item=self._get_nth_step)
    self._episodes = _SequenceAdapter(
        count=self._get_num_episodes(), get_nth_item=self._get_nth_episode)
    self._episode_metadata = _SequenceAdapter(
        count=self._get_num_episodes(),
        get_nth_item=self._get_nth_episode_metadata)
    logging.info('Done creating visitors.')

  @abc.abstractmethod
  def _get_nth_step(self, i: int) -> step_data.StepData:
    pass

  @abc.abstractmethod
  def _get_num_steps(self) -> int:
    pass

  @abc.abstractmethod
  def _get_num_episodes(self) -> int:
    pass

  @abc.abstractmethod
  def _get_nth_episode_info(self,
                            i: int,
                            include_metadata: bool = False
                           ) -> episode_info.EpisodeInfo:
    pass

  def _get_nth_episode(self, i: int) -> Sequence[step_data.StepData]:
    """Yields timesteps for episode `i` (0-based)."""
    episode = self._get_nth_episode_info(i, include_metadata=False)

    def get_nth_step_from_episode(j: int):
      return self._get_nth_step(episode.start + j)

    return _SequenceAdapter(
        count=episode.num_steps, get_nth_item=get_nth_step_from_episode)

  def _get_nth_episode_metadata(self, i: int) -> Optional[Any]:
    """Returns the metadata for episode `i` (0-based)."""
    episode = self._get_nth_episode_info(i, include_metadata=True)
    return codec.decode(episode.metadata)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, tb):
    self.close()

  def __del__(self):
    self.close()

  @abc.abstractmethod
  def close(self) -> None:
    pass

  @abc.abstractmethod
  def metadata(self) -> Dict[str, Any]:
    pass

  @property
  def episodes(self) -> Sequence[Sequence[step_data.StepData]]:
    return self._episodes

  def episode_metadata(self) -> Sequence[Optional[Any]]:
    return self._episode_metadata

  @property
  def steps(self) -> Sequence[step_data.StepData]:
    return self._steps
