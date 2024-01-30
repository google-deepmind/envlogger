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

"""Utils to convert Envlogger data into RLDS."""

from typing import Any, Dict, Optional

from absl import logging
from envlogger import step_data
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

Step = Dict[str, Any]


def to_rlds_step(prev_step: step_data.StepData,
                 step: Optional[step_data.StepData]) -> Step:
  """Builds an RLDS step from two Envlogger steps.

  Steps follow the RLDS convention from https://github.com/google-research/rlds.

  Args:
    prev_step: previous step.
    step: current step. If None, it builds the last step (where the observation
      is the last one, and the action, reward and discount are undefined).

  Returns:
     RLDS Step.

  """
  metadata = {}
  if isinstance(prev_step.custom_data, dict):
    metadata = prev_step.custom_data
  return {
      'action':
          step.action if step else tf.nest.map_structure(
              np.zeros_like, prev_step.action),
      'discount':
          step.timestep.discount if step else tf.nest.map_structure(
              np.zeros_like, prev_step.timestep.discount),
      'is_first':
          prev_step.timestep.first(),
      'is_last':
          prev_step.timestep.last(),
      'is_terminal': (prev_step.timestep.last() and
                      prev_step.timestep.discount == 0.0),
      'observation':
          prev_step.timestep.observation,
      'reward':
          step.timestep.reward if step else tf.nest.map_structure(
              np.zeros_like, prev_step.timestep.reward),
      **metadata,
  }


def _find_extra_shard(split_info: tfds.core.SplitInfo) -> Optional[Any]:
  """Returns the filename of the extra shard, or None if all shards are in the metadata."""
  filepath = split_info.filename_template.sharded_filepath(
      shard_index=split_info.num_shards, num_shards=split_info.num_shards + 1)
  if tf.io.gfile.exists(filepath):
    # There is one extra shard for which we don't have metadata.
    return filepath
  return None


def maybe_recover_last_shard(builder: tfds.core.DatasetBuilder):
  """Goes through the splits and recovers the incomplete shards.

  It checks if the last shard is missing. If that is the case, it rewrites the
  metadata. This requires to read the full shard so it may take some time.

  We assume that only the last shard can be unaccounted for in the
  metadata because the logger generates shards sequentially and it updates the
  metadata once a shard is done and before starting the new shard.

  Args:
    builder: TFDS builder of the dataset that may have incomplete shards.

  Returns:
    A builder with the new split information.

  """
  split_infos = builder.info.splits
  splits_to_update = 0
  for _, split_info in split_infos.items():
    extra_shard = _find_extra_shard(split_info)
    if extra_shard is None:
      continue
    logging.info('Recovering data for shard %s.', extra_shard)
    splits_to_update += 1
    ds = tf.data.TFRecordDataset(extra_shard)
    num_examples = 0
    num_bytes = 0
    for ex in ds:
      num_examples += 1
      num_bytes += len(ex.numpy())

    new_split_info = split_info.replace(
        shard_lengths=split_info.shard_lengths + [num_examples],
        num_bytes=split_info.num_bytes + num_bytes)
    old_splits = [
        v for k, v in builder.info.splits.items() if k != new_split_info.name
    ]
    builder.info.set_splits(tfds.core.SplitDict(old_splits + [new_split_info]))
  if splits_to_update > 0:
    builder.info.write_to_directory(builder.data_dir)
  return builder
