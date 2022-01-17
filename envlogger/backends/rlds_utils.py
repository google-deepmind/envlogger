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

"""Utils to convert Envlogger data into RLDS."""

import os
from typing import Optional

from absl import logging
from envlogger import step_data
import numpy as np
from rlds import rlds_types
import tensorflow as tf
import tensorflow_datasets as tfds


def to_rlds_step(prev_step: step_data.StepData,
                 step: Optional[step_data.StepData]) -> rlds_types.Step:
  """Builds an RLDS step from two Envlogger steps.

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
      rlds_types.ACTION:
          step.action if step else np.zeros_like(prev_step.action),
      rlds_types.DISCOUNT:
          step.timestep.discount
          if step else np.zeros_like(prev_step.timestep.discount),
      rlds_types.IS_FIRST:
          prev_step.timestep.first(),
      rlds_types.IS_LAST:
          prev_step.timestep.last(),
      rlds_types.IS_TERMINAL: (prev_step.timestep.last() and
                               prev_step.timestep.discount == 0.0),
      rlds_types.OBSERVATION:
          prev_step.timestep.observation,
      rlds_types.REWARD:
          step.timestep.reward
          if step else np.zeros_like(prev_step.timestep.reward),
      **metadata,
  }


def _find_extra_shard(data_dir: str, dataset_name: str, split_name: str,
                      split_info: tfds.core.SplitInfo) -> Optional[str]:
  """Returns the filename of the extra shard, or None if all shards are in the metadata."""

  extra_shards = split_info.num_shards + 1
  filenames = tfds.core.naming.filenames_for_dataset_split(
      dataset_name=dataset_name,
      split=split_name,
      num_shards=extra_shards,
      filetype_suffix='tfrecord')
  suffix = f'-of-{extra_shards:05d}'
  # The extra shard file doesn't have the shards suffix. Otherwise, it means
  # that this shard is already accounted for in the metadata.
  filename = filenames[-1]
  no_suffix = filename[:-len(suffix)]
  old_path = os.path.join(data_dir, no_suffix)
  if tf.io.gfile.exists(old_path):
    # There is one extra shard for which we don't have metadata, so it was
    # not renamed.
    return old_path
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
  for split_name, split_info in split_infos.items():
    extra_shard = _find_extra_shard(
        data_dir=builder.data_dir,
        dataset_name=builder.name,
        split_name=split_name,
        split_info=split_info)
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

    new_split_info = tfds.core.SplitInfo(
        name=split_info.name,
        shard_lengths=split_info.shard_lengths + [num_examples],
        num_bytes=split_info.num_bytes + num_bytes)
    old_splits = [
        v for k, v in builder.info.splits.items() if k != new_split_info.name
    ]
    builder.info.set_splits(
        tfds.core.SplitDict(
            old_splits + [new_split_info], dataset_name=builder.name))
  if splits_to_update > 0:
    builder.info.write_to_directory(builder.data_dir)
    # If we recover a shard, shard files have to be renamed
    rename_shards(
        data_dir=builder.data_dir,
        split_infos=builder.info.splits,
        ds_name=builder.name,
        check_wrong_shards=True)
  return builder


def rename_shards(data_dir: str,
                  split_infos: tfds.core.SplitDict,
                  ds_name: str,
                  check_wrong_shards=False) -> None:
  """Renames shards in the current data_dir.

  It checks for files without the shard suffix and renames them to include
  `-of-yyyyy`, where `yyyyy` is the number of shards of the split. If
  `check_wrong_shards` is True, it also renames files that end in
  `of-zzzzz`, where `zzzzz` is the number of shards -1.

  It assumes that only one file per shard exists.

  Args:
    data_dir: data containing the split shards.
    split_infos: SplitDict with the shard information.
    ds_name: dataset name.
    check_wrong_shards: renames also files that end in 'num_shards-1'.
  """
  for split_name, split_info in split_infos.items():
    if not split_info.num_examples:
      raise ValueError(
          'Metadata empty. This means that the dataset hasn\'t been generated.')

    filenames = tfds.core.naming.filenames_for_dataset_split(
        dataset_name=ds_name,
        split=split_name,
        num_shards=split_info.num_shards,
        filetype_suffix='tfrecord')
    suffix = f'-of-{split_info.num_shards:05d}'
    wrong_suffix = f'-of-{(split_info.num_shards-1):05d}'
    for f in filenames:
      no_suffix = f[:-len(suffix)]
      old_path = os.path.join(data_dir, no_suffix)
      new_path = os.path.join(data_dir, f)
      if tf.io.gfile.exists(old_path):
        tf.io.gfile.rename(old_path, new_path, overwrite=False)
        logging.info('%s renamed to %s', old_path, new_path)
      elif check_wrong_shards:
        old_path = f'{old_path}{wrong_suffix}'
        if tf.io.gfile.exists(old_path):
          tf.io.gfile.rename(old_path, new_path, overwrite=False)
          logging.info('%s renamed to %s', old_path, new_path)
