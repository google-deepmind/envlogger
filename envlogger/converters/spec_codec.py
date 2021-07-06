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

"""Encoder/decoder for dm_env.specs.Array (and subclasses).
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import dm_env
from dm_env import specs
import numpy as np


_ENVIRONMENT_SPEC_NAMES = [
    'observation_spec',
    'action_spec',
    'reward_spec',
    'discount_spec',
]


def encode_environment_specs(
    env: Optional[dm_env.Environment]) -> Dict[str, Any]:
  """Encodes all the specs from a given environment."""
  if env:
    return {
        'observation_spec': encode(env.observation_spec()),
        'action_spec': encode(env.action_spec()),
        'reward_spec': encode(env.reward_spec()),
        'discount_spec': encode(env.discount_spec()),
    }
  return {}


def decode_environment_specs(
    encoded_specs: Dict[str, Any]) -> Dict[str, Optional[specs.Array]]:
  """Decodes all the specs of an environment."""
  if encoded_specs:
    return {spec_name: decode(encoded_specs[spec_name])
            for spec_name in _ENVIRONMENT_SPEC_NAMES}
  return {spec_name: None for spec_name in _ENVIRONMENT_SPEC_NAMES}


def _array_spec_to_dict(array_spec: specs.Array) -> Dict[str, Any]:
  """Encodes an Array spec as a dictionary."""
  dict_spec = {
      'shape': np.array(array_spec.shape, dtype=np.int64),
      'dtype': str(array_spec.dtype),
      'name': array_spec.name,
  }
  if isinstance(array_spec, specs.BoundedArray):
    dict_spec.update({
        'minimum': array_spec.minimum,
        'maximum': array_spec.maximum,
    })
    if isinstance(array_spec, specs.DiscreteArray):
      dict_spec.update({'num_values': array_spec.num_values})
  return dict_spec


def encode(
    spec: Union[specs.Array, List[Any], Tuple[Any], Dict[str, Any]]
) -> Union[List[Any], Tuple[Any], Dict[str, Any]]:
  """Encodes `spec` using plain Python objects.

  This function supports bare Array specs, lists of Array specs, Tuples of Array
  specs, Dicts of string to Array specs and any combination of these things such
  as Dict[str, Tuple[List[Array, Array]]].

  Args:
    spec: The actual spec to encode.
  Returns:
    The same spec encoded in a way that can be serialized to disk.
  Raises:
    TypeError: When the argument is not among the supported types.
  """
  if isinstance(spec, specs.Array):
    return _array_spec_to_dict(spec)
  if isinstance(spec, list):
    return [encode(x) for x in spec]
  if isinstance(spec, tuple):
    return tuple((encode(x) for x in spec))
  if isinstance(spec, dict):
    return {k: encode(v) for k, v in spec.items()}
  raise TypeError(
      'encode() should be called with an argument of type specs.Array (and '
      f'subclasses), list, tuple or dict. Found {type(spec)}: {spec}.')


def decode(
    spec: Union[List[Any], Tuple[Any], Dict[str, Any]]
) -> Union[specs.Array, List[Any], Tuple[Any], Dict[str, Any]]:
  """Parses `spec` into the supported dm_env spec formats."""
  if isinstance(spec, dict):
    if 'shape' in spec and 'dtype' in spec:
      shape = spec['shape'] if spec['shape'] is not None else ()
      if 'num_values' in spec:
        # DiscreteArray case.
        return specs.DiscreteArray(
            num_values=spec['num_values'],
            dtype=spec['dtype'],
            name=spec['name'])
      elif 'minimum' in spec and 'maximum' in spec:
        # BoundedArray case.
        return specs.BoundedArray(
            shape=shape,
            dtype=spec['dtype'],
            minimum=spec['minimum'],
            maximum=spec['maximum'],
            name=spec['name'])
      else:
        # Base Array spec case.
        return specs.Array(shape=shape, dtype=spec['dtype'], name=spec['name'])
    # Recursively decode array elements.
    return {k: decode(v) for k, v in spec.items()}
  elif isinstance(spec, list):
    return [decode(x) for x in spec]
  elif isinstance(spec, tuple):
    return tuple(decode(x) for x in spec)
  raise TypeError(
      'decode() should be called with an argument of type list, tuple or dict.'
      f' Found: {type(spec)}: {spec}.')
