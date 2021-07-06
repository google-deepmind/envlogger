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

"""Tests for spec_codec."""

from typing import Any, Dict

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from dm_env import specs
from envlogger.converters import spec_codec
import numpy as np


class CustomSpecsEnvironment(dm_env.Environment):
  """An Environment that allows us to customize its specs."""

  def __init__(self,
               observation_spec,
               action_spec,
               reward_spec,
               discount_spec):
    self._observation_spec = observation_spec
    self._action_spec = action_spec
    self._reward_spec = reward_spec
    self._discount_spec = discount_spec

  def reset(self):
    pass

  def step(self, unused_actions):
    pass

  def discount_spec(self):
    return self._discount_spec

  def reward_spec(self):
    return self._reward_spec

  def observation_spec(self):
    return self._observation_spec

  def action_spec(self):
    return self._action_spec


class ArraySpecCodecTest(parameterized.TestCase):

  def _compare_spec_dicts(self, actual: Dict[str, Any], expected: Dict[str,
                                                                       Any]):
    """Checks that `actual` spec dict is equal to `expected`."""
    # Check 'name'.
    self.assertIn(
        'name',
        actual,
        msg=f'`name` must be present in `actual`. Current contents: {actual}')
    self.assertEqual(actual['name'], expected['name'])
    # Check 'dtype'.
    self.assertIn(
        'dtype',
        actual,
        msg=f'`dtype` must be present in `actual`. Current contents: {actual}')
    self.assertEqual(actual['dtype'], expected['dtype'])
    # Check 'shape'.
    self.assertIn(
        'shape',
        actual,
        msg=f'`shape` must be present in `actual`. Current contents: {actual}')
    np.testing.assert_equal(actual['shape'], expected['shape'])
    # If 'minimum' and 'maximum' exist, check that it's equal to `actual`'s.
    if 'minimum' in expected and 'maximum' in expected:
      msg_min = 'Expected actual["minimum"] to be equal to expected["minimum"].'
      msg_max = 'Expected actual["maximum"] to be equal to expected["maximum"].'
      # If dtypes are float we allow for some decimal imprecision.
      if actual['dtype'] == 'float32' or actual['dtype'] == 'float64':
        self.assertAlmostEqual(
            actual['minimum'], expected['minimum'], msg=msg_min)
        self.assertAlmostEqual(
            actual['maximum'], expected['maximum'], msg=msg_max)
      else:
        self.assertEqual(actual['minimum'], expected['minimum'], msg=msg_min)
        self.assertEqual(actual['maximum'], expected['maximum'], msg=msg_max)
    # If 'num_values' is in `expected`, check that it's equal to `actual`'s.
    if 'num_values' in expected:
      self.assertEqual(actual['num_values'], expected['num_values'])

  ##############################################################################
  # encode() tests.
  ##############################################################################

  # Single specs.Array.

  @parameterized.named_parameters(
      ('int', 123),
      ('float', 3.14),
      ('object', object()),
      ('function', np.abs),
      ('module', np),
  )
  def test_encode_unsupported(self, arg):
    """Checks that `encode(unsupported type)` raises a TypeError."""
    self.assertRaises(TypeError, spec_codec.encode, arg)

  @parameterized.named_parameters(
      ('zero_shape_float', specs.Array((), np.float32, 'my_spec'), {
          'shape': np.array([], np.int64),
          'dtype': 'float32',
          'name': 'my_spec',
      }),
      ('zero_shape_int', specs.Array((), np.int32, 'another_spec'), {
          'shape': np.array([], np.int64),
          'dtype': 'int32',
          'name': 'another_spec',
      }),
      # `name` is not required, so this example should also be valid.
      ('zero_shape_float_no_name', specs.Array((), np.float32), {
          'shape': np.array([], np.int64),
          'dtype': 'float32',
          'name': None,
      }),
      ('one_dim_shape_float', specs.Array(
          (123,), np.float32, 'yet_another_spec'), {
              'shape': np.array([123], np.int64),
              'dtype': 'float32',
              'name': 'yet_another_spec',
          }),
      ('one_dim_shape_int', specs.Array((321,), np.int32, 'me_again'), {
          'shape': np.array([321], np.int64),
          'dtype': 'int32',
          'name': 'me_again',
      }),
      ('two_dim_shape_float', specs.Array((1, 2), np.float32, 'still_here'), {
          'shape': np.array([1, 2], np.int64),
          'dtype': 'float32',
          'name': 'still_here',
      }),
      ('two_dim_shape_int', specs.Array((2, 1), np.int32, 'come_on'), {
          'shape': np.array([2, 1], np.int64),
          'dtype': 'int32',
          'name': 'come_on',
      }),
  )
  def test_encode_array(self, input_spec, expected_spec_dict):
    """Checks that we can encode specs.Arrays."""
    self._compare_spec_dicts(spec_codec.encode(input_spec), expected_spec_dict)

  # Single specs.BoundedArray.

  @parameterized.named_parameters(
      ('zero_shape_float',
       specs.BoundedArray(
           (), np.float32, minimum=3.0, maximum=10.0, name='my_spec'), {
               'shape': np.array([], np.int64),
               'dtype': 'float32',
               'name': 'my_spec',
               'minimum': 3.0,
               'maximum': 10.0,
           }),
      ('zero_shape_int',
       specs.BoundedArray(
           (), np.int32, minimum=-100, maximum=100, name='another_spec'), {
               'shape': np.array([], np.int64),
               'dtype': 'int32',
               'name': 'another_spec',
               'minimum': -100,
               'maximum': 100,
           }),
      # `name` is not required, so this example should also be valid.
      ('zero_shape_float_no_name',
       specs.BoundedArray((), np.float32, minimum=0.0, maximum=1.0), {
           'shape': np.array([], np.int64),
           'dtype': 'float32',
           'name': None,
           'minimum': 0.0,
           'maximum': 1.0,
       }),
      ('one_dim_shape_float',
       specs.BoundedArray((123,),
                          np.float32,
                          minimum=123.0,
                          maximum=321.0,
                          name='yet_another_spec'), {
                              'shape': np.array([123], np.int64),
                              'dtype': 'float32',
                              'name': 'yet_another_spec',
                              'minimum': 123.0,
                              'maximum': 321.0,
                          }),
      ('one_dim_shape_int',
       specs.BoundedArray(
           (321,), np.int32, minimum=314, maximum=628, name='me_again'), {
               'shape': np.array([321], np.int64),
               'dtype': 'int32',
               'name': 'me_again',
               'minimum': 314,
               'maximum': 628,
           }),
      ('two_dim_shape_float',
       specs.BoundedArray((1, 2),
                          np.float32,
                          minimum=-1.0 / 12.0,
                          maximum=2.73,
                          name='still_here'), {
                              'shape': np.array([1, 2], np.int64),
                              'dtype': 'float32',
                              'name': 'still_here',
                              'minimum': -1.0 / 12.0,
                              'maximum': 2.73,
                          }),
      ('two_dim_shape_int',
       specs.BoundedArray((2, 1),
                          np.int32,
                          # Notice that sequence minimums/maximums should also
                          # be supported.
                          minimum=[1729],
                          maximum=[4525],
                          name='come_on'), {
                              'shape': np.array([2, 1], np.int64),
                              'dtype': 'int32',
                              'name': 'come_on',
                              'minimum': [1729],
                              'maximum': [4525],
                          }),
  )
  def test_encode_bounded_array(self, input_spec, expected_spec_dict):
    """Checks that we can encode specs.BoundedArrays."""
    self._compare_spec_dicts(
        spec_codec.encode(input_spec), expected_spec_dict)

  # Single specs.DiscreteArray.

  @parameterized.named_parameters(
      ('zero_shape_int', specs.DiscreteArray(
          100, np.int64, name='another_spec'), {
              'shape': np.array([], np.int64),
              'dtype': 'int64',
              'num_values': 100,
              'name': 'another_spec',
          }),
      # `name` is not required, so this example should also be valid.
      ('zero_shape_int_no_name', specs.DiscreteArray(42, np.int32), {
          'shape': np.array([], np.int64),
          'dtype': 'int32',
          'num_values': 42,
          'name': None,
      }),
  )
  def test_encode_discrete_array(self, input_spec, expected_spec_dict):
    """Checks that we can encode specs.DiscreArrays."""
    self._compare_spec_dicts(
        spec_codec.encode(input_spec), expected_spec_dict)

  # Lists of specs.Arrays.

  @parameterized.named_parameters(
      ('empty_list', [], []),
      ('single_spec', [specs.Array((), np.float32, 'my_spec')], [{
          'shape': np.array([], np.int64),
          'dtype': 'float32',
          'name': 'my_spec',
      }]),
      ('two_specs', [
          specs.Array((1, 2, 3), np.float32, 'spec1'),
          specs.Array((3, 2, 1), np.int32, 'spec2')
      ], [{
          'shape': np.array([1, 2, 3], np.int64),
          'dtype': 'float32',
          'name': 'spec1',
      }, {
          'shape': np.array([3, 2, 1], np.int64),
          'dtype': 'int32',
          'name': 'spec2',
      }]),
  )
  def test_encode_list_of_specs(self, input_spec, expected_spec_list):
    """Checks that we can encode lists of Array specs."""
    actual_spec_list = spec_codec.encode(input_spec)
    self.assertLen(actual_spec_list, len(expected_spec_list))
    for actual, expected in zip(actual_spec_list, expected_spec_list):
      self._compare_spec_dicts(actual, expected)

  # Tuples of specs.Arrays.

  @parameterized.named_parameters(
      ('empty_tuple', (), ()),
      ('single_spec', (specs.Array((), np.float32, 'my_spec'),), ({
          'shape': np.array([], np.int64),
          'dtype': 'float32',
          'name': 'my_spec',
      },)),
      ('two_specs', (
          specs.Array((1, 2, 3), np.float32, 'spec1'),
          specs.Array((3, 2, 1), np.int32, 'spec2')
      ), ({
          'shape': np.array([1, 2, 3], np.int64),
          'dtype': 'float32',
          'name': 'spec1',
      }, {
          'shape': np.array([3, 2, 1], np.int64),
          'dtype': 'int32',
          'name': 'spec2',
      })),
  )
  def test_encode_tuple_of_specs(self, input_spec, expected_spec_tuple):
    """Checks that we can encode tuples of Array specs."""
    actual_spec_tuple = spec_codec.encode(input_spec)
    self.assertLen(actual_spec_tuple, len(expected_spec_tuple))
    for actual, expected in zip(actual_spec_tuple, expected_spec_tuple):
      self._compare_spec_dicts(actual, expected)

  # Dicts of specs.Arrays.

  @parameterized.named_parameters(
      ('empty_dict', {}, {}),
      ('single_spec', {
          'my_favorite_spec': specs.Array((), np.float32, 'my_spec')
      }, {
          'my_favorite_spec': {
              'shape': np.array([], np.int64),
              'dtype': 'float32',
              'name': 'my_spec',
          }
      }),
      ('two_specs', {
          'hello': specs.Array((1, 2, 3), np.float32, 'spec1'),
          'world': specs.Array((3, 2, 1), np.int32, 'spec2')
      }, {
          'hello': {
              'shape': np.array([1, 2, 3], np.int64),
              'dtype': 'float32',
              'name': 'spec1',
          },
          'world': {
              'shape': np.array([3, 2, 1], np.int64),
              'dtype': 'int32',
              'name': 'spec2',
          }
      }),
  )
  def test_encode_dict_of_specs(self, input_spec, expected_spec_dict):
    """Checks that we can encode dicts of Array specs."""
    actual_spec_dict = spec_codec.encode(input_spec)
    self.assertLen(actual_spec_dict, len(expected_spec_dict))
    for actual, expected in zip(sorted(actual_spec_dict.items()),
                                sorted(expected_spec_dict.items())):
      actual_key, actual_value = actual
      expected_key, expected_value = expected
      self.assertEqual(actual_key, expected_key)
      self._compare_spec_dicts(actual_value, expected_value)

  ##############################################################################
  # decode() tests.
  ##############################################################################

  @parameterized.named_parameters(
      ('int', 123),
      ('float', 3.14),
      ('object', object()),
      ('function', np.abs),
      ('module', np),
  )
  def test_decode_unsupported(self, arg):
    """Checks that `decode(unsupported type)` raises a TypeError."""
    self.assertRaises(TypeError, spec_codec.decode, arg)

  # Single specs.Arrays.

  @parameterized.named_parameters(
      ('no_shape', {
          'shape': None,  # None shapes are interpreted as scalars.
          'dtype': 'float32',
          'name': 'no_shape_spec',
      }, specs.Array((), np.float32, 'no_shape_spec')),
      ('no_dtype', {
          'shape': (),
          'dtype': None,  # None dtypes are interpreted as float64.
          'name': 'no_dtype_spec',
      }, specs.Array((), np.float64, 'no_dtype_spec')),
      ('no_shape_dtype', {
          'shape': None,  # None shapes are interpreted as scalars.
          'dtype': None,  # None dtypes are interpreted as float64.
          'name': 'no_shape_dtype_spec',
      }, specs.Array((), np.float64, 'no_shape_dtype_spec')),
      ('no_name_float', {
          'shape': np.array([1], np.int64),
          'dtype': 'float32',
          'name': None,  # `name` is optional.
      }, specs.Array((1,), np.float32)),
      ('zero_shape_float', {
          'shape': np.array([], np.int64),
          'dtype': 'float32',
          'name': 'my_spec',
      }, specs.Array((), np.float32, 'my_spec')),
      ('zero_shape_int', {
          'shape': np.array([], np.int64),
          'dtype': 'int64',
          'name': 'int_spec',
      }, specs.Array((), np.int64, 'int_spec')),
      ('one_dim_shape_float', {
          'shape': np.array([123], np.int64),
          'dtype': 'float32',
          'name': 'one_dim_float',
      }, specs.Array((123,), np.float32, 'one_dim_float')),
      ('one_dim_shape_int', {
          'shape': np.array([321], np.int64),
          'dtype': 'int64',
          'name': 'one_dim_int',
      }, specs.Array((321,), np.int64, 'one_dim_int')),
      ('two_dim_shape_float', {
          'shape': np.array([1, 2], np.int64),
          'dtype': 'float32',
          'name': 'two_dim_float',
      }, specs.Array((1, 2), np.float32, 'two_dim_float')),
      ('two_dim_shape_int', {
          'shape': np.array([4, 3], np.int64),
          'dtype': 'int64',
          'name': 'two_dim_int',
      }, specs.Array((4, 3), np.int64, 'two_dim_int')),
  )
  def test_decode_array(self, input_spec_dict, expected_spec):
    result = spec_codec.decode(input_spec_dict)
    self.assertIsInstance(result, specs.Array)
    self.assertEqual(result, expected_spec)

  # Single specs.BoundedArrays.

  @parameterized.named_parameters(
      ('zero_shape_float', {
          'shape': np.array([], np.int64),
          'dtype': 'float32',
          'minimum': 0.0,
          'maximum': 1.0,
          'name': 'my_spec',
      },
       specs.BoundedArray(
           (), np.float32, minimum=0.0, maximum=1.0, name='my_spec')),
      ('zero_shape_int', {
          'shape': np.array([], np.int64),
          'dtype': 'int64',
          'minimum': 0,
          'maximum': 3,
          'name': 'int_spec',
      }, specs.BoundedArray(
          (), np.int64, minimum=0, maximum=3, name='int_spec')),
      ('one_dim_shape_float', {
          'shape': np.array([123], np.int64),
          'dtype': 'float32',
          'minimum': -1.0,
          'maximum': 1.0,
          'name': 'one_dim_float',
      },
       specs.BoundedArray(
           (123,), np.float32, minimum=-1.0, maximum=1.0,
           name='one_dim_float')),
      ('one_dim_shape_int', {
          'shape': np.array([321], np.int64),
          'dtype': 'int64',
          'minimum': 1000,
          'maximum': 2000,
          'name': 'one_dim_int',
      },
       specs.BoundedArray(
           (321,), np.int64, minimum=1000, maximum=2000, name='one_dim_int')),
      # Decoding sequence minimums/maximums should also be supported.
      ('two_dim_shape_float', {
          'shape': np.array([1, 2], np.int64),
          'dtype': 'float32',
          'minimum': [0.0, 5.0],
          'maximum': [1.0, 10.0],
          'name': 'two_dim_float',
      },
       specs.BoundedArray((1, 2),
                          np.float32,
                          minimum=[0.0, 5.0],
                          maximum=[1.0, 10.0],
                          name='two_dim_float')),
      ('two_dim_shape_int', {
          'shape': np.array([4, 3], np.int64),
          'dtype': 'int64',
          'minimum': -10,
          'maximum': 10,
          'name': 'two_dim_int',
      },
       specs.BoundedArray(
           (4, 3), np.int64, minimum=-10, maximum=10, name='two_dim_int')),
      ('no_name_float', {
          'shape': np.array([1], np.int64),
          'dtype': 'float32',
          'minimum': 0.0,
          'maximum': 1.0,
          'name': None,  # `name` is optional.
      }, specs.BoundedArray((1,), np.float32, minimum=0.0, maximum=1.0)),
  )
  def test_decode_bounded_array(self, input_spec_dict, expected_spec):
    result = spec_codec.decode(input_spec_dict)
    self.assertIsInstance(result, specs.BoundedArray)
    self.assertEqual(result, expected_spec)

  # Single specs.DiscreteArrays.

  @parameterized.named_parameters(
      ('zero_shape', {
          'shape': np.array([], np.int64),
          'dtype': 'int32',
          'num_values': 123,
          'name': 'my_spec',
      }, specs.DiscreteArray(123, name='my_spec')),
      ('custom_dtype', {
          'shape': np.array([], np.int64),
          'dtype': 'int64',
          'num_values': 123,
          'name': 'custom_spec',
      }, specs.DiscreteArray(123, dtype=np.int64, name='custom_spec')),
      ('no_name', {
          'shape': np.array([], np.int64),
          'dtype': 'int32',
          'num_values': 666,
          'name': None,  # `name` is optional.
      }, specs.DiscreteArray(666, np.int32)),
  )
  def test_decode_discrete_array(self, input_spec_dict, expected_spec):
    result = spec_codec.decode(input_spec_dict)
    self.assertIsInstance(result, specs.DiscreteArray)
    self.assertEqual(result, expected_spec)

  # Lists of specs.Arrays.

  @parameterized.named_parameters(
      ('empty_list', [], []),
      ('single_spec', [{
          'shape': np.array([], np.int64),
          'dtype': 'float32',
          'name': 'my_spec',
      }], [specs.Array((), np.float32, 'my_spec')]),
      ('two_specs', [{
          'shape': np.array([1, 2, 3], np.int64),
          'dtype': 'float32',
          'name': 'spec1',
      }, {
          'shape': np.array([3, 2, 1], np.int64),
          'dtype': 'int32',
          'name': 'spec2',
      }], [
          specs.Array((1, 2, 3), np.float32, 'spec1'),
          specs.Array((3, 2, 1), np.int32, 'spec2')
      ]),
  )
  def test_decode_list_of_specs(self, input_spec, expected_spec_list):
    """Checks that we can encode lists of Array specs."""
    result = spec_codec.decode(input_spec)
    self.assertIsInstance(result, list)
    self.assertEqual(result, expected_spec_list)

  # Tuples of specs.Arrays.

  @parameterized.named_parameters(
      ('empty_tuple', (), ()),
      ('single_spec', ({
          'shape': np.array([], np.int64),
          'dtype': 'float32',
          'name': 'my_spec',
      },), (specs.Array((), np.float32, 'my_spec'),)),
      ('two_specs', ({
          'shape': np.array([1, 2, 3], np.int64),
          'dtype': 'float32',
          'name': 'spec1',
      }, {
          'shape': np.array([3, 2, 1], np.int64),
          'dtype': 'int32',
          'name': 'spec2',
      }), (
          specs.Array((1, 2, 3), np.float32, 'spec1'),
          specs.Array((3, 2, 1), np.int32, 'spec2')
      )),
  )
  def test_decode_tuple_of_specs(self, input_spec, expected_spec_tuple):
    """Checks that we can encode tuples of Array specs."""
    result = spec_codec.decode(input_spec)
    self.assertIsInstance(result, tuple)
    self.assertEqual(result, expected_spec_tuple)

  # Dicts of specs.Arrays.

  @parameterized.named_parameters(
      ('empty_dict', {}, {}),
      ('single_spec', {
          'my_favorite_spec': {
              'shape': np.array([], np.int64),
              'dtype': 'float32',
              'name': 'my_spec',
          }
      }, {
          'my_favorite_spec': specs.Array((), np.float32, 'my_spec')
      }),
      ('two_specs', {
          'hello': {
              'shape': np.array([1, 2, 3], np.int64),
              'dtype': 'float32',
              'name': 'spec1',
          },
          'world': {
              'shape': np.array([3, 2, 1], np.int64),
              'dtype': 'int32',
              'name': 'spec2',
          }
      }, {
          'hello': specs.Array((1, 2, 3), np.float32, 'spec1'),
          'world': specs.Array((3, 2, 1), np.int32, 'spec2')
      }),
  )
  def test_decode_dict_of_specs(self, input_spec, expected_spec_dict):
    """Checks that we can encode dicts of Array specs."""
    result = spec_codec.decode(input_spec)
    self.assertIsInstance(result, dict)
    self.assertEqual(result, expected_spec_dict)

  @parameterized.named_parameters(
      ('single_spec',
       specs.Array(shape=(1, 2, 3), dtype=np.float64, name='my_3d_spec')),
      (
          'spec_list',
          [
              specs.Array(shape=(1, 2), dtype=np.float32, name='my_2d_spec'),
              specs.BoundedArray(
                  shape=(),
                  dtype=np.uint8,
                  minimum=32,
                  maximum=64,
                  name='scalar_spec')
          ],
      ),
      (
          'spec_tuple',
          (specs.BoundedArray(
              shape=(),
              dtype=np.uint8,
              minimum=32,
              maximum=64,
              name='scalar_spec'),
           specs.Array(shape=(1, 2), dtype=np.float32, name='my_2d_spec')),
      ),
      (
          'spec_dict',
          {
              'spec1':
                  specs.BoundedArray(
                      shape=(),
                      dtype=np.uint8,
                      minimum=32,
                      maximum=64,
                      name='scalar_spec'),
              'spec2':
                  specs.Array(
                      shape=(1, 2), dtype=np.float32, name='my_2d_spec'),
          },
      ),
      # Any combination of tuples, lists and dicts should be supported.
      (
          'complicated_spec',
          {
              'spec1': [
                  specs.BoundedArray(
                      shape=(),
                      dtype=np.uint8,
                      minimum=32,
                      maximum=64,
                      name='scalar_spec'),
                  specs.Array(
                      shape=(1, 2), dtype=np.float32, name='my_2d_spec')
              ],
              'spec2': (specs.Array(
                  shape=(1, 2), dtype=np.float32, name='my_2d_spec'), {
                      'deeply_nested':
                          specs.DiscreteArray(
                              num_values=999, name='hard_to_find')
                  }),
          },
      ),
  )
  def test_roundtrip_encoding_decoding(self, input_spec):
    self.assertEqual(
        spec_codec.decode(spec_codec.encode(input_spec)),
        input_spec)

  def test_environment_specs_roundtrip(self):
    """Checks that {encode|decode}_environment_specs work correctly.
    """
    # Each spec has a different shape, type and name
    observation_spec = specs.Array((1, 2, 3), np.float32, 'spec1')
    action_spec = specs.Array((4, 5), np.float64, 'spec2')
    reward_spec = specs.Array((1,), np.int32, 'spec3')
    discount_spec = specs.Array((2,), np.int64, 'spec4')

    env = CustomSpecsEnvironment(observation_spec, action_spec, reward_spec,
                                 discount_spec)

    env_specs = spec_codec.encode_environment_specs(env)

    decoded_specs = spec_codec.decode_environment_specs(env_specs)
    self.assertEqual(decoded_specs['observation_spec'], observation_spec)
    self.assertEqual(decoded_specs['action_spec'], action_spec)
    self.assertEqual(decoded_specs['reward_spec'], reward_spec)
    self.assertEqual(decoded_specs['discount_spec'], discount_spec)

  def test_environment_specs_roundtrip_no_env(self):
    """Checks that {encode|decode}_environment_specs with no environment.
    """
    env_specs = spec_codec.encode_environment_specs(None)

    decoded_specs = spec_codec.decode_environment_specs(env_specs)
    self.assertIsNone(decoded_specs['observation_spec'])
    self.assertIsNone(decoded_specs['action_spec'])
    self.assertIsNone(decoded_specs['reward_spec'])
    self.assertIsNone(decoded_specs['discount_spec'])


if __name__ == '__main__':
  absltest.main()
