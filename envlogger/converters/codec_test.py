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

"""Tests for codec."""

from absl.testing import absltest
from absl.testing import parameterized
from envlogger.converters import codec
from envlogger.proto import storage_pb2
import numpy as np


class NumpyConvertersTest(parameterized.TestCase):

  ##############################################################################
  #
  # Datum tests (i.e. not Array/Tuple/Dict of Datums)
  #
  ##############################################################################

  ##############################################################################
  #
  # Scalar tests
  #
  ##############################################################################

  ##############################################################################
  # Empty and None values
  ##############################################################################

  def test_encode_none(self):
    """The proto should be completely empty if given a None value."""
    self.assertEqual(codec.encode(None), storage_pb2.Data())

  def test_decode_none(self):
    """Decoding a None value should produce None."""
    self.assertIsNone(codec.decode(None))

  def test_decode_empty_proto(self):
    """Decoding an empty proto should produce None."""
    user_data = storage_pb2.Data()
    self.assertIsNone(codec.decode(user_data))

  def test_encode_empty_ndarray(self):
    """The proto should be completely empty if given zero shape numpy array."""
    self.assertEqual(codec.encode(np.array([])), storage_pb2.Data())
    # Also test other explicit types.
    self.assertEqual(
        codec.encode(np.array([], dtype='float')), storage_pb2.Data())

  def test_identity_none(self):
    """Encoding and decoding it back should not change its value."""
    self.assertIsNone(codec.decode(codec.encode(None)))

  ##############################################################################
  # float32
  ##############################################################################

  def test_encode_32bit_float_scalar(self):
    """Proto supports float32 so we expect no precision loss in encoding."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.float_values.append(np.float32(3.14))
    datum.shape.dim.add().size = -438
    self.assertEqual(codec.encode(np.float32(3.14)), expected)

  def test_decode_32bit_float_scalar(self):
    """Proto supports float32 so we expect no precision loss in decoding."""
    user_data = storage_pb2.Data()
    datum = user_data.datum
    datum.values.float_values.append(np.float32(3.14))
    datum.shape.dim.add().size = -438
    decoded = codec.decode(user_data)
    self.assertTrue(
        np.isscalar(decoded), 'The returned data should be a plain scalar.\n'
        f'Actual type: {type(decoded)}\n'
        f'user_data: {user_data}\n'
        f'decoded: {decoded}')
    self.assertIsInstance(decoded, np.float32)
    self.assertEqual(decoded, np.float32(3.14))

  def test_identity_32bit_float_scalar(self):
    """Encoding and decoding it back should not change its value."""
    decoded = codec.decode(codec.encode(np.float32(3.14)))
    self.assertIsInstance(decoded, np.float32)
    self.assertEqual(decoded, np.float32(3.14))

  ##############################################################################
  # float32 buffer
  ##############################################################################

  def test_decode_32bit_float_scalar_buffer(self):
    """Proto supports float32 so we expect no precision loss in decoding."""
    user_data = storage_pb2.Data()
    datum = user_data.datum
    # 3.14159 in big-endian byte array.
    datum.values.float_values_buffer = b'\x40\x49\x0f\xd0'
    datum.shape.dim.add().size = -438
    decoded = codec.decode(user_data)
    self.assertTrue(
        np.isscalar(decoded), 'The returned data should be a plain scalar.\n'
        f'Actual type: {type(decoded)}\n'
        f'user_data: {user_data}\n'
        f'decoded: {decoded}')
    self.assertIsInstance(decoded, np.float32)
    self.assertEqual(decoded, np.float32(3.14159))

  ##############################################################################
  # float64 (aka double)
  ##############################################################################

  def test_encode_double_scalar(self):
    """Proto supports double so we expect no precision loss in encoding."""
    # Ordinary floats in python are 64-bit floats.
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.double_values.append(3.14159265358979)
    datum.shape.dim.add().size = -438
    self.assertEqual(codec.encode(3.14159265358979), expected)
    # np.float64 should also work.
    self.assertEqual(codec.encode(np.float64(3.14159265358979)), expected)

  def test_decode_double_scalar(self):
    """Proto supports double so we expect no precision loss in decoding."""
    user_data = storage_pb2.Data()
    datum = user_data.datum
    datum.values.double_values.append(3.14159265358979)
    datum.shape.dim.add().size = -438
    decoded = codec.decode(user_data)
    self.assertTrue(
        np.isscalar(decoded), 'The returned data should be a plain scalar.\n'
        f'Actual type: {type(decoded)}\n'
        f'user_data: {user_data}\n'
        f'decoded: {decoded}')
    self.assertIsInstance(decoded, np.float64)
    self.assertEqual(decoded, np.float64(3.14159265358979))

  def test_identity_double_scalar(self):
    """Encoding and decoding it back should not change its value."""
    decoded = codec.decode(codec.encode(np.float64(3.14159265358979)))
    self.assertIsInstance(decoded, np.float64)
    self.assertEqual(decoded, np.float64(3.14159265358979))

  ##############################################################################
  # int32
  ##############################################################################

  def test_encode_int32_scalar(self):
    """Proto supports int32 so we expect no precision loss in encoding."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.int32_values.append(np.int32(3))
    datum.shape.dim.add().size = -438
    self.assertEqual(codec.encode(np.int32(3)), expected)

  def test_decode_int32_scalar(self):
    """Proto supports int32 so we expect no precision loss in encoding."""
    user_data = storage_pb2.Data()
    datum = user_data.datum
    datum.values.int32_values.append(np.int32(-32))
    datum.shape.dim.add().size = -438
    decoded = codec.decode(user_data)
    self.assertTrue(
        np.isscalar(decoded), 'The returned data should be a plain scalar.\n'
        f'Actual type: {type(decoded)}\n'
        f'user_data: {user_data}\n'
        f'decoded: {decoded}')
    self.assertIsInstance(decoded, np.int32)
    self.assertEqual(decoded, np.int32(-32))

  def test_identity_int32_scalar(self):
    """Encoding and decoding it back should not change its value."""
    decoded = codec.decode(codec.encode(np.int32(-3)))
    self.assertIsInstance(decoded, np.int32)
    self.assertEqual(decoded, np.int32(-3))

  ##############################################################################
  # int64
  ##############################################################################

  def test_encode_int64_scalar(self):
    """Proto supports int64 so we expect no precision loss in encoding."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.int64_values.append(np.int64(-3))
    datum.shape.dim.add().size = -438
    self.assertEqual(codec.encode(np.int64(-3)), expected)

  def test_decode_int64_scalar(self):
    """Proto supports int64 so we expect no precision loss in decoding."""
    user_data = storage_pb2.Data()
    datum = user_data.datum
    datum.values.int64_values.append(np.int64(-64))
    datum.shape.dim.add().size = -438
    decoded = codec.decode(user_data)
    self.assertTrue(
        np.isscalar(decoded), 'The returned data should be a plain scalar.\n'
        f'Actual type: {type(decoded)}\n'
        f'user_data: {user_data}\n'
        f'decoded: {decoded}')
    self.assertIsInstance(decoded, np.int64)
    self.assertEqual(decoded, np.int64(-64))

  def test_identity_int64_scalar(self):
    """Encoding and decoding it back should not change its value."""
    decoded = codec.decode(codec.encode(np.int64(-1234567890123)))
    self.assertIsInstance(decoded, np.int64)
    self.assertEqual(decoded, np.int64(-1234567890123))

  ##############################################################################
  # uint32
  ##############################################################################

  def test_encode_uint32_scalar(self):
    """Proto supports uint32 so we expect no precision loss in encoding."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.uint32_values.append(np.uint32(12345))
    datum.shape.dim.add().size = -438
    self.assertEqual(codec.encode(np.uint32(12345)), expected)

  def test_decode_uint32_scalar(self):
    """Proto supports uint32 so we expect no precision loss in decoding."""
    user_data = storage_pb2.Data()
    datum = user_data.datum
    datum.values.uint32_values.append(np.uint32(32))
    datum.shape.dim.add().size = -438
    decoded = codec.decode(user_data)
    self.assertTrue(
        np.isscalar(decoded), 'The returned data should be a plain scalar.\n'
        f'Actual type: {type(decoded)}\n'
        f'user_data: {user_data}\n'
        f'decoded: {decoded}')
    self.assertIsInstance(decoded, np.uint32)
    self.assertEqual(decoded, np.uint32(32))

  def test_identity_uint32_scalar(self):
    """Encoding and decoding it back should not change its value."""
    decoded = codec.decode(codec.encode(np.uint32(4294967295)))
    self.assertIsInstance(decoded, np.uint32)
    self.assertEqual(decoded, np.uint32(4294967295))

  ##############################################################################
  # uint64
  ##############################################################################

  def test_encode_uint64_scalar(self):
    """Proto supports uint64 so we expect no precision loss in encoding."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.uint64_values.append(np.uint64(12345))
    datum.shape.dim.add().size = -438
    self.assertEqual(codec.encode(np.uint64(12345)), expected)

  def test_decode_uint64_scalar(self):
    """Proto supports uint64 so we expect no precision loss in decoding."""
    user_data = storage_pb2.Data()
    datum = user_data.datum
    datum.values.uint64_values.append(np.uint64(64))
    datum.shape.dim.add().size = -438
    decoded = codec.decode(user_data)
    self.assertTrue(
        np.isscalar(decoded), 'The returned data should be a plain scalar.\n'
        f'Actual type: {type(decoded)}\n'
        f'user_data: {user_data}\n'
        f'decoded: {decoded}')
    self.assertIsInstance(decoded, np.uint64)
    self.assertEqual(decoded, np.uint64(64))

  def test_identity_uint64_scalar(self):
    """Encoding and decoding it back should not change its value."""
    decoded = codec.decode(codec.encode(np.uint64(18446744073709551615)))
    self.assertIsInstance(decoded, np.uint64)
    self.assertEqual(decoded, np.uint64(18446744073709551615))

  ##############################################################################
  # bool
  ##############################################################################

  def test_encode_bool_scalar(self):
    """Proto supports bool so we expect no precision loss in encoding."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.bool_values.append(True)
    datum.shape.dim.add().size = -438
    self.assertEqual(codec.encode(True), expected)
    # Numpy's booleans should also work.
    self.assertEqual(codec.encode(np.bool(True)), expected)
    self.assertEqual(codec.encode(np.bool_(True)), expected)

  def test_decode_bool_scalar(self):
    """Proto supports bool so we expect no precision loss in decoding."""
    user_data = storage_pb2.Data()
    datum = user_data.datum
    datum.values.bool_values.append(True)
    datum.shape.dim.add().size = -438
    decoded = codec.decode(user_data)
    self.assertTrue(
        np.isscalar(decoded), 'The returned data should be a plain scalar.\n'
        f'Actual type: {type(decoded)}\n'
        f'user_data: {user_data}\n'
        f'decoded: {decoded}')
    self.assertEqual(decoded, True)

  def test_identity_bool_scalar_true(self):
    """Encoding and decoding it back should not change its value."""
    decoded = codec.decode(codec.encode(True))
    self.assertIsInstance(decoded, bool)
    self.assertEqual(decoded, True)
    # Numpy's booleans should also work, but they all become Python bools.
    decoded = codec.decode(codec.encode(np.bool_(True)))
    self.assertIsInstance(decoded, bool)
    self.assertEqual(decoded, True)
    decoded = codec.decode(codec.encode(np.bool(True)))
    self.assertIsInstance(decoded, bool)
    self.assertEqual(decoded, True)

  def test_identity_bool_scalar_false(self):
    """Encoding and decoding it back should not change its value."""
    decoded = codec.decode(codec.encode(False))
    self.assertIsInstance(decoded, bool)
    self.assertEqual(decoded, False)

  ##############################################################################
  # string
  ##############################################################################

  def test_encode_string_scalar(self):
    """Proto supports string so we expect no loss in encoding."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.string_values.append('pi')
    datum.shape.dim.add().size = -438
    self.assertEqual(codec.encode('pi'), expected)

  def test_decode_string_scalar(self):
    """Proto supports string so we expect no loss in decoding."""
    user_data = storage_pb2.Data()
    datum = user_data.datum
    datum.values.string_values.append('ravel')
    datum.shape.dim.add().size = -438
    decoded = codec.decode(user_data)
    self.assertTrue(
        np.isscalar(decoded), 'The returned data should be a plain scalar.\n'
        f'Actual type: {type(decoded)}\n'
        f'user_data: {user_data}\n'
        f'decoded: {decoded}')
    self.assertIsInstance(decoded, str)
    self.assertEqual(decoded, 'ravel')

  def test_identity_string_scalar(self):
    """Encoding and decoding it back should not change its value."""
    decoded = codec.decode(codec.encode('do not change me, please!'))
    self.assertIsInstance(decoded, str)
    self.assertEqual(decoded, 'do not change me, please!')

  ##############################################################################
  # bytes
  ##############################################################################

  def test_encode_bytes_scalar(self):
    """Proto supports bytes so we expect no precision loss in encoding."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.bytes_values.append(b'pi')
    datum.shape.dim.add().size = -438
    self.assertEqual(codec.encode(b'pi'), expected)

  def test_decode_bytes_scalar(self):
    """Proto supports bytes so we expect no precision loss in decoding."""
    user_data = storage_pb2.Data()
    datum = user_data.datum
    datum.values.bytes_values.append(b'xu xin')
    datum.shape.dim.add().size = -438
    decoded = codec.decode(user_data)
    self.assertTrue(
        np.isscalar(decoded), 'The returned data should be a plain scalar.\n'
        f'Actual type: {type(decoded)}\n'
        f'user_data: {user_data}\n'
        f'decoded: {decoded}')
    self.assertIsInstance(decoded, bytes)
    self.assertEqual(decoded, b'xu xin')

  def test_identity_bytes_scalar(self):
    """Encoding and decoding it back should not change its value."""
    decoded = codec.decode(codec.encode(b'awesome bytes'))
    self.assertIsInstance(decoded, bytes)
    self.assertEqual(decoded, b'awesome bytes')

  ##############################################################################
  # big int (arbitrarily long)
  ##############################################################################

  def test_encode_int_small_scalar(self):
    """Ensures that a vanilla Python int can be stored as bytes."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.bigint_values.append(b'\x03')
    datum.shape.dim.add().size = -438
    self.assertEqual(codec.encode(3), expected)

  def test_encode_bigint_scalar(self):
    """Ensures that a large Python int can be stored as bytes."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.bigint_values.append(
        b'\x01\x8e\xe9\x0f\xf6\xc3s\xe0\xeeN?\n\xd2')
    datum.shape.dim.add().size = -438
    self.assertEqual(codec.encode(123456789012345678901234567890), expected)

  def test_encode_negative_bigint_scalar(self):
    """Ensures that a large negative Python int can be stored as bytes."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.bigint_values.append(
        b'\xfeq\x16\xf0\t<\x8c\x1f\x11\xb1\xc0\xf5.')
    datum.shape.dim.add().size = -438
    self.assertEqual(codec.encode(-123456789012345678901234567890), expected)

  def test_decode_int_scalar(self):
    """Ensures that a large negative integer can be decoded to a Python int."""
    user_data = storage_pb2.Data()
    datum = user_data.datum
    datum.values.bigint_values.append(
        b'\xfeq\x16\xf0\t<\x8c\x1f\x11\xb1\xc0\xf5.')
    datum.shape.dim.add().size = -438
    decoded = codec.decode(user_data)
    self.assertTrue(
        np.isscalar(decoded), 'The returned data should be a plain scalar.\n'
        f'Actual type: {type(decoded)}\n'
        f'user_data: {user_data}\n'
        f'decoded: {decoded}')
    self.assertIsInstance(decoded, int)
    self.assertEqual(decoded, -123456789012345678901234567890)

  def test_identity_int_scalar_positive(self):
    """Encoding and decoding it back should not change its value."""
    decoded = codec.decode(codec.encode(12345678901234567890))
    self.assertIsInstance(decoded, int)
    self.assertEqual(decoded, 12345678901234567890)

  def test_identity_int_scalar_zero(self):
    """Encoding and decoding it back should not change its value."""
    decoded = codec.decode(codec.encode(0))
    self.assertIsInstance(decoded, int)
    self.assertEqual(decoded, 0)

  def test_identity_int_scalar_negative(self):
    """Encoding and decoding it back should not change its value."""
    decoded = codec.decode(codec.encode(-98765432109876543210))
    self.assertIsInstance(decoded, int)
    self.assertEqual(decoded, -98765432109876543210)

  ##############################################################################
  # int8
  ##############################################################################

  def test_encode_int8_scalar(self):
    """Ensures that an np.int8 can be stored as bytes."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.int8_values = b'\x03'
    datum.shape.dim.add().size = -438
    self.assertEqual(codec.encode(np.int8(3)), expected)

  def test_decode_int8_scalar(self):
    """Ensures that int8s can be retrieved as np.int8."""
    user_data = storage_pb2.Data()
    datum = user_data.datum
    datum.values.int8_values = b'\xfd'
    datum.shape.dim.add().size = -438
    decoded = codec.decode(user_data)
    self.assertTrue(
        np.isscalar(decoded), 'The returned data should be a plain scalar.\n'
        f'Actual type: {type(decoded)}\n'
        f'user_data: {user_data}\n'
        f'decoded: {decoded}')
    self.assertIsInstance(decoded, np.int8)
    self.assertEqual(decoded, np.int8(-3))

  def test_identity_int8_scalar_negative(self):
    """Encoding and decoding it back should not change its value."""
    decoded = codec.decode(codec.encode(np.int8(-123)))
    self.assertIsInstance(decoded, np.int8)
    self.assertEqual(decoded, np.int8(-123))

  def test_identity_int8_scalar_zero(self):
    """Encoding and decoding it back should not change its value."""
    decoded = codec.decode(codec.encode(np.int8(0)))
    self.assertIsInstance(decoded, np.int8)
    self.assertEqual(decoded, np.int8(0))

  def test_identity_int8_scalar_positive(self):
    """Encoding and decoding it back should not change its value."""
    decoded = codec.decode(codec.encode(np.int8(127)))
    self.assertIsInstance(decoded, np.int8)
    self.assertEqual(decoded, np.int8(127))

  ##############################################################################
  # int16
  ##############################################################################

  def test_encode_int16_scalar(self):
    """Ensures that an np.int16 can be stored as bytes."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.int16_values = b'\xfe\xd4'
    datum.shape.dim.add().size = -438
    self.assertEqual(codec.encode(np.int16(-300)), expected)

  def test_decode_int16_scalar(self):
    """Ensures that int16s can be retrieved as np.int16."""
    user_data = storage_pb2.Data()
    datum = user_data.datum
    datum.values.int16_values = b'\x07\xd0'
    datum.shape.dim.add().size = -438
    decoded = codec.decode(user_data)
    self.assertTrue(
        np.isscalar(decoded), 'The returned data should be a plain scalar.\n'
        f'Actual type: {type(decoded)}\n'
        f'user_data: {user_data}\n'
        f'decoded: {decoded}')
    self.assertIsInstance(decoded, np.int16)
    self.assertEqual(decoded, np.int16(2000))

  def test_identity_int16_scalar_negative(self):
    """Encoding and decoding it back should not change its value."""
    decoded = codec.decode(codec.encode(np.int16(-123)))
    self.assertIsInstance(decoded, np.int16)
    self.assertEqual(decoded, np.int16(-123))

  def test_identity_int16_scalar_zero(self):
    """Encoding and decoding it back should not change its value."""
    decoded = codec.decode(codec.encode(np.int16(0)))
    self.assertIsInstance(decoded, np.int16)
    self.assertEqual(decoded, np.int16(0))

  def test_identity_int16_scalar_positive(self):
    """Encoding and decoding it back should not change its value."""
    decoded = codec.decode(codec.encode(np.int16(127)))
    self.assertIsInstance(decoded, np.int16)
    self.assertEqual(decoded, np.int16(127))

  ##############################################################################
  # uint8
  ##############################################################################

  def test_encode_uint8_scalar(self):
    """Ensures that an np.uint8 can be stored as bytes."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.uint8_values = b'\xfb'
    datum.shape.dim.add().size = -438
    self.assertEqual(codec.encode(np.uint8(251)), expected)

  def test_decode_uint8_scalar(self):
    user_data = storage_pb2.Data()
    datum = user_data.datum
    datum.values.uint8_values = b'\xed'
    datum.shape.dim.add().size = -438
    decoded = codec.decode(user_data)
    self.assertTrue(
        np.isscalar(decoded), 'The returned data should be a plain scalar.\n'
        f'Actual type: {type(decoded)}\n'
        f'user_data: {user_data}\n'
        f'decoded: {decoded}')
    self.assertIsInstance(decoded, np.uint8)
    self.assertEqual(decoded, np.uint8(237))

  def test_identity_uint8_scalar_zero(self):
    """Encoding and decoding it back should not change its value."""
    decoded = codec.decode(codec.encode(np.uint8(0)))
    self.assertIsInstance(decoded, np.uint8)
    self.assertEqual(decoded, np.uint8(0))

  def test_identity_uint8_scalar_positive(self):
    """Encoding and decoding it back should not change its value."""
    decoded = codec.decode(codec.encode(np.uint8(255)))
    self.assertIsInstance(decoded, np.uint8)
    self.assertEqual(decoded, np.uint8(255))

  ##############################################################################
  # uint16
  ##############################################################################

  def test_encode_uint16_scalar(self):
    """Ensures that an np.uint16 can be stored as bytes."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.uint16_values = b'\x03\xe8'
    datum.shape.dim.add().size = -438
    self.assertEqual(codec.encode(np.uint16(1000)), expected)

  def test_decode_uint16_scalar(self):
    user_data = storage_pb2.Data()
    datum = user_data.datum
    datum.values.uint16_values = b'\x0b\xb8'
    datum.shape.dim.add().size = -438
    decoded = codec.decode(user_data)
    self.assertTrue(
        np.isscalar(decoded), 'The returned data should be a plain scalar.\n'
        f'Actual type: {type(decoded)}\n'
        f'user_data: {user_data}\n'
        f'decoded: {decoded}')
    self.assertIsInstance(decoded, np.uint16)
    self.assertEqual(decoded, np.uint16(3000))

  def test_identity_uint16_scalar_zero(self):
    """Encoding and decoding it back should not change its value."""
    decoded = codec.decode(codec.encode(np.uint16(0)))
    self.assertIsInstance(decoded, np.uint16)
    self.assertEqual(decoded, np.uint16(0))

  def test_identity_uint16_scalar_positive(self):
    """Encoding and decoding it back should not change its value."""
    decoded = codec.decode(codec.encode(np.uint16(12345)))
    self.assertIsInstance(decoded, np.uint16)
    self.assertEqual(decoded, np.uint16(12345))

  ##############################################################################
  #
  # Array tests
  #
  ##############################################################################

  ##############################################################################
  # Empty and None values
  ##############################################################################

  def test_encode_empty_list(self):
    """Tests that a Python list of one None element is represented by an Array."""
    expected = storage_pb2.Data()
    self.assertEqual(codec.encode([]), expected)

  def test_encode_none_list(self):
    """Tests that a Python list of one None element is represented by an Array."""
    expected = storage_pb2.Data()
    expected.array.values.add()
    self.assertEqual(codec.encode([None]), expected)

  def test_encode_two_none_list(self):
    """Tests that a Python list of one None element is represented by an Array."""
    expected = storage_pb2.Data()
    expected.array.values.add()
    expected.array.values.add()
    self.assertEqual(codec.encode([None, None]), expected)

  def test_encode_decode_empty_list(self):
    """Tests that an empty Python list becomes None when decoded."""
    self.assertIsNone(codec.decode(codec.encode([])), None)

  ##############################################################################
  # float32
  ##############################################################################

  def test_encode_float32_list(self):
    """Tests that a Python list of one element is represented by an Array."""
    expected = storage_pb2.Data()
    datum = expected.array.values.add().datum
    datum.values.float_values.append(np.float32(3.14))
    datum.shape.dim.add().size = -438
    self.assertEqual(codec.encode([np.float32(3.14)]), expected)

  def test_decode_float32_list(self):
    """Tests that we get a Python list from a proto Array."""
    user_data = storage_pb2.Data()
    datum = user_data.array.values.add().datum
    datum.values.float_values.append(np.float32(3.14))
    datum.shape.dim.add().size = -438
    decoded = codec.decode(user_data)
    self.assertNotEmpty(decoded)
    self.assertIsInstance(decoded[0], np.float32)
    self.assertListEqual(decoded, [np.float32(3.14)])

  def test_encode_float32_nested_list(self):
    """Ensures that [[1.2, 3.4], [5.6, 7.8]] is represented correctly."""
    expected = storage_pb2.Data()
    array1 = expected.array.values.add().array
    datum1 = array1.values.add().datum
    datum1.values.float_values.append(np.float32(1.2))
    datum1.shape.dim.add().size = -438
    datum2 = array1.values.add().datum
    datum2.values.float_values.append(np.float32(3.4))
    datum2.shape.dim.add().size = -438

    array2 = expected.array.values.add().array
    datum3 = array2.values.add().datum
    datum3.values.float_values.append(np.float32(5.6))
    datum3.shape.dim.add().size = -438
    datum4 = array2.values.add().datum
    datum4.values.float_values.append(np.float32(7.8))
    datum4.shape.dim.add().size = -438

    self.assertEqual(
        codec.encode([[np.float32(1.2), np.float32(3.4)],
                      [np.float32(5.6), np.float32(7.8)]]), expected)

  ##############################################################################
  # float64
  ##############################################################################

  def test_encode_float64_list(self):
    """Tests that a Python list of one element is represented by an Array."""
    expected = storage_pb2.Data()
    datum = expected.array.values.add().datum
    datum.values.double_values.append(np.float64(6.28))
    datum.shape.dim.add().size = -438
    self.assertEqual(codec.encode([np.float64(6.28)]), expected)

  def test_decode_float64_list(self):
    """Tests that we get a Python list from a proto Array."""
    user_data = storage_pb2.Data()
    datum = user_data.array.values.add().datum
    datum.values.double_values.append(np.float64(6.28))
    datum.shape.dim.add().size = -438
    decoded = codec.decode(user_data)
    self.assertNotEmpty(decoded)
    self.assertIsInstance(decoded[0], np.float64)
    self.assertListEqual(decoded, [np.float64(6.28)])

  ##############################################################################
  # int32
  ##############################################################################

  def test_encode_int32_list(self):
    """Tests that a Python list of one element is represented by an Array."""
    expected = storage_pb2.Data()
    datum = expected.array.values.add().datum
    datum.values.int32_values.append(np.int32(-12345))
    datum.shape.dim.add().size = -438
    self.assertEqual(codec.encode([np.int32(-12345)]), expected)

  def test_decode_int32_list(self):
    """Tests that a Python list of one element is represented by an Array."""
    user_data = storage_pb2.Data()
    datum = user_data.array.values.add().datum
    datum.values.int32_values.append(np.int32(-12345))
    datum.shape.dim.add().size = -438
    decoded = codec.decode(user_data)
    self.assertNotEmpty(decoded)
    self.assertIsInstance(decoded[0], np.int32)
    self.assertListEqual(decoded, [np.int32(-12345)])

  ##############################################################################
  # int64
  ##############################################################################

  def test_encode_int64_list(self):
    """Tests that a Python list of one element is represented by an Array."""
    expected = storage_pb2.Data()
    datum = expected.array.values.add().datum
    datum.values.int64_values.append(np.int64(-1234567890123456))
    datum.shape.dim.add().size = -438
    self.assertEqual(codec.encode([np.int64(-1234567890123456)]), expected)

  def test_decode_int64_list(self):
    """Tests that a Python list of one element is represented by an Array."""
    user_data = storage_pb2.Data()
    datum = user_data.array.values.add().datum
    datum.values.int64_values.append(np.int64(-1234567890123456))
    datum.shape.dim.add().size = -438
    decoded = codec.decode(user_data)
    self.assertNotEmpty(decoded)
    self.assertIsInstance(decoded[0], np.int64)
    self.assertListEqual(decoded, [np.int64(-1234567890123456)])

  # Homogeneity.

  def test_encode_heterogeneous_list(self):
    """Tests that an error is thrown for a list with different types."""
    user_data = [np.int64(-1234567890123456), np.int32(1)]
    self.assertRaises(TypeError, codec.encode, user_data)

  ##############################################################################
  #
  # ndarray tests
  #
  ##############################################################################

  def test_encode_one_float_elem_scalar_ndarray(self):
    """Ensures that np arrays with shape 0 can be encoded in our proto."""
    a = np.array(1.5, dtype=np.float32)
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.float_values_buffer = a.astype('>f').tobytes()
    self.assertEqual(codec.encode(a), expected)

  def test_encode_one_float_elem_ndarray(self):
    """Ensures that np float32 arrays can be encoded in our proto."""
    a = np.array([1.5], dtype=np.float32)
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.shape.dim.add().size = 1
    datum.values.float_values_buffer = a.astype('>f').tobytes()
    self.assertEqual(codec.encode(a), expected)

  def test_identity_one_float_elem_ndarray(self):
    """Ensures that np float32 arrays can be written and read back."""
    a = np.array(1.5, dtype=np.float32)
    np.testing.assert_equal(codec.decode(codec.encode(a)), a)

  def test_decode_one_float_elem_ndarray(self):
    """Once encoded, the proto should be decodeable."""
    user_data = storage_pb2.Data()
    user_data.datum.shape.dim.add().size = 1
    user_data.datum.values.float_values.append(0.1512)
    np.testing.assert_equal(
        codec.decode(user_data), np.array([0.1512], dtype=np.float32))

  def test_decode_one_float_elem_ndarray_buffer(self):
    """Tests that we get a Python list from a float32 buffer."""
    user_data = storage_pb2.Data()
    user_data.datum.shape.dim.add().size = 1
    # 3.141519 in big-endian byte array.
    user_data.datum.values.float_values_buffer = b'\x40\x49\x0f\xd0'
    decoded = codec.decode(user_data)
    self.assertEqual(decoded.dtype, np.float32)
    np.testing.assert_equal(decoded, np.array([3.14159], dtype=np.float32))

  def test_encode_one_double_elem_scalar_ndarray(self):
    """Ensures that np arrays with shape 0 can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.double_values.append(512.123)
    self.assertEqual(
        codec.encode(np.array(512.123, dtype=np.float64)), expected)

  def test_encode_one_double_elem_ndarray(self):
    """Ensures that np float64 arrays can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.shape.dim.add().size = 1
    datum.values.double_values.append(512.123)
    self.assertEqual(
        codec.encode(np.array([512.123], dtype=np.float64)), expected)

  def test_decode_one_double_elem_ndarray(self):
    """Once encoded, the proto should be decodeable."""
    user_data = storage_pb2.Data()
    user_data.datum.shape.dim.add().size = 1
    user_data.datum.values.double_values.append(0.63661)
    np.testing.assert_equal(
        codec.decode(user_data), np.array([0.63661], dtype=np.float64))

  def test_encode_multiple_double_elem_ndarray(self):
    """Ensures that np float64 multi-element arrays can be encoded."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.shape.dim.add().size = 2
    datum.values.double_values.extend([987.654, 321.098])
    self.assertEqual(codec.encode(np.array([987.654, 321.098])), expected)

  def test_decode_multiple_double_elem_ndarray(self):
    """Once encoded, the proto should be decodeable."""
    user_data = storage_pb2.Data()
    user_data.datum.shape.dim.add().size = 3
    user_data.datum.values.double_values.extend([0.74048, 2.09455, 0.69314])
    np.testing.assert_equal(
        codec.decode(user_data),
        np.array([0.74048, 2.09455, 0.69314], dtype=np.float64))

  def test_encode_one_int32_elem_scalar_ndarray(self):
    """Ensures that np arrays with shape 0 can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.int32_values.append(415)
    self.assertEqual(codec.encode(np.array(415, dtype=np.int32)), expected)

  def test_encode_one_int32_elem_ndarray(self):
    """Ensures that np int32 arrays can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.shape.dim.add().size = 1
    datum.values.int32_values.append(415)
    self.assertEqual(codec.encode(np.array([415], dtype=np.int32)), expected)

  def test_decode_one_int32_elem_ndarray(self):
    """Once encoded, the proto should be decodeable."""
    user_data = storage_pb2.Data()
    user_data.datum.shape.dim.add().size = 1
    user_data.datum.values.int32_values.append(9)
    np.testing.assert_equal(
        codec.decode(user_data), np.array([9], dtype=np.int32))

  def test_encode_one_int64_elem_scalar_ndarray(self):
    """Ensures that np arrays with shape 0 can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.int64_values.append(415)
    self.assertEqual(codec.encode(np.array(415, dtype=np.int64)), expected)

  def test_encode_one_int64_elem_ndarray(self):
    """Ensures that np int64 arrays can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.shape.dim.add().size = 1
    datum.values.int64_values.append(415)
    self.assertEqual(codec.encode(np.array([415])), expected)

  def test_encode_multiple_int64_elem_ndarray(self):
    """Ensures that np int64 multi-element arrays can be encoded."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.shape.dim.add().size = 2
    datum.values.int64_values.extend([123, 456])
    self.assertEqual(codec.encode(np.array([123, 456])), expected)

  def test_decode_one_int64_elem_ndarray(self):
    """Once encoded, the proto should be decodeable."""
    user_data = storage_pb2.Data()
    user_data.datum.shape.dim.add().size = 1
    user_data.datum.values.int64_values.append(9)
    np.testing.assert_equal(
        codec.decode(user_data), np.array([9], dtype=np.int64))

  def test_decode_multiple_int64_elem_ndarray(self):
    """Once encoded, the proto should be decodeable."""
    user_data = storage_pb2.Data()
    user_data.datum.shape.dim.add().size = 3
    user_data.datum.shape.dim.add().size = 2
    user_data.datum.values.int64_values.extend([6, 5, 4, 3, 2, 1])
    np.testing.assert_equal(
        codec.decode(user_data),
        np.array([[6, 5], [4, 3], [2, 1]], dtype=np.int64))

  def test_encode_one_uint32_elem_scalar_ndarray(self):
    """Ensures that np arrays with shape 0 can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.uint32_values.append(415)
    self.assertEqual(codec.encode(np.array(415, dtype=np.uint32)), expected)

  def test_encode_one_uint32_elem_ndarray(self):
    """Ensures that np uint32 arrays can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.shape.dim.add().size = 1
    datum.values.uint32_values.append(415)
    self.assertEqual(codec.encode(np.array([415], dtype=np.uint32)), expected)

  def test_decode_one_uint32_elem_ndarray(self):
    """Once encoded, the proto should be decodeable."""
    user_data = storage_pb2.Data()
    user_data.datum.shape.dim.add().size = 1
    user_data.datum.values.uint32_values.append(9)
    np.testing.assert_equal(
        codec.decode(user_data), np.array([9], dtype=np.uint32))

  def test_encode_one_uint64_elem_scalar_ndarray(self):
    """Ensures that np arrays with shape 0 can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.uint64_values.append(415)
    self.assertEqual(codec.encode(np.array(415, dtype=np.uint64)), expected)

  def test_encode_one_uint64_elem_ndarray(self):
    """Ensures that np uint64 arrays can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.shape.dim.add().size = 1
    datum.values.uint64_values.append(415)
    self.assertEqual(codec.encode(np.array([415], dtype=np.uint64)), expected)

  def test_decode_one_uint64_elem_ndarray(self):
    """Once encoded, the proto should be decodeable."""
    user_data = storage_pb2.Data()
    user_data.datum.shape.dim.add().size = 1
    user_data.datum.values.uint64_values.append(9)
    np.testing.assert_equal(
        codec.decode(user_data), np.array([9], dtype=np.uint64))

  def test_encode_one_bool_elem_scalar_ndarray(self):
    """Ensures that np arrays with shape 0 can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.bool_values.append(True)
    self.assertEqual(codec.encode(np.array(True, dtype=np.bool)), expected)

  def test_encode_one_bool_elem_ndarray(self):
    """Ensures that np bool arrays can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.shape.dim.add().size = 1
    datum.values.bool_values.append(True)
    self.assertEqual(codec.encode(np.array([True], dtype=np.bool)), expected)

  def test_decode_one_bool_elem_ndarray(self):
    """Once encoded, the proto should be decodeable."""
    user_data = storage_pb2.Data()
    user_data.datum.shape.dim.add().size = 1
    user_data.datum.values.bool_values.append(True)
    np.testing.assert_equal(
        codec.decode(user_data), np.array([True], dtype=np.bool))

  def test_encode_one_string_elem_scalar_ndarray(self):
    """Ensures that np arrays with shape 0 can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.string_values.append('dream theater')
    self.assertEqual(codec.encode(np.array('dream theater')), expected)

  def test_encode_one_string_elem_ndarray(self):
    """Ensures that np string arrays can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.shape.dim.add().size = 1
    datum.values.string_values.append('rachmaninov')
    self.assertEqual(codec.encode(np.array(['rachmaninov'])), expected)

  def test_decode_one_string_elem_ndarray(self):
    """Once encoded, the proto should be decodeable."""
    user_data = storage_pb2.Data()
    user_data.datum.shape.dim.add().size = 1
    user_data.datum.values.string_values.append('scriabin')
    np.testing.assert_equal(codec.decode(user_data), np.array(['scriabin']))

  def test_encode_one_bytes_elem_scalar_ndarray(self):
    """Ensures that np arrays with shape 0 can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.bytes_values.append(b'a1b2c3d4e5f6')
    self.assertEqual(codec.encode(np.array(b'a1b2c3d4e5f6')), expected)

  def test_encode_one_bytes_elem_ndarray(self):
    """Ensures that np bytes arrays can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.shape.dim.add().size = 1
    datum.values.bytes_values.append(b'a1b2c3d4e5f6')
    self.assertEqual(codec.encode(np.array([b'a1b2c3d4e5f6'])), expected)

  def test_decode_one_bytes_elem_ndarray(self):
    """Once encoded, the proto should be decodeable."""
    user_data = storage_pb2.Data()
    user_data.datum.shape.dim.add().size = 1
    user_data.datum.values.bytes_values.append(b'6f5e4d3c2b1a')
    np.testing.assert_equal(
        codec.decode(user_data), np.array([b'6f5e4d3c2b1a']))

  def test_encode_one_int_elem_scalar_ndarray(self):
    """Ensures that ndarrays with dtype==object raise an error."""
    self.assertRaises(TypeError, codec.encode,
                      np.array(12345678901234567890, dtype=object))

  def test_decode_one_int_elem_ndarray(self):
    """Ensures that non-scalar Datums with dtype==object raise an error."""
    user_data = storage_pb2.Data()
    user_data.datum.shape.dim.add().size = 1
    user_data.datum.values.bigint_values.append(
        b'\000\253T\251\214\353\037\n\322')
    self.assertRaises(TypeError, codec.decode, user_data)

  def test_encode_one_int8_elem_scalar_ndarray(self):
    """Ensures that np arrays with shape 0 can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.int8_values = b'\x85'
    self.assertEqual(codec.encode(np.array(-123, dtype=np.int8)), expected)

  def test_encode_one_int8_elem_ndarray(self):
    """Ensures that np int8 arrays can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.shape.dim.add().size = 1
    datum.values.int8_values = b'\x85'
    self.assertEqual(codec.encode(np.array([-123], dtype=np.int8)), expected)

  def test_encode_two_int8_elem_ndarray(self):
    """Ensures that np int8 2-element arrays can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.shape.dim.add().size = 2
    datum.values.int8_values = b'\x85\x84'
    self.assertEqual(
        codec.encode(np.array([-123, -124], dtype=np.int8)), expected)

  def test_decode_one_int8_elem_ndarray(self):
    """Once encoded, the proto should be decodeable."""
    user_data = storage_pb2.Data()
    user_data.datum.shape.dim.add().size = 1
    user_data.datum.values.int8_values = b'\x91'
    decoded = codec.decode(user_data)
    self.assertEqual(decoded.dtype, np.int8)
    np.testing.assert_equal(decoded, np.array([-111], dtype=np.int8))

  def test_decode_two_int8_elem_ndarray(self):
    """Once encoded, the proto should be decodeable."""
    user_data = storage_pb2.Data()
    user_data.datum.shape.dim.add().size = 2
    user_data.datum.values.int8_values = b'\xa1\xb2'
    np.testing.assert_equal(
        codec.decode(user_data), np.array([-95, -78], dtype=np.int8))

  def test_encode_one_int16_elem_scalar_ndarray(self):
    """Ensures that np arrays with shape 0 can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.int16_values = b'\xfe\xa7'
    self.assertEqual(codec.encode(np.array(-345, dtype=np.int16)), expected)

  def test_encode_one_int16_elem_ndarray(self):
    """Ensures that np int16 arrays can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.shape.dim.add().size = 1
    datum.values.int16_values = b'\xfe\xa7'
    self.assertEqual(codec.encode(np.array([-345], dtype=np.int16)), expected)

  def test_encode_two_int16_elem_ndarray(self):
    """Ensures that np int16 2-element arrays can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.shape.dim.add().size = 2
    datum.values.int16_values = b'\xfe\xa7\xfe\xa6'
    self.assertEqual(
        codec.encode(np.array([-345, -346], dtype=np.int16)), expected)

  def test_decode_one_int16_elem_ndarray(self):
    """Once encoded, the proto should be decodeable."""
    user_data = storage_pb2.Data()
    user_data.datum.shape.dim.add().size = 1
    user_data.datum.values.int16_values = b'\xfe\xa7'
    decoded = codec.decode(user_data)
    self.assertEqual(decoded.dtype, np.int16)
    np.testing.assert_equal(decoded, np.array([-345], dtype=np.int16))

  def test_decode_two_int16_elem_ndarray(self):
    """Once encoded, the proto should be decodeable."""
    user_data = storage_pb2.Data()
    user_data.datum.shape.dim.add().size = 2
    user_data.datum.values.int16_values = b'\xa1\xb2\xc3\xd4'
    np.testing.assert_equal(
        codec.decode(user_data), np.array([-24142, -15404], dtype=np.int16))

  def test_encode_one_uint8_elem_scalar_ndarray(self):
    """Ensures that np arrays with shape 0 can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.uint8_values = b'\x7b'
    self.assertEqual(codec.encode(np.array(123, dtype=np.uint8)), expected)

  def test_encode_one_uint8_elem_ndarray(self):
    """Ensures that np uint8 arrays can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.shape.dim.add().size = 1
    datum.values.uint8_values = b'\x7b'
    self.assertEqual(codec.encode(np.array([123], dtype=np.uint8)), expected)

  def test_encode_two_uint8_elem_ndarray(self):
    """Ensures that np uint8 arrays can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.shape.dim.add().size = 2
    datum.values.uint8_values = b'\x7b\x7a'
    self.assertEqual(
        codec.encode(np.array([123, 122], dtype=np.uint8)), expected)

  def test_decode_one_uint8_elem_ndarray(self):
    """Once encoded, the proto should be decodeable."""
    user_data = storage_pb2.Data()
    user_data.datum.shape.dim.add().size = 1
    user_data.datum.values.uint8_values = b'\xa1'
    np.testing.assert_equal(
        codec.decode(user_data), np.array([161], dtype=np.uint8))

  def test_decode_two_uint8_elem_ndarray(self):
    """Once encoded, the proto should be decodeable."""
    user_data = storage_pb2.Data()
    user_data.datum.shape.dim.add().size = 2
    user_data.datum.values.uint8_values = b'\xa1\xb2'
    np.testing.assert_equal(
        codec.decode(user_data), np.array([161, 178], dtype=np.uint8))

  def test_encode_one_uint16_elem_scalar_ndarray(self):
    """Ensures that np arrays with shape 0 can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.values.uint16_values = b'\x01Y'
    self.assertEqual(codec.encode(np.array(345, dtype=np.uint16)), expected)

  def test_encode_one_uint16_elem_ndarray(self):
    """Ensures that np uint16 arrays can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.shape.dim.add().size = 1
    datum.values.uint16_values = b'\x01Y'
    self.assertEqual(codec.encode(np.array([345], dtype=np.uint16)), expected)

  def test_encode_two_uint16_elem_ndarray(self):
    """Ensures that np uint16 2-element arrays can be encoded in our proto."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.shape.dim.add().size = 2
    datum.values.uint16_values = b'\x01Y\x01X'
    self.assertEqual(
        codec.encode(np.array([345, 344], dtype=np.uint16)), expected)

  def test_decode_one_uint16_elem_ndarray(self):
    """Once encoded, the proto should be decodeable."""
    user_data = storage_pb2.Data()
    user_data.datum.shape.dim.add().size = 1
    user_data.datum.values.uint16_values = b'\xa1\xb2'
    np.testing.assert_equal(
        codec.decode(user_data), np.array([41394], dtype=np.uint16))

  def test_decode_two_uint16_elem_ndarray(self):
    user_data = storage_pb2.Data()
    user_data.datum.shape.dim.add().size = 2
    user_data.datum.values.uint16_values = b'\xa1\xb2\xc3\xd4'
    np.testing.assert_equal(
        codec.decode(user_data), np.array([41394, 50132], dtype=np.uint16))

  # Multi-dimensional arrays.

  def test_encode_2d_int64_elem_ndarray(self):
    """A 2D np int64 array should also be reprentable."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.shape.dim.add().size = 2
    datum.shape.dim.add().size = 3
    datum.values.int64_values.extend([1, 3, 5, 7, 9, 11])
    self.assertEqual(codec.encode(np.array([[1, 3, 5], [7, 9, 11]])), expected)

  def test_encode_2d_double_elem_ndarray(self):
    """A 2D np float64 array should also be reprentable."""
    expected = storage_pb2.Data()
    datum = expected.datum
    datum.shape.dim.add().size = 3
    datum.shape.dim.add().size = 2
    datum.values.double_values.extend([10.0, 8.0, 6.0, 4.0, 2.0, 0.0])
    self.assertEqual(
        codec.encode(np.array([[10.0, 8.0], [6.0, 4.0], [2.0, 0.0]])), expected)

  ##############################################################################
  #
  # Array of np arrays tests
  #
  ##############################################################################

  # float64

  def test_encode_one_double_elem_ndarray_list(self):
    """A list of one np float64 array should be representable."""
    expected = storage_pb2.Data()
    datum = expected.array.values.add().datum
    datum.shape.dim.add().size = 1
    datum.values.double_values.append(3.14)
    self.assertEqual(codec.encode([np.array([3.14])]), expected)

  def test_encode_multiple_double_elem_ndarray_list(self):
    """A list of one multidimensional np int64 array should be representable."""
    expected = storage_pb2.Data()
    datum = expected.array.values.add().datum
    datum.shape.dim.add().size = 5
    datum.values.double_values.extend([0.0, 0.25, 0.5, 0.75, 1.0])
    self.assertEqual(
        codec.encode([np.array([0.0, 0.25, 0.5, 0.75, 1.0])]), expected)

  def test_decode_double_elem_ndarray_list(self):
    user_data = storage_pb2.Data()
    datum1 = user_data.array.values.add().datum
    datum1.shape.dim.add().size = 1
    datum1.values.double_values.append(1.2345)
    datum2 = user_data.array.values.add().datum
    datum2.shape.dim.add().size = 2
    datum2.values.double_values.extend([4.567, 8.9011])
    datum3 = user_data.array.values.add().datum
    datum3.shape.dim.add().size = 3
    datum3.shape.dim.add().size = 1
    datum3.values.double_values.extend([9.8765, 4.321, -0.12345])
    decoded = codec.decode(user_data)
    self.assertLen(decoded, 3)
    self.assertIsInstance(decoded, list)
    np.testing.assert_equal(decoded[0], np.array([1.2345], dtype=np.float64))
    np.testing.assert_equal(decoded[1],
                            np.array([4.567, 8.9011], dtype=np.float64))
    np.testing.assert_equal(
        decoded[2], np.array([[9.8765], [4.321], [-0.12345]], dtype=np.float64))

  # int64

  def test_encode_one_int64_elem_ndarray_list(self):
    """A list of one np int64 array should be representable."""
    expected = storage_pb2.Data()
    datum = expected.array.values.add().datum
    datum.shape.dim.add().size = 1
    datum.values.int64_values.append(719)
    self.assertEqual(codec.encode([np.array([719])]), expected)

  def test_encode_multiple_int64_elem_ndarray_list(self):
    """A list of one multidimensional np int64 array should be representable."""
    expected = storage_pb2.Data()
    datum = expected.array.values.add().datum
    datum.shape.dim.add().size = 5
    datum.values.int64_values.extend([1, 1, 2, 3, 5])
    self.assertEqual(codec.encode([np.array([1, 1, 2, 3, 5])]), expected)

  def test_decode_int64_elem_ndarray_list(self):
    user_data = storage_pb2.Data()
    datum1 = user_data.array.values.add().datum
    datum1.shape.dim.add().size = 1
    datum1.values.int64_values.append(1000)
    datum2 = user_data.array.values.add().datum
    datum2.shape.dim.add().size = 2
    datum2.values.int64_values.extend([2000, 3000])
    datum3 = user_data.array.values.add().datum
    datum3.shape.dim.add().size = 3
    datum3.shape.dim.add().size = 1
    datum3.values.int64_values.extend([4000, 5000, 6000])
    decoded = codec.decode(user_data)
    self.assertLen(decoded, 3)
    self.assertIsInstance(decoded, list)
    np.testing.assert_equal(decoded[0], np.array([1000], dtype=np.int64))
    np.testing.assert_equal(decoded[1], np.array([2000, 3000], dtype=np.int64))
    np.testing.assert_equal(decoded[2],
                            np.array([[4000], [5000], [6000]], dtype=np.int64))

  ##############################################################################
  #
  # Tuple tests
  #
  ##############################################################################

  def test_encode_one_double_elem_ndarray_tuple(self):
    """Tuples of np float64 arrays should be representable."""
    expected = storage_pb2.Data()
    datum = expected.tuple.values.add().datum
    datum.shape.dim.add().size = 1
    datum.values.double_values.append(-1 / 12)
    self.assertEqual(codec.encode((np.array([-1 / 12]),)), expected)

  def test_encode_multiple_double_elem_ndarray_tuple(self):
    """Tuples of np float64 arrays should be representable."""
    expected = storage_pb2.Data()
    datum = expected.tuple.values.add().datum
    datum.shape.dim.add().size = 2
    datum.values.double_values.extend([6.28, 2.71828])
    self.assertEqual(codec.encode((np.array([6.28, 2.71828]),)), expected)

  def test_decode_double_elem_ndarray_tuple(self):
    """Once encoded, the proto should be decodeable."""
    user_data = storage_pb2.Data()
    datum1 = user_data.tuple.values.add().datum
    datum1.shape.dim.add().size = 1
    datum1.values.double_values.append(1.2345)
    datum2 = user_data.tuple.values.add().datum
    datum2.shape.dim.add().size = 2
    datum2.values.double_values.extend([4.567, 8.9011])
    datum3 = user_data.tuple.values.add().datum
    datum3.shape.dim.add().size = 3
    datum3.shape.dim.add().size = 1
    datum3.values.double_values.extend([9.8765, 4.321, -0.12345])
    decoded = codec.decode(user_data)
    self.assertLen(decoded, 3)
    self.assertIsInstance(decoded, tuple)
    np.testing.assert_equal(decoded[0], np.array([1.2345], dtype=np.float64))
    np.testing.assert_equal(decoded[1],
                            np.array([4.567, 8.9011], dtype=np.float64))
    np.testing.assert_equal(
        decoded[2], np.array([[9.8765], [4.321], [-0.12345]], dtype=np.float64))

  def test_encode_one_int64_elem_ndarray_tuple(self):
    """Tuples of np int64 arrays should be representable."""
    expected = storage_pb2.Data()
    datum = expected.tuple.values.add().datum
    datum.shape.dim.add().size = 1
    datum.values.int64_values.append(1729)
    self.assertEqual(codec.encode((np.array([1729]),)), expected)

  def test_encode_multiple_int64_elem_ndarray_tuple(self):
    """Tuples of np int64 arrays should be representable."""
    expected = storage_pb2.Data()
    datum = expected.tuple.values.add().datum
    datum.shape.dim.add().size = 6
    datum.values.int64_values.extend([2, 3, 5, 7, 9, 11])
    self.assertEqual(codec.encode((np.array([2, 3, 5, 7, 9, 11]),)), expected)

  def test_decode_int64_elem_ndarray_tuple(self):
    user_data = storage_pb2.Data()
    datum1 = user_data.tuple.values.add().datum
    datum1.shape.dim.add().size = 1
    datum1.values.int64_values.append(1000)
    datum2 = user_data.tuple.values.add().datum
    datum2.shape.dim.add().size = 2
    datum2.values.int64_values.extend([2000, 3000])
    datum3 = user_data.tuple.values.add().datum
    datum3.shape.dim.add().size = 3
    datum3.shape.dim.add().size = 1
    datum3.values.int64_values.extend([4000, 5000, 6000])
    decoded = codec.decode(user_data)
    self.assertLen(decoded, 3)
    self.assertIsInstance(decoded, tuple)
    np.testing.assert_equal(decoded[0], np.array([1000], dtype=np.int64))
    np.testing.assert_equal(decoded[1], np.array([2000, 3000], dtype=np.int64))
    np.testing.assert_equal(decoded[2],
                            np.array([[4000], [5000], [6000]], dtype=np.int64))

  ##############################################################################
  #
  # Dict tests
  #
  ##############################################################################

  def test_encode_int64_elem_ndarray_dict(self):
    """Dict of int64 and of other dicts."""
    expected = storage_pb2.Data()
    d = expected.dict.values
    datum1 = d['good'].datum
    datum1.shape.dim.add().size = 1
    datum1.values.int64_values.append(1)
    datum2 = d['bad'].datum
    datum2.shape.dim.add().size = 1
    datum2.values.int64_values.append(-1)
    # Dict also supports nested dicts.
    datum3 = d['nested_dict'].dict.values['cumulants'].datum
    datum3.shape.dim.add().size = 2
    datum3.values.int64_values.extend([1000, -2])
    self.assertEqual(
        codec.encode({
            'good': np.array([1]),
            'bad': np.array([-1]),
            'nested_dict': {
                'cumulants': np.array([1000, -2])
            }
        }), expected)

  def test_encode_double_elem_ndarray_dict(self):
    """Dicts of np arrays."""
    expected = storage_pb2.Data()
    d = expected.dict.values
    datum1 = d['golden'].datum
    datum1.shape.dim.add().size = 1
    datum1.values.double_values.append(1.618)
    datum2 = d['sqrt2'].datum
    datum2.shape.dim.add().size = 1
    datum2.values.double_values.append(1.41421)
    self.assertEqual(
        codec.encode({
            'golden': np.array([1.618]),
            'sqrt2': np.array([1.41421])
        }), expected)

  def test_encode_mixed_elem_ndarray_dict(self):
    """Dicts of np arrays of different dtypes."""
    expected = storage_pb2.Data()
    d = expected.dict.values
    datum1 = d['mozart_death'].datum
    datum1.shape.dim.add().size = 1
    datum1.values.int64_values.append(35)
    datum2 = d['sqrt3'].datum
    datum2.shape.dim.add().size = 1
    datum2.values.double_values.append(1.73205)
    self.assertEqual(
        codec.encode({
            'mozart_death': np.array([35]),
            'sqrt3': np.array([1.73205])
        }), expected)

  def test_decode_dict(self):
    user_data = storage_pb2.Data()
    datum1 = user_data.dict.values['pi'].datum
    datum1.shape.dim.add().size = 1
    datum1.values.double_values.append(3.14159265)
    datum2 = user_data.dict.values['primes'].datum
    datum2.shape.dim.add().size = 5
    datum2.values.int64_values.extend([2, 3, 5, 7, 11])
    datum3 = user_data.dict.values['negative_squares_doubles'].datum
    datum3.shape.dim.add().size = 5
    datum3.shape.dim.add().size = 2
    datum3.values.int64_values.extend(
        [-1, -4, -9, -16, -25, -2, -8, -18, -32, -50])
    decoded = codec.decode(user_data)
    self.assertIsInstance(decoded, dict)
    self.assertIn('pi', decoded)
    np.testing.assert_equal(decoded['pi'],
                            np.array([3.14159265], dtype=np.float64))
    self.assertIn('primes', decoded)
    np.testing.assert_equal(decoded['primes'],
                            np.array([2, 3, 5, 7, 11], dtype=np.int64))
    self.assertIn('negative_squares_doubles', decoded)
    np.testing.assert_equal(
        decoded['negative_squares_doubles'],
        np.array([[-1, -4], [-9, -16], [-25, -2], [-8, -18], [-32, -50]],
                 dtype=np.int64))

  ##############################################################################
  #
  # Unsupported types tests
  #
  ##############################################################################

  @parameterized.named_parameters(
      ('modules_are_not_supported', np),
      ('classes_are_not_supported', set),
      ('functions_are_not_supported', map),
      ('type_classes_are_not_supported', type(int)),
      ('sets_are_not_supported', set()),
      ('complex_numbers_are_not_supported', complex(1, 2)),
  )
  def test_unsupported_types(self, arg):
    """Ensures that TypeError is raised when an unsupported type is encoded."""
    self.assertRaises(TypeError, codec.encode, arg)


if __name__ == '__main__':
  absltest.main()
