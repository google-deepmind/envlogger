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

"""Converters to/from np arrays from/to the storage_pb2 proto storage format.

The encode() method transforms Python values into storage_pb2.Data() objects
that can be serialized to disk or network. Please refer to storage.proto for the
exact proto schema for details.

The decode() method transforms storage_pb2.Data() objects into Python values of
types from the following specific pre-defined list:

- 32-bit floats become np.float32.
- 64-bit floats become np.float64.
- 8-bit integers become np.int8.
- 16-bit integers become np.int16.
- 32-bit integers become np.int32.
- 64-bit integers become np.int64.
- 8-bit unsigned integers become np.uint8.
- 16-bit unsigned integers become np.uint16.
- 32-bit unsigned integers become np.uint32.
- 64-bit unsigned integers become np.uint64.
- arbitrarily long integers become int().
- boolean values become bool().
- string values become str().
- bytes values become bytes().

In particular, values that can be represented by different types in Python will
be cast to the above types. For example:

type(decode(encode(3.14159265))) == np.float64

which means that Python floats are implicitly cast to np.float64. This is true
even though type(np.float64(3.14159265)) != type(3.14159265).

We can also store multidimensional arrays (np.ndarray):

encode(np.array([[1, 2], [3, 4]], dtype=np.int8))

The shape is preserved and the dtype is cast to one of the types mentioned
above.

We can also compose values with lists, tuples and dictionaries. For example:

encode([1, 2, 3, 4])

or even:

encode([np.int8(1), np.int8(2), np.int8(3), np.int8(4)])

Note however that np.ndarrays are MUCH more efficiently stored because all
elements are packed within a single Datum instead of one Data with multiple
Datums, requiring multiple decoding steps. np.ndarrays also enforce that its
elements have the same type, which prevents bugs such as b/156304574. The bottom
line is: store your data as an np.ndarray if you can (e.g. tensors), and use
Python lists for everything else.

NOTE: encoding an empty list, dict or tuple stores nothing:

decode(encode([])) is None == True

Tuples:

encode((1, 2, 3, 4))

And dictionary of string to another value:

encode({'primes': np.array([2, 3, 5, 7, 11], dtype=np.int64)})
"""

import struct
from typing import Any, Dict, List, Optional, Tuple, Union

from envlogger.proto import storage_pb2

import numpy as np


# A type annotation that represents all the possible number types we support.
ScalarNumber = Union[float, int, np.float32, np.float64, np.int32, np.int64,
                     np.uint32, np.uint64]


# Dimension size reserved for scalars. Please see proto definition.
_SCALAR_DIM_SIZE = -438


# Converters for scalar int8, int16, uint8, uint16 and float32 values stored in
# big-endian format.
int8struct = struct.Struct('>b')
int16struct = struct.Struct('>h')
uint8struct = struct.Struct('>B')
uint16struct = struct.Struct('>H')
float32struct = struct.Struct('>f')


def _python_int_to_bytes(py_int: int) -> bytes:
  """Encodes a vanilla Python integer into bytes.

  The output is a signed, big-endian byte string with as many bytes as needed to
  encode it without losing precision.
  NOTE: Only signed integers are accepted.

  Args:
    py_int: The integer to be encoded.
  Returns:
    The integer represented as bytes.
  """
  # Calculate the number of bytes needed to encode this _signed_ integer.
  # For example, to encode int(127) we need 7 // 8 + 1 == 1 byte. This is the
  # same as what we need to encode int(-127). However, to encode int(128) or
  # int(-128) we actually need 2 bytes.
  num_bytes_needed = py_int.bit_length() // 8 + 1
  return py_int.to_bytes(num_bytes_needed, byteorder='big', signed=True)


def _set_datum_values_from_scalar(scalar: Union[ScalarNumber, bool, str, bytes],
                                  datum: storage_pb2.Datum) -> bool:
  """Populates `datum` using `scalar` in a best effort way.

  Notice that unrecognized scalar datum will be ignored.

  Args:
    scalar: The source of the data.
    datum: The destination of the copy.
  Returns:
    True if the population was successful, False otherwise.
  """
  values = datum.values
  shape = datum.shape

  if isinstance(scalar, str):
    values.string_values.append(scalar)
    shape.dim.add().size = _SCALAR_DIM_SIZE
    return True
  if isinstance(scalar, bytes):
    values.bytes_values.append(scalar)
    shape.dim.add().size = _SCALAR_DIM_SIZE
    return True

  try:
    fdtype = np.finfo(scalar).dtype
    if fdtype == np.float32:
      values.float_values.append(scalar)
      shape.dim.add().size = _SCALAR_DIM_SIZE
      return True
    if fdtype == np.float64:
      values.double_values.append(scalar)
      shape.dim.add().size = _SCALAR_DIM_SIZE
      return True
  except ValueError:
    pass

  try:
    # Vanilla Python ints.
    if isinstance(scalar, int) and not isinstance(scalar, bool):
      values.bigint_values.append(_python_int_to_bytes(scalar))
      shape.dim.add().size = _SCALAR_DIM_SIZE
      return True

    # Numpy ints.
    idtype = np.iinfo(scalar).dtype
    if idtype == np.int8:
      values.int8_values = int8struct.pack(scalar)
      shape.dim.add().size = _SCALAR_DIM_SIZE
      return True
    if idtype == np.int16:
      values.int16_values = int16struct.pack(scalar)
      shape.dim.add().size = _SCALAR_DIM_SIZE
      return True
    if idtype == np.int32:
      values.int32_values.append(scalar)
      shape.dim.add().size = _SCALAR_DIM_SIZE
      return True
    if idtype == np.int64:
      values.int64_values.append(scalar)
      shape.dim.add().size = _SCALAR_DIM_SIZE
      return True
    if idtype == np.uint8:
      values.uint8_values = uint8struct.pack(scalar)
      shape.dim.add().size = _SCALAR_DIM_SIZE
      return True
    if idtype == np.uint16:
      values.uint16_values = uint16struct.pack(scalar)
      shape.dim.add().size = _SCALAR_DIM_SIZE
      return True
    if idtype == np.uint32:
      values.uint32_values.append(scalar)
      shape.dim.add().size = _SCALAR_DIM_SIZE
      return True
    if idtype == np.uint64:
      values.uint64_values.append(scalar)
      shape.dim.add().size = _SCALAR_DIM_SIZE
      return True
  except ValueError:
    pass

  if isinstance(scalar, bool) or isinstance(scalar, np.bool_):
    values.bool_values.append(bool(scalar))
    shape.dim.add().size = _SCALAR_DIM_SIZE
    return True

  return False


def _set_datum_values_from_array(array: np.ndarray,
                                 values: storage_pb2.Datum.Values) -> None:
  """Populates `values` from entries in `array`.

  Args:
    array: The source of the data.
    values: The destination of the copy.
  """

  if array.dtype == np.float32:
    setattr(values, 'float_values_buffer', array.astype('>f').tobytes())
    return

  for vs, dtype, cast_type in [
      (values.double_values, np.float64, np.float64),
      (values.int32_values, np.int32, np.int32),
      (values.int64_values, np.int64, np.int64),
      (values.uint32_values, np.uint32, np.uint32),
      (values.uint64_values, np.uint64, np.uint64),
      (values.bool_values, np.bool_, bool),
      (values.string_values, np.unicode_, np.unicode_),
      (values.bytes_values, np.bytes_, np.bytes_),
  ]:
    if np.issubdtype(array.dtype, dtype):
      for x in array.flatten():
        vs.append(cast_type(x))
      return

  for key, dtype, cast_type in [('int8_values', np.int8, '>b'),
                                ('int16_values', np.int16, '>h'),
                                ('uint8_values', np.uint8, '>B'),
                                ('uint16_values', np.uint16, '>H')]:
    if np.issubdtype(array.dtype, dtype):
      setattr(values, key, array.astype(cast_type).tobytes())
      return

  raise TypeError(f'Unsupported `array.dtype`: {array.dtype}')


def encode(
    user_data: Union[np.ndarray, List[Any], Tuple[Any, ...], Dict[str, Any]]
) -> storage_pb2.Data:
  """Converts common Python data objects to storage_pb2.Data() proto.

  This function converts numpy arrays, lists of numpy arrays, tuples of numpy
  arrays, dicts of numpy arrays and their nested versions (e.g. lists of lists
  of numpy arrays) to a proto format that can be written to disk.

  NOTE: When converting numpy arrays of strings or bytes, ensure that its dtype
  is np.object to ensure that no wrong conversions will occur.

  Usage:
    # A bare numpy array.
    proto_data = encode(np.ones((3, 4), dtype=np.int64))
    # A list of numpy arrays.
    proto_data = encode([np.ones((5.5, 4.4), np.array([1.1, 2.2])],
                         dtype=np.float64))
    # Please see the unit test for examples of other data types.

  Args:
    user_data: The python data to convert to proto.

  Returns:
    A storage_pb2.Data properly filled.
  Raises:
    TypeError: This error is raised in two different situations:
      1. An unsupported type is passed. We support only a subset of all python
        types. Things like functions, classes and even sets() are not supported.
      2. A heterogeneous list is passed. For compatibility with other
        programming languages, our definition of a list is narrower than
        Python's and we enforce that all elements in the list have the exact
        same type. We do not support something like [123, 'hello', True].
  """
  output = storage_pb2.Data()
  if user_data is None:
    return output

  datum = output.datum

  if isinstance(user_data, list):
    type_x = None
    for index, x in enumerate(user_data):
      # Ensure that all elements have the same type.
      # This is intentionally verbose so that we can provide a useful message
      # when an exception is raised.
      if type_x is None:
        type_x = type(x)
      elif not isinstance(x, type_x):
        raise TypeError(
            'We assume list is homogeneous, i.e., data are of the same type.'
            f' Expecting value of type {type_x} (type of the first element).'
            f' Got {x} of type {type(x)}, index: {index},'
            f' Whole list: {user_data}')
      # Copy each element to the array.
      output.array.values.add().CopyFrom(encode(x))
    return output
  if isinstance(user_data, tuple):
    for x in user_data:
      output.tuple.values.add().CopyFrom(encode(x))
    return output
  if isinstance(user_data, dict):
    for k, v in user_data.items():
      output.dict.values[k].CopyFrom(encode(v))
    return output
  if isinstance(user_data, np.ndarray):
    pass  # The "base" ndarray case.
  else:  # Try to encode scalars.
    if _set_datum_values_from_scalar(user_data, datum):
      return output
    raise TypeError(f'Unsupported data type: {type(user_data)}')

  # Set shape.
  for dim in user_data.shape:
    if dim > 0:
      proto_dim = datum.shape.dim.add()
      proto_dim.size = dim

  # Copy values.
  _set_datum_values_from_array(user_data, datum.values)

  return output


def decode_datum(
    datum: storage_pb2.Datum
) -> Union[np.ndarray, ScalarNumber, bool, str, bytes]:
  """Creates a numpy array or scalar from a Datum protobuf.

  Args:
    datum: The source data.

  Returns:
    A Python object ready to be consumed.
  """
  # Adjust shape.
  shape = [dim.size for dim in datum.shape.dim]
  is_scalar = len(shape) == 1 and shape[0] == _SCALAR_DIM_SIZE

  array = None
  values = datum.values

  # Normal values.
  for vs, dtype in [(values.float_values, np.float32),
                    (values.double_values, np.float64),
                    (values.int32_values, np.int32),
                    (values.int64_values, np.int64),
                    (values.uint32_values, np.uint32),
                    (values.uint64_values, np.uint64),
                    (values.bool_values, np.bool)]:
    if vs:
      if is_scalar:
        return dtype(vs[0])
      array = np.array(vs, dtype=dtype)
      break

  # Values packed in bytes.
  for vs, converter, dtype, dtype_code in [
      (values.float_values_buffer, float32struct, np.float32, '>f'),
      (values.int8_values, int8struct, np.int8, '>b'),
      (values.int16_values, int16struct, np.int16, '>h'),
      (values.uint8_values, uint8struct, np.uint8, '>B'),
      (values.uint16_values, uint16struct, np.uint16, '>H'),
  ]:
    if vs:
      if is_scalar:
        return dtype(converter.unpack(vs)[0])
      array = np.frombuffer(vs, dtype=dtype_code).astype(dtype)

  if values.string_values:
    if is_scalar:
      return values.string_values[0]
    array = np.array([x for x in values.string_values], dtype=np.object)
  elif values.bytes_values:
    if is_scalar:
      return values.bytes_values[0]
    array = np.array([x for x in values.bytes_values], dtype=np.object)
  elif values.bigint_values:
    def from_bigint(int_bytes):
      return int.from_bytes(int_bytes, byteorder='big', signed=True)
    if is_scalar:
      return from_bigint(values.bigint_values[0])
    raise TypeError(
        f'Unsupported Datum of arbitrarily big ints: {values.bigint_values}')

  if array is None:
    return None

  return np.reshape(array, shape)


def decode(
    user_data: storage_pb2.Data
) -> Optional[Union[ScalarNumber, bool, str, bytes, np.ndarray, List[Any],
                    Tuple[Any, ...], Dict[str, Any]]]:
  """Converts from storage_pb2.Data to common Python data objects.

  This function converts a storage_pb2.Data protobuf to numpy arrays, lists of
  numpy arrays, tuples of numpy arrays, dicts of numpy arrays and their nested
  versions (e.g. lists of lists of numpy arrays).

  For usage examples, please see the unit tests for this function.

  NOTE: `string_values` and `bytes_values` will both use numpy's dtype ==
  np.object. This is to avoid wrong conversions and unintended narrowing.

  Args:
    user_data: The protobuf data to convert to Python data objects.

  Returns:
    A Python object of numpy arrays.
  """
  if user_data is None:
    return None

  # The type of the wrapped protocol buffer is different and consequently
  # with the existing versions of the dependencies, accessing the (map)
  # fields are triggering a check failure.
  if not isinstance(user_data, storage_pb2.Data):
    s = user_data.SerializeToString()
    user_data = storage_pb2.Data()
    user_data.ParseFromString(s)

  if user_data.HasField('datum'):
    return decode_datum(user_data.datum)
  if user_data.HasField('array'):
    return [decode(x) for x in user_data.array.values]
  if user_data.HasField('tuple'):
    return tuple((decode(x) for x in user_data.tuple.values))
  if user_data.HasField('dict'):
    return {k: decode(x) for k, x in user_data.dict.values.items()}
  return None
