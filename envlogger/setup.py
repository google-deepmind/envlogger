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

"""Install script for setuptools."""

import os
import posixpath
import shutil

import pkg_resources
import setuptools
from setuptools.command import build_ext
from setuptools.command import build_py

PROJECT_NAME = 'envlogger'

__version__ = '1.2'

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

_ENVLOGGER_PROTOS = (
    'proto/storage.proto',
)


class _GenerateProtoFiles(setuptools.Command):
  """Command to generate protobuf bindings for EnvLogger protos."""

  descriptions = 'Generates Python protobuf bindings for EnvLogger protos.'
  user_options = []

  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

  def run(self):
    # We have to import grpc_tools here, after setuptools has installed
    # setup_requires dependencies.
    from grpc_tools import protoc

    grpc_protos_include = pkg_resources.resource_filename(
        'grpc_tools', '_proto')

    for proto_path in _ENVLOGGER_PROTOS:
      proto_args = [
          'grpc_tools.protoc',
          '--proto_path={}'.format(grpc_protos_include),
          '--proto_path={}'.format(_ROOT_DIR),
          '--python_out={}'.format(_ROOT_DIR),
          '--grpc_python_out={}'.format(_ROOT_DIR),
          os.path.join(_ROOT_DIR, proto_path),
      ]
      if protoc.main(proto_args) != 0:
        raise RuntimeError('ERROR: {}'.format(proto_args))


class _BuildPy(build_py.build_py):
  """Generate protobuf bindings in build_py stage."""

  def run(self):
    self.run_command('generate_protos')
    build_py.build_py.run(self)


class BazelExtension(setuptools.Extension):
  """A C/C++ extension that is defined as a Bazel BUILD target."""

  def __init__(self, bazel_target):
    self.bazel_target = bazel_target
    self.relpath, self.target_name = (
        posixpath.relpath(bazel_target, '//').split(':'))
    ext_name = os.path.join(
        self.relpath.replace(posixpath.sep, os.path.sep), self.target_name)
    super().__init__(ext_name, sources=[])


class _BuildExt(build_ext.build_ext):
  """A command that runs Bazel to build a C/C++ extension."""

  def run(self):
    self.run_command('generate_protos')
    self.bazel_build()
    build_ext.build_ext.run(self)

  def bazel_build(self):

    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)

    bazel_argv = [
        'bazel',
        'build',
        '...',
        '--symlink_prefix=' + os.path.join(self.build_temp, 'bazel-'),
        '--compilation_mode=' + ('dbg' if self.debug else 'opt'),
        '--verbose_failures',
    ]

    self.spawn(bazel_argv)

    for ext in self.extensions:
      ext_bazel_bin_path = os.path.join(
          self.build_temp, 'bazel-bin',
          ext.relpath, ext.target_name + '.so')

      ext_name = ext.name
      ext_dest_path = self.get_ext_fullpath(ext_name)
      ext_dest_dir = os.path.dirname(ext_dest_path)
      if not os.path.exists(ext_dest_dir):
        os.makedirs(ext_dest_dir)
      shutil.copyfile(ext_bazel_bin_path, ext_dest_path)

      # Copy things from /external to their own libs
      # E.g. /external/some_repo/some_lib --> /some_lib
      if ext_name.startswith('external/'):
        split_path = ext_name.split('/')
        ext_name = '/'.join(split_path[2:])
        ext_dest_path = self.get_ext_fullpath(ext_name)
        ext_dest_dir = os.path.dirname(ext_dest_path)
        if not os.path.exists(ext_dest_dir):
          os.makedirs(ext_dest_dir)
        shutil.copyfile(ext_bazel_bin_path, ext_dest_path)


setuptools.setup(
    name=PROJECT_NAME,
    version=__version__,
    description='EnvLogger: A tool for recording trajectories.',
    author='DeepMind',
    license='Apache 2.0',
    ext_modules=[
        BazelExtension('//envlogger/backends/python:episode_info'),
        BazelExtension('//envlogger/backends/python:riegeli_dataset_reader'),
        BazelExtension('//envlogger/backends/python:riegeli_dataset_writer'),
    ],
    cmdclass={
        'build_ext': _BuildExt,
        'build_py': _BuildPy,
        'generate_protos': _GenerateProtoFiles,
    },
    packages=setuptools.find_packages(),
    setup_requires=[
        # Some software packages have problems with older versions already
        # installed by pip. In particular DeepMind Acme uses grpcio-tools 1.45.0
        # (as of 2022-04-20) so we use the same version here.
        'grpcio-tools>=1.45.0',
    ],
    install_requires=[
        'absl-py',
        'dm_env',
        'numpy',
        'protobuf>=3.14',
        'setuptools!=50.0.0',  # https://github.com/pypa/setuptools/issues/2350
    ],
    extras_require={
        'tfds': [
            'tensorflow',
            'tfds-nightly',
        ],
    })
