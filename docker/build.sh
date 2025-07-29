# Copyright 2025 DeepMind Technologies Limited..
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

#!/bin/bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/..

# Default to Python 3.11.
PY3_VERSION=${1:-11}
echo Python version 3.${PY3_VERSION}

# Default image label to "envlogger"
IMAGE_LABEL=${2:-envlogger}
echo Output docker image label: ${IMAGE_LABEL}

docker build -t ${IMAGE_LABEL} -f docker/Dockerfile . --build-arg PY3_VERSION=${PY3_VERSION}
