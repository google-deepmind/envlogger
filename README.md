# EnvironmentLogger

`EnvLogger` is a standard `dm_env.Environment` class wrapper that
records interactions between a real environment and an agent. These interactions
are saved on disk as trajectories and can be retrieved in whole, by individual
timesteps or by specific episodes.

![drawing](docs/images/envlogger.png "EnvironmentLogger Diagram")

## Metadata

To better categorize your logged data, you may want to add some tags in the
metadata when you construct the logger wrapper.  The metadata is written once
per `EnvLogger` instance.

```python
env = envlogger.EnvLogger(
    env,
    data_directory='/tmp/experiment_logs',
    metadata={
        'environment_type': 'dm_control',
        'agent_type': 'D4PG'
    })
```
## Examples

Most of the time, it is just a one-liner wrapper, e.g.

```python
import envlogger
from envlogger.testing import catch_env
import numpy as np

env = catch_env.Catch()
with envlogger.EnvLogger(
    env, data_directory='/tmp/experiment_logs') as env:

  env.reset()
  for step in range(100):
    action = np.random.randint(low=0, high=3)
    timestep = env.step(action)
```

## Reading stored trajectories

`Reader` can read stored trajectories. Example:

```python
from envlogger import reader

with reader.Reader(
    data_directory='/tmp/experiment_logs') as r:
  for episode in r.episodes:
    for step in episode:
       # step is a step_data.StepData.
       # Use step.timestep.observation, step.timestep.reward, step.action etc...
```

## Getting Started

> EnvLogger currently only supports Linux based OSes and Python 3.

You can install EnvLogger via `pip`:

```
pip install envlogger
```

##### Compiling from source

For this option you will need to [install Bazel](https://docs.bazel.build/versions/main/install.html) and [GMP](https://gmplib.org/) (`libgmp-dev` on Debian-based systems).
Please note that Bazel versions >4.0 are not supported. Our recommended version
is [3.7.2](https://github.com/bazelbuild/bazel/releases/tag/3.7.2). Then:

```
git clone https://github.com/deepmind/envlogger/
cd envlogger
bazel test --test_output=errors envlogger/...
```

##### Running inside Docker

We provide a Docker image that can be used to run tests in a more controlled
environment. You can run it as follows:

```
sh docker/build.sh
docker run -it envlogger bash
bazel test --test_output=errors envlogger/...
```

## Benchmarks

Wrapping your environment with EnvLogger adds an approximately 2 millisecond overhead per environment step.
See the following difference in distributions in the case of random agent on a 100 steps per second Catch environment (measured in milliseconds).

Percentiles      | 50th      | 95th       | 99th      | 99.9th     | mean (± std)
---------------- | --------- | -----------| ----------| -----------| -----------
w/o EnvLogger    | 10.15     | 10.23      | 11.51     | 14.70      | 10.19 (± 1.06)
w/ EnvLogger     | 12.65     | 14.39      | 15.94     | 19.43      | 12.88 (± 0.34)

<img src="docs/images/timings.png" width="40%">


## Acknowledgements

We greatly appreciate all the support from the
[TF-Agents](https://github.com/tensorflow/agents) team in setting up building
and testing for EnvLogger.
