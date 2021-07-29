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

env = catch_env.Catch()
with envlogger.EnvLogger(
    env, root_directory='/tmp/experiment_logs') as env:
  for step in range(num_steps):
    action = np.random.randint(low=0, high=3)
    timestep = env.step(action)
```

## Reading stored trajectories

`Reader` can read stored trajectories. Example:

```python
from envlogger import reader

with reader.Reader(directory) as reader:
  for episode in reader.episodes:
    for step in episode:
       # step is a step_data.StepData.
       # Use step.timestep.observation, step.timestep.reward, step.action etc...
```

## Getting Started

> EnvLogger currently only supports Linux based OSes and Python 3.

##### Via Docker

You'll need [Docker](https://docs.docker.com/get-docker/) set up on your machine and then:

```
git clone https://github.com/deepmind/envlogger/
cd envlogger
sh docker/build.sh  # may require sudo
docker run -it envlogger bash  # may require sudo
bazel test --test_output=errors envlogger/...
```

##### Via Bazel

For this option you will need to [install Bazel](https://docs.bazel.build/versions/main/install.html).
Please note that Bazel versions >4.0 are not supported. Our recommended version
is [3.7.2](https://github.com/bazelbuild/bazel/releases/tag/3.7.2). Then:

```
git clone https://github.com/deepmind/envlogger/
cd envlogger
bazel test --test_output=errors envlogger/...
```

## Acknowledgements

We greatly appreciate all the support from the
[TF-Agents](https://github.com/tensorflow/agents) team in setting up building
and testing for EnvLogger.
