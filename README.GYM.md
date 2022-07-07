
# TFImpala with OpenAI Gym envs.


## Conda setup

```shell
$ conda create -n tfimpala-py36 python=3.6
$ conda activate tfimpala-py36
$ pip install tensorflow-gpu==1.9.0 dm-sonnet==1.23 gym[atari] opencv-python
```


## Running TFImpala

Example command:

```shell
rm -rf /tmp/agent && python3 experiment.py --total_environment_frames=200000000 --batch_size 32 --num_actors 48 --level_name PongNoFrameskip-v4 --unroll_length 100
```

Weight checkpoints and Tensorboard data is saved into `/tmp/agent` by
default. The above command *deletes that directory*. A different
directory can be set using the `--logdir` flag.


## Configuring the environment

Look at `environments.py` to change the Gym environment used. Use the
`--num_actions` command line flag to change the number of actions.


## Sweeps

We added simple sweep script. Test it with

```shell
$ pip install coolname
$ python launch.py --dry
```

## Tensorboard

To see the results, run in a new terminal:

```shell
$ module unload cuda
$ module load cuda/9.0  # Yes, for TensorBoard. IDK why.
$ cd /tmp/agent
$ python -m tensorboard.main --logdir=. --port=8888
```

Then navigate to [http://localhost:8888](http://localhost:8888).


## Dynamic batching

Technically, running `make` creates a file called `batcher.so`. If
that file is present, TFImpala tries to use it for dynamic
batching. Depending on the C++ compiler used it might also just segfault.
