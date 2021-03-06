# CS294-112 HW 1: Imitation Learning

Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**

Once Python **3.5** is installed, you can install the remaining dependencies using `pip install -r requirements.txt`.

**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible with recent Mac machines.
There is a request for OpenAI to support it that can be followed [here](https://github.com/openai/gym/issues/638).

**Note**: Students enrolled in the course will receive an email with their MuJoCo activation key. Please do **not** share this key.

The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v2.pkl
* HalfCheetah-v2.pkl
* Hopper-v2.pkl
* Humanoid-v2.pkl
* Reacher-v2.pkl
* Walker2d-v2.pkl

The name of the pickle file corresponds to the name of the gym environment.

------------------------------

## Behavior Cloning

### Performance
1. Humanoid-v2: Make a 3D two-legged robot walk.

    model structure: 
    ```
    Train_data: 20 roll-outs
    FC(input_dim*128*128*64*output_dim), 
    Adam Optimizer,
    mse error, 
    epoch=60, 
    learning_rate = 0.01,
    batch_size = 128
    ```
    model performance:
    ```
    mean return 355.33142605852265
    std of return 84.7170588145909
    ```
    expert performance:
    ```
    mean return 10416.404812957171
    std of return 46.27421077209787
    ```

## Dagger

### Performance
1. Humanoid-v2: Make a 3D two-legged robot walk.

    model structure: 
    ```
    Train_data: 20 roll-outs
    FC(input_dim*128*128*64*output_dim), 
    Adam Optimizer,
    mse error, 
    epoch=60, 
    learning_rate = 0.01,
    batch_size = 128
    ```
    model performance:
    ```
    mean return 274.8870598152557
    std of return 11.416691108570921
    ```
    expert performance:
    ```
    mean return 10416.404812957171
    std of return 46.27421077209787
    ```

## Issues
Dagger's average return is worse than Behavior Cloning, standard deviation is better than Behavor Cloning
Why Dagger performs not as good as expected, need more experiments

