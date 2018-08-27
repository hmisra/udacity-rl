# Project 1: Navigation
## Introduction
This repository contains code to train an agent to successfully navigate (and collect yellow bananas) in a large, square world, using techniques learned in Parts 1 and 2 of Udacity's Deep Reinforcement Learning nanodegree [(Github repository)](https://github.com/udacity/deep-reinforcement-learning).

### Environment
A reward of `+1` is provided for collecting a yellow banana, and a negative reward of `-1` is provided for collecting a blue banana. The goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

![Example scenario](https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif)

#### State and action spaces
The state space has 37 dimensions, including the agent's current velocity and its perception of objects (ray-based). Given the state the agent is in, it can choose its next action from the following discrete action choices:
* `0` - Move forward
* `1` - Move backward
* `2` - Turn left
* `3` - Turn right

#### Solving the environment
In order to solve the environment, the trained agent must obtain an average score of `+13` over 100 consecutive episodes.

#### Download
The environment for Linux operating systems can be downloaded [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip). This environment requires the use of a display so if you would like to train the agent on Amazon Web Services (AWS), please follow the instructions under [Configuring your own instance](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md#configuring-your-own-instance) to set up X server as a virtual display.

## Training
### Set up for Linux
To train an agent using the code in this repository, you will need to follow the set up instructions below:
1. Create and activate a Python 3.6 environment using Anaconda (`env_name` can be replaced with any name):
``` bash
    $ conda create --name env_name python=3.6
    $ source activate env_name
```
2. Clone this repository and install the dependencies:
``` bash
    $ git clone https://github.com/howkhang/udacity-rl.git
    $ cd python
    $ pip install .
```
3. The structure of the code repository is as follows:
``` 
    p1_navigation/
        checkpoint.pth
        dqn_agent.py
        model.py
        navigation.ipynb
        unity-environment.log
        README.md
        Report.md
    __pycache__/
 ```
4. Place the environment file `Banana_Linux.zip` in the `p1_navigation/` folder and unzip (decompress) the file:
``` bash
    $ unzip Banana_Linux.zip
```
5. Launch `navigation.ipynb` in Jupyter and start training the agent by executing the code in the notebook. 
``` bash
    $ jupyter notebook
```
### Understanding the files
1. `dqn_agent.py` contains the `Agent` and `ReplayBuffer` classes for the agent to interact with the environment.
2. `model.py` contains the Pytorch neural network used to approximate the Q-value functions that the agent will be using.
3. `navigation.ipynb` is the code entry point for starting the environment and the training loop.
4. `unity-environment.log` is the log file that is created during the training loop.
5. `checkpoint.pth` contains the weights of the Pytorch model once the environment is successfully solved.
6. `Report.md` provides a description of the implementation (learning algorithm, hyperparameters etc).
