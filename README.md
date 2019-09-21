# DRLND Project 3: Collaboration and Competition

<iframe width="560" height="315" src="https://www.youtube.com/embed/VhOlxvI-CNs" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

The goal of the agents is to play tennis; they have to bounce a ball between them over a net using 
rackets, without letting the ball hit the ground or fly out of bounds.

An agent receives information of the positions and velocities of the ball and the rackets, 
in total 8 variables corresponding to a 24-dimensional state space.
The agents control their movement back and forth, and can also jump up, which leads
to a 2-dimensional action space.

The agents receive a reward of +0.1 if it hits the ball over the net, but if the ball goes out of bounds or
drop to the floor, the agent instead receives a reward of -0.01. The task is considered solved when the average of
the maximum score of either agent per episode stays over +0.5 for 100 consecutive episodes. 

## Installation

Clone this repository and install the requirements needed as per the instructions below.

### Python Requirements

Follow the instructions in the Udacity [Deep Reinforcement Learning repository](https://github.com/udacity/deep-reinforcement-learning)
on how to set up the `drlnd` environment, and then also install the [Click](https://click.palletsprojects.com/en/7.x/)
package (used for handling command line arguments):
```shell
pip install click
```

Alternatively, on some systems it might be enough to install the required packages
from the provided `requirements.txt` file:
```shell
pip install -r requirements.txt
```

### Unity environment

Download the Unity environment appropriate for your operating system using the links below and unzip
it into the project folder.

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

## Training and Running the Agent

To train the agent, use the `train.py` program which takes the Unity environment,
and optionally locations of output files and/or a random seed.

```shell
(drlnd) $ python train.py --help
Usage: train.py [OPTIONS]

Options:
  --environment PATH     Path to Unity environment  [required]
  --plot-output PATH     Output file for score plot
  --scores-output PATH   Output file for scores
  --weights-output PATH  File to save weights to after success
  --seed INTEGER         Random seed
  --help                 Show this message and exit.
```

For example:

```shell
(drlnd) $ python train.py --environment=Tennis.app --seed=20190415 
```

After successfully training the agent, use the `run.py` program to load
weights and run the simulation, which takes similar parameters as
the training program:

```shell
(drlnd) $ python run.py --help
Usage: run.py [OPTIONS]

Options:
  --environment PATH    Path to Unity environment  [required]
  --n-episodes INTEGER  Number of episodes to run
  --weights-input PATH  Network weights
  --help                Show this message and exit.
```

For example:
```
(drlnd) $ python run.py --environment=Tennis.app --weights-input weights.pth
```