# Reinforcement Learning with Minihack
Minihack comes with a large set of enviroments, ranging in difficultieis. The enviroment explored in this repo is ***"MiniHack-Quest-
Hard-v0 environment"***.

![image](https://user-images.githubusercontent.com/30756824/201344566-3064525e-8bcf-4bad-8630-926f6c493d70.png)

## Contributers
* Ashton Naidoo
* Clerence Mathonsi
* Rushil Patel

## Installation
Full installation guide for minihack can be found here: https://github.com/facebookresearch/minihack/blob/main/docs/getting-started/installation.md <br>

**Note:** Please refer to [NLE installation instructions](https://github.com/facebookresearch/nle#installation) when having NLE-related dependency issues on __MacOS__ and __Ubuntu 18.04__. NLE requires `cmake>=3.15` to be installed when building the package. <br>

## Agents
Two agents were implemented, Duelling DQN and REINFORCE.
* **DDQN** had performed poorly over several runs but did manage to explore much of the first phases of the quest which is the maze, but never really solved the maze.
* **REINFORCE** failed to perform well in part due to the complicated reward function which needed to be constructed.

### DDQN
To run the DDQN agent, the `train_ddqn.py` needs to be run which has dependancy on `agent.py`, `model.py`, `replay_buffer.py`, `wrappers.py` <br>

### REINFORCE
To run the REINFORCE agent run the jupyter notbook.
