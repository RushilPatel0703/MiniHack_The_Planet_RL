import random
import numpy as np
import gym
import torch
from agent import *
from model import *
from wrappers import *
from matplotlib import pyplot as plt
from minihack import RewardManager
from nle import nethack
import cv2
import minihack

# for rendering video
cv2.ocl.setUseOpenCL(False)
class RenderRGB(gym.Wrapper):
    def __init__(self, env, key_name="pixel"):
        super().__init__(env)
        self.last_pixels = None
        self.viewer = None
        self.key_name = key_name

        render_modes = env.metadata['render.modes']
        render_modes.append("rgb_array")
        env.metadata['render.modes'] = render_modes

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.last_pixels = obs[self.key_name]
        return obs, reward, done, info

    def render(self, mode="human", **kwargs):
        img = self.last_pixels

        if mode != "human":
            return img
        else:
            from gym.envs.classic_control import rendering

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def reset(self):
        obs = self.env.reset()
        self.last_pixels = obs[self.key_name]
        return obs

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

# for plots
def subtask_graph(counts):
    c = ['#527143', '#AF000A', "#483d8b", '#964B00']
    x = ['Maze_Exit','Lava_Cross','Monster_Kill','Stairs_Reach']
    heights = [counts[0], counts[1], counts[2], counts[3]]
    plt.xlabel("Sub-Tasks Completion")
    plt.ylabel("Counts")
    plt.bar(x, heights, color=c)
    plt.savefig("sub_task_completion.png")
    plt.show()

def rewards_graph(episode_rewards):
    x = range(len(episode_rewards))
    plt.plot(x,episode_rewards)
    plt.title("Mean Rewards Per Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.savefig("training_reward_curves.png")
    plt.show()

if __name__ == '__main__':

    ACTIONS = (
        nethack.CompassDirection.N,
        nethack.CompassDirection.E,
        nethack.CompassDirection.S,
        nethack.CompassDirection.W,
        nethack.CompassDirectionLonger.N,
        nethack.CompassDirectionLonger.E,
        nethack.CompassDirectionLonger.S,
        nethack.CompassDirectionLonger.W,
        nethack.CompassDirection.NE,
        nethack.CompassDirection.SE,
        nethack.CompassDirection.SW,
        nethack.CompassDirection.NW,
        nethack.Command.PICKUP, 
        nethack.Command.INVENTORY, 
        nethack.Command.LOOK, 
        nethack.Command.OPEN,
        nethack.Command.ZAP,
        nethack.Command.FIRE,
        nethack.Command.APPLY,
        nethack.Command.PRAY,
        nethack.Command.PUTON,
        nethack.Command.QUAFF,
        nethack.Command.RUSH,
        nethack.Command.WEAR
    ) 
    
    reward_manager = RewardManager()
    reward_manager.add_location_event("door", reward=0.5)
    reward_manager.add_kill_event("demon", reward=0.7)
    reward_manager.add_kill_event("lava", reward=0.6)

    hyper_params = {
            "env": "MiniHack-Quest-Hard-v0",
            "replay-buffer-size": int(5e3),  # replay buffer size 5e3
            "learning-rate": 1e-4,  # learning rate for RMSprop optimizer
            "discount-factor": 0.99,  # discount factor
            "num-steps": int(1e6),  # total number of steps to run the environment for
            "batch-size": 128,  # number of transitions to optimize at the same time 128
            "learning-starts": 10000,  # number of steps before learning starts
            "learning-freq": 5,  # number of iterations between every optimization step
            "use-double-dqn": True,  # use double deep Q-learning
            "target-update-freq": 1000,  # number of iterations between every target network update
            "eps-start": 1.0,  # e-greedy start threshold
            "eps-end": 0.01,  # e-greedy end threshold
            "eps-fraction": 0.6,  # fraction of num-steps
            "print-freq": 10,
    }

    env = gym.make(hyper_params["env"],
                    observation_keys = ("glyphs", "chars", "colors", "pixel", "pixel_crop", "blstats", "colors_crop"),
                    penalty_time = -0.1,
                    reward_win=7,
                    seeds = [42],
                    reward_manager = reward_manager,
                    actions = ACTIONS)

    fname = 'model.h5'
    env = RenderRGB(env, "pixel")
    env = gym.wrappers.RecordVideo(env, "recordings", episode_trigger= lambda x: x % 1 == 0 )

    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = FrameStack(env, 4)
    state = np.array(env.reset())

    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])
    agent = DDQN(observation_space=env.observation_space,
                    action_space=env.action_space,
                    replay_buffer=replay_buffer,
                    use_double_dqn=hyper_params['use-double-dqn'],
                    lr=hyper_params['learning-rate'],
                    batch_size=hyper_params['batch-size'],
                    gamma=hyper_params['discount-factor']
    )

    eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])
    episode_rewards = [0.0]
    best_score = env.reward_range[0]
    lava_count = 0
    monster_count = 0
    stairs_count = 0
    maze_count = 0
    last_action = None
    prev_action = None

    state = env.reset()
    for t in range(hyper_params["num-steps"]):

        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * (hyper_params["eps-end"] - hyper_params["eps-start"])
        sample = random.random()

        if sample <= eps_threshold:
            action = np.random.choice(env.action_space.n)
        else:
            action = agent.act(state)

        next_state, reward, done, _ = env.step(action)
        if(reward == 0.5):
            maze_count += 1

        if(reward == 0.6):
            lava_count +=1

        if(reward == 0.7):
            monster_count += 1

        if(reward == 7):
            stairs_count += 1

        prev_action = last_action
        last_action = action
        ep_rew = []
        ep_loss = []

        replay_buffer.add(state, action, reward, next_state, float(done))
        ep_rew.append(reward)
        state = next_state

        episode_rewards[-1] += reward
        avg_score = np.mean(ep_rew[-100:])

        if done:
            state = env.reset()
            episode_rewards.append(0.0)

            last_action = None
            prev_action = None

        if (t > hyper_params["learning-starts"] and t % hyper_params["learning-freq"] == 0):
            loss = agent.optimise_td_loss()
            ep_loss.append(loss)

        if (t > hyper_params["learning-starts"] and t % hyper_params["target-update-freq"] == 0):
            agent.update_target_network()

        num_episodes = len(episode_rewards)

        if (avg_score > best_score):
            best_score = avg_score
            torch.save(agent, fname)

        if (done and hyper_params["print-freq"] is not None and len(episode_rewards) % hyper_params["print-freq"] == 0):
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(t))
            print("epsilon: {}".format(eps_threshold))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")
            print("Maze Passed:", maze_count)
            print("Lava Passed:",lava_count)
            print("Monster Killed:",monster_count)
            print("Stairs:",stairs_count)

            rewards_graph(episode_rewards)
            subtask_graph([maze_count, lava_count, monster_count, stairs_count])

    torch.save(agent, fname)
