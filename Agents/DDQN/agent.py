from gym import spaces
import numpy as np
import torch
import torch.nn as nn
from model import *
from replay_buffer import ReplayBuffer


device = "cuda" if torch.cuda.is_available() else "cpu"

class DDQN:
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        replay_buffer: ReplayBuffer,
        use_double_dqn,
        lr,
        batch_size,
        gamma,
    ):
        """
        Initialise the Duelling DDQN algorithm using the RMSprop optimiser

        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.replay_buffer = replay_buffer
        self.use_double_dqn = use_double_dqn
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma

        self.Q_net = DDQNNet(observation_space, action_space)
        self.target_net = DDQNNet(observation_space, action_space)

        self.optimizer = torch.optim.RMSprop(self.Q_net.parameters(), lr = self.lr)
        self.learn_count = 0 #used to know when to update target network

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        self.optimizer.zero_grad()

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype =torch.float).to(self.Q_net.device)
        states_ = torch.tensor(next_states, dtype =torch.float).to(self.Q_net.device)
        rewards = torch.tensor(rewards, dtype =torch.float).to(self.Q_net.device)
        dones = torch.tensor(dones, dtype =torch.bool).to(self.Q_net.device)

        indices = np.arange(self.batch_size)

        V_s, A_s = self.Q_net.predict(states)
        V_s_, A_s_ = self.target_net.predict(states_)

        V_s_eval, A_s_eval = self.Q_net.predict(states_)

        q_pred = torch.squeeze(torch.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True))))[indices, actions]
        q_next = torch.squeeze(torch.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True))))

        q_eval = torch.squeeze(torch.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True))))

        max_actions = torch.argmax(q_eval, dim=1)

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next[indices, max_actions]

        loss = nn.MSELoss()
        L = loss(q_target, q_pred).to(self.Q_net.device)
        L.backward()
        self.optimizer.step()

        return L

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        self.target_net.load_state_dict(self.Q_net.state_dict())

    def act(self, state: np.ndarray):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        state = torch.squeeze(torch.tensor(np.array([state]), dtype=torch.float).to(self.Q_net.device))
        state = state.unsqueeze(0)
        _, advantage = self.Q_net.predict(state)
        action = torch.argmax(advantage).item()
        return action
