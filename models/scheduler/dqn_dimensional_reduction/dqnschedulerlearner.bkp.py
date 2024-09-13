#!/usr/bin/env python3
#-*- coding: utf-8 -*-

'''A DQN-based scheduler; overwrites libsimnet.basenode.Scheduler.
'''

# import concrete classes overwritten from abstract classes
from libsimnet.basenode import Scheduler

# Python imports
import sys
import math
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Scheduler(Scheduler):
    '''A DQN-based resource scheduler.
    '''
    def __init__(self, ul_dl="DL", debug=False, action_size=10, batch_size=64, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, learning_rate=0.001, target_update=1000, memory_size=10000, save_every=100):
        '''Constructor.

        @param debug: if True prints messages
        '''
        super().__init__()
        self.debug = debug
        self.ul_dl = ul_dl
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.target_update = target_update
        self.memory = deque(maxlen=memory_size)
        self.save_every = save_every
        self.steps_done = 0

        # Placeholder for state_size, will be set during first call to assign_res
        self.state_size = None

    def assign_res(self, ls_usreqs, ls_res, time_t):
        '''Assigns resources based on DQN.

        @param ls_usreqs: a list of UserEquipment objects.
        @param ls_res: a list of Resource objects.
        @param time_t: simulation time.
        @return: a list of [user equipment, resource, ...] with assigned resources.
        '''
        self.ls_usreqs = ls_usreqs

        if self.state_size is None:
            self.state_size = len(ls_usreqs)
            self.model = QNetwork(self.state_size, self.action_size)
            self.target_model = QNetwork(self.state_size, self.action_size)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        state = np.array([sum(ue.pktque_dl.size_bits(pkt) for pkt in ue.pktque_dl.ls_recvd) for ue in ls_usreqs], dtype=np.float32)

        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.tensor(state).unsqueeze(0)
                action = self.model(state_tensor).argmax().item()
        else:
            action = random.randrange(len(ls_usreqs))

        ls_res_tmp = list(ls_res)
        ls_usr_res = [[ue] for ue in ls_usreqs]

        while ls_res_tmp:
            res = ls_res_tmp.pop(0)
            ue_idx = action % len(ls_usreqs)
            if res.res_type == ls_usreqs[ue_idx].usr_grp.dc_profile["res_type"]:
                ls_usr_res[ue_idx].append(res)
                #action //= len(ls_usreqs)

        # reward 
        next_state = np.array([sum(ue.pktque_dl.size_bits(pkt) for pkt in ue.pktque_dl.ls_recvd) for ue in ls_usreqs], dtype=np.float32)
        reward = -np.sum(next_state)

        #change sys.stdout to sys.stderr
        #print(f"Time {time_t}: Resource assignment complete")

        print(f"state: {state}, action: {action}, reward: {reward}, next_state: {next_state}", file=sys.stderr)

        # save
        self.memory.append((state, action, reward, next_state))

        # nn
        self.learn()

        # target network
        if self.steps_done % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        # Save model
        if self.steps_done % self.save_every == 0:
            torch.save({
            'state_size': self.state_size,
            'action_size': self.action_size,
            'model_state_dict': self.model.state_dict(),
        }, f"/content/drive/MyDrive/simnet/extensions/sim5gnr/gui/models/dqn_model_{self.steps_done}.pth")
            #torch.save(self.model.state_dict(), f"models/dqn_model_{self.steps_done}.pth")

        self.steps_done += 1

        # Debug output
        if self.debug:
            print(f"Time {time_t}: Resource assignment complete")
            print(f"State: {state}")
            print(f"Action: {action}")
            print(f"Reward: {reward}")
            print(f"Next state: {next_state}")

        return ls_usr_res

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch = zip(*batch)

        state_batch = torch.tensor(state_batch, dtype=torch.float32)
        action_batch = torch.tensor(action_batch, dtype=torch.int64)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32)

        q_values = self.model(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_model(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + self.gamma * next_q_values

        loss = nn.functional.mse_loss(q_values.squeeze(), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def __str__(self):
        '''For pretty printing, on overwritten class.'''
        msg = super().__str__()
        msg += ", overwritten for DQN example"
        return msg
