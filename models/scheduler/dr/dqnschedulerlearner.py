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

# Define the Q-network
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
    def __init__(self, ul_dl="DL", debug=False, action_size=4, batch_size=64, gamma=0.1, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, learning_rate=0.001, target_update=10, memory_size=10000, save_every=100):
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
        '''Assigns resources based on DQN with consideration for grouped UEs.

        @param ls_usreqs: a list of UserEquipment objects.
        @param ls_res: a list of Resource objects.
        @param time_t: simulation time.
        @return: a list of [user equipment, resource, ...] with assigned resources.
        '''
        # Define the groups
        group1 = [ue for ue in ls_usreqs if ue.id_object in ["UE-1", "UE-2"]]
        group2 = [ue for ue in ls_usreqs if ue.id_object in ["UE-3", "UE-4"]]

        # Set state_size dynamically based on number of UEs if not already set
        if self.state_size is None:
            self.state_size = 2  # 2 groups 
            self.model = QNetwork(self.state_size, self.action_size)
            self.target_model = QNetwork(self.state_size, self.action_size)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        try:
            # Get the current state (total bits in buffer of each user)
            state = np.array([sum(ue.pktque_dl.size_bits(pkt) for pkt in ue.pktque_dl.ls_recvd) for ue in ls_usreqs], dtype=np.float32)
            # The state must be the sum of the buffer sizes of each user within the group
            state = [sum(state[:2]), sum(state[2:])]
        except Exception as e:
            if self.debug:
                print(f"Error calculating state: {e}")
            return [[ue] for ue in ls_usreqs]  # Return empty resource allocation in case of error

        # Select action
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.tensor(state).unsqueeze(0)
                action = self.model(state_tensor).argmax().item()
        else:
            action = random.randrange(3)  # 3 possible actions: all to group 1, all to group 2, or split 1 each.

        # Apply action: assign resources to groups
        if action == 0:  # Both resources to Group 1
            group1_res = ls_res
            group2_res = []
        elif action == 1:  # Both resources to Group 2
            group1_res = []
            group2_res = ls_res
        else:  # Split resources
            group1_res = [ls_res[0]]
            group2_res = [ls_res[1]] if len(ls_res) > 1 else []

        # Distribute resources within each group using Round Robin
        ls_usr_res = [[ue] for ue in ls_usreqs]
        group1_rr_index, group2_rr_index = 0, 0

        for res in group1_res:
            ue = group1[group1_rr_index % len(group1)]
            ls_usr_res[ls_usreqs.index(ue)].append(res)
            group1_rr_index += 1

        for res in group2_res:
            ue = group2[group2_rr_index % len(group2)]
            ls_usr_res[ls_usreqs.index(ue)].append(res)
            group2_rr_index += 1

        # Calculate reward (new buffer sizes)
        try:
            next_state = np.array([sum(ue.pktque_dl.size_bits(pkt) for pkt in ue.pktque_dl.ls_recvd) for ue in ls_usreqs], dtype=np.float32)
            next_state = [sum(next_state[:2]), sum(next_state[2:])]
            reward = -np.sum(next_state)  # reward based on the sum of next states
        except Exception as e:
            if self.debug:
                print(f"Error calculating next state or reward: {e}")
            reward = 0  # Set default reward in case of error

        # Store transition in memory
        self.memory.append((state, action, reward, next_state))

        # Learn from experience
        self.learn()

        # Update target network
        if self.steps_done % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        # Save model every `save_every` steps
        if self.steps_done % self.save_every == 0:
            torch.save({
                'state_size': self.state_size,
                'action_size': self.action_size,
                'model_state_dict': self.model.state_dict(),
            }, f"/home/lingles/ownCloud/Summer_school/simnet/extensions/sim5gnr/gui/models/dimensional_reduction/dqn_model1.pt")

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
