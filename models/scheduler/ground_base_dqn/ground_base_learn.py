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

from itertools import product

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
    def __init__(self, ul_dl="DL", debug=False, action_size=4, batch_size=64, gamma=0.1, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, learning_rate=0.001, target_update=10, memory_size=10000, save_every=100, reset_interval=1000, buffer_threshold=2000):
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

        self.buffer_threshold = buffer_threshold
        self.reset_interval = reset_interval

        # Placeholder for state_size, will be set during first call to assign_res
        self.state_size = None

        self.selected_action = 0
        self.action_idx = 0
        self.penalty = 0
        self.state_memory = None

    def reset_simulation(self):
        '''Reset the simulation state.'''
        # Implement the logic to reset the simulation state
        # This might include resetting the environment, buffers, etc.
        print("Resetting simulation...")
        # Reset buffers, states, etc.
        for ue in self.ls_usreqs:
            ue.pktque_dl.ls_recvd.clear()
        self.steps_done = 0

    def generate_actions(self, num_users, total_resources):
        actions = []
        for c in product(range(total_resources + 1), repeat=num_users):
            if sum(c) == total_resources:
                actions.append(c)
        return actions

    def flatten_state(self, state):
        return [item for sublist in state for item in sublist]

    def assign_res(self, ls_usreqs, ls_res, time_t):
        '''Assigns resources based on DQN.

        @param ls_usreqs: a list of UserEquipment objects.
        @param ls_res: a list of Resource objects.
        @param time_t: simulation time.
        @return: a list of [user equipment, resource, ...] with assigned resources.
        '''
        self.ls_usreqs = ls_usreqs
        num_users = len(ls_usreqs)
        total_resources = len(ls_res)
        num_intervals = 20

        # Set state_size dynamically based on number of UEs
        if self.state_size is None:
            self.state_size = num_users
            # Adjust the action size according to the number of valid actions
            self.actions = self.generate_actions(num_users, total_resources)
            self.action_size = len(self.actions)
            self.model = QNetwork(self.state_size, self.action_size)
            self.target_model = QNetwork(self.state_size, self.action_size)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if self.state_memory is None:
            self.state_memory = np.zeros(num_users)


        ## Assign resources to each user according to the selected action tuple
        #for ue_idx, num_resources in enumerate(selected_action):
        #    for _ in range(num_resources):
        #        if ls_res_tmp:
        #            res = ls_res_tmp.pop(0)
        #            if res.res_type == ls_usreqs[ue_idx].usr_grp.dc_profile["res_type"]:
        #                ls_usr_res[ue_idx].append(res)




        # Calculate reward (new buffer sizes)

        try:

            #next_state = np.array([sum(ue.pktque_dl.size_bits(pkt) for pkt in ue.pktque_dl.ls_recvd) for ue in ls_usreqs], dtype=np.float32)
            #next_state = self.flatten_state(next_state)
            reward = -np.sum(self.state_memory) - self.penalty # You can adjust this reward function
        except Exception as e:
            if self.debug:
                print(f"Error calculating next state or reward: {e}")
            reward = 0  # Set default reward in case of error
        
        try:
            # Get the current state (total bits in buffer of each user)
            state = np.array([sum(ue.pktque_dl.size_bits(pkt) for pkt in ue.pktque_dl.ls_recvd) for ue in ls_usreqs], dtype=np.float32)
            
        except Exception as e:
            if self.debug:
                print(f"Error calculating state: {e}")
            return [[ue] for ue in ls_usreqs]  # Return empty resource allocation in case of error
        

        # Store transition in memory
        self.memory.append((self.state_memory, self.action_idx, reward, state))

        self.debug = 1
        # Debug output
        if self.debug:
            print(f"Time {time_t}: Resource assignment complete")
            print(f"State: {self.state_memory}")
            print(f"Selected action (allocation): {self.selected_action}")
            print(f"Reward: {reward}")
            print(f"Next state: {state}")


        self.state_memory = state
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
            }, f"/home/lingles/ownCloud/Summer_school/simnet/models/scheduler/ground_base_dqn/dqn_model.pth")

        if self.steps_done >= self.reset_interval or any(state > self.buffer_threshold for state in state):
            self.reset_simulation()

        self.steps_done += 1

        # Select action
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.tensor(state).unsqueeze(0)
                self.action_idx = self.model(state_tensor).argmax().item()
        else:
            self.action_idx = random.randrange(self.action_size)

        # Get the corresponding action (tuple) from the actions list
        self.selected_action = self.actions[self.action_idx]  # This is a tuple like (2, 0, 1)

        self.penalty = 0

        # Assign resources based on selected_action
        ls_res_tmp = list(ls_res)
        ls_usr_res = [[ue] for ue in ls_usreqs]


        # Assign resources to each user according to the selected action tuple
        for ue_idx, num_resources in enumerate(self.selected_action):
            if num_resources > 0 and len(ls_usreqs[ue_idx].pktque_dl.ls_recvd) == 0:
                # Penalize if resources are assigned to a user with an empty buffer
                self.penalty += 1000  # Adjust the penalty value as needed
            for _ in range(num_resources):
                if ls_res_tmp:
                    res = ls_res_tmp.pop(0)
                    if res.res_type == ls_usreqs[ue_idx].usr_grp.dc_profile["res_type"]:
                        ls_usr_res[ue_idx].append(res)




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
