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
    def __init__(self, ul_dl="DL", debug=False, model_path="/home/lingles/ownCloud/Summer_school/simnet/models/scheduler/ground_base_dqn/dqn_model.pth"):
        '''Constructor.

        @param debug: if True prints messages
        @param model_path: path to the pre-trained model file
        '''
        super().__init__()
        self.debug = debug
        self.ul_dl = ul_dl
        self.model_path = model_path

        # Placeholder for state_size and action_size, will be set during model loading
        self.state_size = None
        self.action_size = None

        # Load the pre-trained model
        self.load_model()

    def generate_actions(self, num_users, total_resources):
        actions = []
        for c in product(range(total_resources + 1), repeat=num_users):
            if sum(c) == total_resources:
                actions.append(c)
        return actions

    def flatten_state(self, state):
        return [item for sublist in state for item in sublist]

    def load_model(self):
        '''Load the pre-trained model from file.'''
        if self.model_path:
            checkpoint = torch.load(self.model_path)
            self.state_size = checkpoint['state_size']
            self.action_size = checkpoint['action_size']
            self.model = QNetwork(self.state_size, self.action_size)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # Set the model to evaluation mode
        else:
            raise ValueError("Model path must be provided")

    def assign_res(self, ls_usreqs, ls_res, time_t):
        '''Assigns resources based on the pre-trained DQN model.

        @param ls_usreqs: a list of UserEquipment objects.
        @param ls_res: a list of Resource objects.
        @param time_t: simulation time.
        @return: a list of [user equipment, resource, ...] with assigned resources.
        '''
        self.ls_usreqs = ls_usreqs
        num_users = len(ls_usreqs)
        total_resources = len(ls_res)
        #num_intervals = 20

        # Set state_size dynamically based on number of UEs
             #self.state_size = num_users * num_intervals
        self.actions = self.generate_actions(num_users, total_resources)

        try:
            # Get the current state (total bits in buffer of each user)
            state = np.array([sum(ue.pktque_dl.size_bits(pkt) for pkt in ue.pktque_dl.ls_recvd) for ue in ls_usreqs], dtype=np.float32)
            #state = self.flatten_state(state)

        except Exception as e:
            if self.debug:
                print(f"Error calculating state: {e}")
            return [[ue] for ue in ls_usreqs]  # Return empty resource allocation in case of error

        # Ensure the state size matches the expected input size of the model
        if len(state) != self.state_size:
            raise ValueError(f"State size mismatch: expected {self.state_size}, got {len(state)}")

        # Select action
        with torch.no_grad():
            state_tensor = torch.tensor(state).unsqueeze(0)
            action_idx = self.model(state_tensor).argmax().item()

        # Get the corresponding action (tuple) from the actions list
        selected_action = self.actions[action_idx]  # This is a tuple like (2, 0, 1)

        #print('State tensor:', state_tensor)
        #print('Action index:', action_idx)
        #print('Selected action:', selected_action)

        # Assign resources based on selected_action
        ls_res_tmp = list(ls_res)
        ls_usr_res = [[ue] for ue in ls_usreqs]

        # Assign resources to each user according to the selected action tuple
        for ue_idx, num_resources in enumerate(selected_action):
            for _ in range(num_resources):
                if ls_res_tmp:
                    res = ls_res_tmp.pop(0)
                    if res.res_type == ls_usreqs[ue_idx].usr_grp.dc_profile["res_type"]:
                        ls_usr_res[ue_idx].append(res)

        # Debug output
        if self.debug:
            print(f"Time {time_t}: Resource assignment complete")
            print(f"State: {state}")
            print(f"Selected action (allocation): {selected_action}")

        return ls_usr_res

    def __str__(self):
        '''For pretty printing, on overwritten class.'''
        msg = super().__str__()
        msg += ", overwritten for DQN example"
        return msg