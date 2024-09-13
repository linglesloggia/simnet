#!/usr/bin/env python3
#-*- coding: utf-8 -*-

'''A DQN-based scheduler that loads a pre-trained model; overwrites libsimnet.basenode.Scheduler.
'''

# import concrete classes overwritten from abstract classes
from libsimnet.basenode import Scheduler

# Python imports
import sys
import numpy as np
import torch
import torch.nn as nn

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
    '''A DQN-based resource scheduler that loads a pre-trained model.
    '''
    def __init__(self, model_path = '/home/lingles/ownCloud/Summer_school/simnet/extensions/sim5gnr/gui/models/dimensional_reduction/dqn_model1.pt', ul_dl="DL", debug=False):
        '''Constructor.

        @param model_path: path to the saved model.
        @param debug: if True prints messages.
        '''
        super().__init__()
        self.debug = debug
        self.ul_dl = ul_dl
        self.state_size = None
        self.action_size = None
        self.model_path = model_path
        self.model = None

    def load_model(self):
        checkpoint = torch.load(self.model_path)
        self.state_size = checkpoint['state_size']
        self.action_size = checkpoint['action_size']
        self.model = QNetwork(self.state_size, self.action_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def assign_res(self, ls_usreqs, ls_res, time_t):
        '''Assigns resources based on a loaded DQN model with group consideration.

        @param ls_usreqs: a list of UserEquipment objects.
        @param ls_res: a list of Resource objects.
        @param time_t: simulation time.
        @return: a list of [user equipment, resource, ...] with assigned resources.
        '''
        self.ls_usreqs = ls_usreqs

        # Define the groups
        group1 = [ue for ue in ls_usreqs if ue.id_object in ["UE-1", "UE-2"]]
        group2 = [ue for ue in ls_usreqs if ue.id_object in ["UE-3", "UE-4"]]

        # Load the model if not already loaded
        if self.model is None:
            self.load_model()

        # Get state
        state = np.array([sum(ue.pktque_dl.size_bits(pkt) for pkt in ue.pktque_dl.ls_recvd) for ue in ls_usreqs], dtype=np.float32)

        # Select action using the loaded model
        with torch.no_grad():
            state_tensor = torch.tensor(state).unsqueeze(0)
            action = self.model(state_tensor).argmax().item()

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

        # Debug output
        if self.debug:
            print(f"Time {time_t}: Resource assignment complete")
            print(f"State: {state}")
            print(f"Action: {action}")

        # Print in sys.stderr the state and action taken
        print(f"State: {state}, Action: {action}", file=sys.stderr)

        # Print Q function values
        with torch.no_grad():
            q_values = self.model(state_tensor)
            print(f"Q-values: {q_values}", file=sys.stderr)
        
        return ls_usr_res

    def __str__(self):
        '''For pretty printing, on overwritten class.'''
        msg = super().__str__()
        msg += ", overwritten for loading a pre-trained DQN model"
        return msg