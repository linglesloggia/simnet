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
    def __init__(self, model_path = '/content/drive/MyDrive/simnet/extensions/sim5gnr/gui/models/dqn_model_sc.pth', ul_dl="DL", debug=False):
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
        '''Assigns resources based on a loaded DQN model.

        @param ls_usreqs: a list of UserEquipment objects.
        @param ls_res: a list of Resource objects.
        @param time_t: simulation time.
        @return: a list of [user equipment, resource, ...] with assigned resources.
        '''
        self.ls_usreqs = ls_usreqs

        # Load the model if not already loaded
        if self.model is None:
            self.load_model()

        # Get state 
        state = np.array([sum(ue.pktque_dl.size_bits(pkt) for pkt in ue.pktque_dl.ls_recvd) for ue in ls_usreqs], dtype=np.float32)

        # Select action using the loaded model
        with torch.no_grad():
            state_tensor = torch.tensor(state).unsqueeze(0)
            action = self.model(state_tensor).argmax().item()

        # Apply action 
        ls_res_tmp = list(ls_res)
        ls_usr_res = [[ue] for ue in ls_usreqs]

        while ls_res_tmp:
            res = ls_res_tmp.pop(0)
            ue_idx = action % len(ls_usreqs)
            if res.res_type == ls_usreqs[ue_idx].usr_grp.dc_profile["res_type"]:
                ls_usr_res[ue_idx].append(res)
                #action //= len(ls_usreqs)

        # Debug output
        if self.debug:
            print(f"Time {time_t}: Resource assignment complete")
            print(f"State: {state}")
            print(f"Action: {action}")

        # print in sys.stderr the state and action taken
        print(f"State: {state}, Action: {action}", file=sys.stderr)

        # print q function values
        with torch.no_grad():
            q_values = self.model(state_tensor)
            print(f"Q-values: {q_values}", file=sys.stderr)
        return ls_usr_res

    def __str__(self):
        '''For pretty printing, on overwritten class.'''
        msg = super().__str__()
        msg += ", overwritten for loading a pre-trained DQN model"
        return msg
