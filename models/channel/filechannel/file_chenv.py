#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# simnet: a very elementary simulator example
#

'''
Simulator with PyWich channel model.
'''

# PyWiSim imports
from libsimnet.usernode import  ChannelEnvironment



class ChannelEnvironment(ChannelEnvironment):
    '''Customized for channel state from file.
    '''

    def __init__(self):
        '''Constructor.
        '''
        print("--- File ChannelEnvironment created")
        


if __name__ == "__main__":
    ch_env_obj = ChannelEnvironment()

