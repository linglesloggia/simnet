#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# simnet: a very elementary simulator example
#

'''
Simulator with PyWich channel model.
'''

# Math imports
import csv
from random import random

# SimNet imports
from libsimnet.usernode import Channel
from file_chenv import ChannelEnvironment



class Channel(Channel):
    ''' Customized for channel state from file.
    '''

    def __init__(self, ch_env=None,interpol = False):
        '''Constructor.
        '''

        super().__init__()
        self.ch_env = ch_env    # pointer to a ChannelEnvironment object
        self.t_ant = 0
        self.index = 0
        self.ls_state = None
        self.interpol = interpol
        return


    def get_chan_state(self, t, MS_pos, MS_vel):
        '''Return a measure of the channel state.

        Channel state is determined by the file obtained from a channel simulator.
        @return: channel state, a number.
        '''

        if self.ls_state is not None:
            if not self.interpol:
                if t<= self.ls_state[self.index][0]:
                    chan_state = self.ls_state[self.index][1][0]
                else:
                    for j in range(self.index+1,len(self.ls_state)):
                        if t <= self.ls_state[j][0] :
                            self.index = j
                            break
                    chan_state = self.ls_state[self.index][1][0]
            else:
                if t<= self.ls_state[self.index][0]:
                    if self.index == 0: # extrapolation
                         y1 = self.ls_state[self.index][1][0]
                         t1 = self.ls_state[self.index][0]
                         y2 = self.ls_state[self.index+1][1][0]
                         t2 = self.ls_state[self.index+1][0]
                    else:
                         y1 = self.ls_state[self.index-1][1][0]
                         t1 = self.ls_state[self.index-1][0]
                         y2 = self.ls_state[self.index][1][0]
                         t2 = self.ls_state[self.index][0]
                else:
                    for j in range(self.index+1,len(self.ls_state)):
                        if t <= self.ls_state[j][0] :
                            self.index = j
                            break
                    y1 = self.ls_state[self.index-1][1][0]
                    t1 = self.ls_state[self.index-1][0]
                    y2 = self.ls_state[self.index][1][0]
                    t2 = self.ls_state[self.index][0]                
                chan_state = self.interpolate(t1,y1,t2,y2,t)
        else:
            print( "Warning : not file channel for this user return random channel")
            chan_state = random()
        return chan_state


    def interpolate(self, t1, y1, t2, y2, t):
        '''Interpolates time and position values.

        @return: interpolated value.
        '''

        return (y2-y1)/(t2-t1)*(t-t1)+y1
        

if __name__ == "__main__":
    ### main to test the file channel
    fp = open("snr_4ues_1000_5gUmi.csv", "r", encoding='utf8')
    reader = csv.reader(fp, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL, \
        lineterminator='\n')
    dc_fchan = [[]]
    line_count = 0
    for row in reader:
         ue_number = int(row[0][3:])
         if len(dc_fchan) < ue_number:
             dc_fchan.append([])
         state = [float(ele) for ele in row[2].split()]
         dc_fchan[ue_number-1].append((float(row[1]),state))
         line_count += 1
    print(f'Processed {line_count} lines.')
 
    ch_env_obj = ChannelEnvironment()
    chan_obj = Channel(ch_env_obj,True)
    ue_number = 0
    #print("assign ls state ", ue_number,dc_fchan[ue_number])
    chan_obj.ls_state = dc_fchan[ue_number]  # asigna la lista de estados al canal
    #print("channel ",chan_obj.ls_state)

    pos = [0,0,0]
    vel = [0,0,0]
    for time_t in range(1,3):
        t_chan = time_t * 0.00153
        chan_state = chan_obj.get_chan_state(t_chan, pos, vel)
        print("  time {:6.5f}, channel state: {:6.5f}".format(t_chan, chan_state) )


