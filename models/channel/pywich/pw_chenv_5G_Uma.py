#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# simnet: a very elementary simulator example
#

'''
Simulator with PyWich channel model.
'''


import sys


# Math imports
from numpy import array,pi

# PyWich imports, adjust path to reach PyWich modules
sys.path.append('../../../../PyWiCh23/src')

import fading as fad
import antennas as antennas
import scenarios as sc
import frequency_band as fb
import channel_performance as cp

# PyWiSim imports
from libsimnet.usernode import  ChannelEnvironment



class ChannelEnvironment(ChannelEnvironment):
    '''Customized for PyWich channel simulator.
    '''

    def __init__(self):
        '''Constructor.
        '''
        print("--- 5G 3GPP Uma ChannelEnvironment created")
    
        ######## Build the receive and transmit antenna for testing 
        nMS=2
        nBS = 2
        aeBS = antennas.Antenna3gpp3D(8)
        self.aBS  = antennas.AntennaArray3gpp(0.5, 0.5, 1, nBS, 0, 0, 0, aeBS, 1)
        """ Base station antenna array""" 
        aeMS  = antennas.AntennaIsotropic(8)
        self.aMS  = antennas.AntennaArray3gpp(0.5, 0.5, 1, nMS, 0, 0, 0, aeMS, 1)
        """ Ms antenna array""" 
        ###########################################################
        
        ######## Build the scenario 
        self.fcGHz = 30
        """ Scenario frequency in GHz""" 
        posx_min = -300
        posx_max = 300
        posy_min = -300
        posy_max = 300
        grid_number = 30
        BS_pos = array([0,0,20])
        Ptx_db = 30
        self.LOS = 0
        """ LOS condition""" 
    
        self.scf=  sc.Scenario3GPPUma(self.fcGHz, posx_min,posx_max, posy_min, posy_max, grid_number, BS_pos, Ptx_db,True,self.LOS)
        """ Scenario for test 3gpp Indoor""" 
    
        #####################################################
        
        ########## Build the OFDM frequency band
        self.freq_band =  fb.FrequencyBand(fcGHz=self.fcGHz,number_prbs=81,bw_prb=10000000,noise_figure_db=5.0,thermal_noise_dbm_Hz=-174.0) 
        """ OFDM frequency band for test""" 
        self.freq_band.compute_tx_psd(tx_power_dbm=30)
        ###################################################
        
        ###### Build the steering vectors for beamforming according the the BS and MS positions
        self.wBS = self.aBS.compute_phase_steering (0,pi/2,0,0)
        """ BS steering vector for baemforming""" 
        self.wMS = self.aMS.compute_phase_steering (pi,pi/2,0,0)
        """ MS steering vector for baemforming"""
        #####################################################
        
        #### Build the channel performance object to get the results after the simulation
        self.performance  = cp.ChannelPerformance()
        """ Channel performance object"""
        #####################################################        
        self.fading = fad.Fading3gpp(self.scf)
    
    
        return





