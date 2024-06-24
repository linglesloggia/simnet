#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# TransportBlock: overwrites for a 5G transport block model
#

'''S5G transport block model.
'''



# import abstract classes to overwrite
from libsimnet.usernode import TransportBlock,UserEquipment
import math,sys
import numpy as np
from extensions.sim5gnr.tables import loadModTable,loadSINR_MCStable,loadtbsTable,load_bands




class TransportBlock(TransportBlock):
    '''Defines the number of data bits to include in a transport block.
    '''

    def __init__(self,band ="n257",robustMCS=False,ul_dl="DL",mimo_mod ="SU",nlayers=1):
        '''Constructor.
        '''
        super().__init__()
        self.modTable = loadModTable()
        self.sinrModTable = loadSINR_MCStable(True)  # 5G
        self.tbsTable = loadtbsTable()
        self.bands = load_bands()
        self.band = band
        try:
            self.tdd = self.bands[band][2]    # If the system is using TDD (True) or FDD(False)
            self.fr = self.bands[band][0]    # frequency range designation 1 (410 MHz – 7125 MHz) or 2(>24 Ghz)  
        except KeyError:
            print(" Bands ", self.bands)
            print("Band ", band," not in table. Please enter the band in the bands table.")
            sys.exit(0)
        self.robustMCS = robustMCS
        self.mimo_mod = mimo_mod
        self.nlayers = nlayers
        self.ul_dl = ul_dl
        return


    def get_tb_size(self, nr_syms, chan_state,nr_res=1, ul_dl="DL"):
        '''Determines number of bits to include in a transport block.

        Determines TB size according to channel state,  and the modulation and coding scheme 5g Nr table.
        
        @param nr_syms: number of symbols a resource allows.
        @param chan_state: channel state.
        @param nres: quantity of resources of the transport block.
        
        @return: transport block size in bits.
        '''
    
        sinr = chan_state
        mcs_ = self.findMCS(sinr)
        if self.robustMCS and mcs_ > 2:
            mcs_ = mcs_ - 2
        mo = self.modTable[mcs_]["mod"]
        mcsi = self.modTable[mcs_]["mcsi"]
        Qm = self.modTable[mcs_]["bitsPerSymb"]
        R = self.modTable[mcs_]["codeRate"]
        if self.fr == 1:
            fr_name = "FR1"
        else:
            fr_name = "FR2"
        if nr_res > 0:
            tbls = self.setTBS(R, Qm,ul_dl, fr_name, nr_res,self.mimo_mod,self.nlayers,nr_syms)  # bits
        else:
            tbls = 0  
        return tbls

    def findMCS(self, s):
        mcs = -1
        findSINR = False
        while mcs < 27 and not (findSINR):  # By SINR
            mcs = mcs + 1
            if mcs < 27:
                findSINR = s < float(self.sinrModTable[mcs])
            else:
                findSINR = True
        return mcs

    def setTBS(self, r, qm, ul_dl, fr, nprb,mimo_mod,nlayers,nr_syms):  
        """       
        Calculates the transport block size (TBS) for a given amount of information to the number of resources and target code rate and modulation.
        See 3GPP TS 38.214 version 16.2.0 Release 16.

        @param r: code rate.
        @param qm: modulation.
        @param ul_dl: if the TB is for downlink or uplink.
        @param fr: The type of frequency band. 1 (410 MHz – 7125 MHz) or 2(>24 Ghz)  
        @param nprb: quantity of resources of the transport block.
        @param mimo_mod: MIMO multiuser or single user.
        @param nlayers: number of layers in MIMO system.
        @param nr_syms: number of symbols a resource allows.
        @param nprb: quantity of resources of the transport block.    
        @return: transport block size in bits.
        
        """
        OHtable = {"DL": {"FR1": 0.14, "FR2": 0.18}, "UL": {"FR1": 0.08, "FR2": 0.10}}
        OH = OHtable[ul_dl][fr]
        Nre__ = min(156, math.floor(nr_syms * (1 - OH)))
        if mimo_mod == "SU":
            Ninfo = Nre__ * nprb * r * qm * nlayers
            # tbs = Ninfo
            tbs = self.getTbs(Ninfo,r)
        else:
            Ninfo = Nre__ * nprb * r * qm
            
            tbs = self.getTbs(Ninfo,r)
        return tbs


    def getTbs(self,Ninfo: int, r: float) -> int:
        """
        Calculates the transport block size (TBS) for a given amount of information to be transmitted and target code rate.
        If Ninfo is less than 24, return 0. If Ninfo is between 24 and 3824, find the largest TBS value in the TBS table
        that is less than Ninfo and return it. If Ninfo is greater than 3824, calculate TBS value based on the given formula.
        If r is less than or equal to 1/4, calculate TBS value based on the given formula. If r is greater than 1/4,
        calculate TBS value based on the given formula. Return the calculated TBS value. See 3GPP TS 38.214 version 16.2.0 Release 16.
    
        :param Ninfo: An integer representing the amount of information to be transmitted.
        :param r: A float representing the target code rate.
        :return: An integer representing the calculated TBS value.
    
        """
            
        # If the amount of information to be transmitted is less than 24, return 0
        if Ninfo < 24:
            return 0
    
        # If the amount of information to be transmitted is between 24 and 3824, find the largest TBS (Transport Block Size) value in the TBS table
        # that is less than Ninfo and return it
        if Ninfo <= 3824:
            for i in range(len(self.tbsTable)):
                if Ninfo < self.tbsTable[i]:
                    return self.tbsTable[i-1]
    
        # If the amount of information to be transmitted is greater than 3824, calculate TBS value based on the given formula
        else:
            # Calculate n based on the formula
            n = np.floor(np.log2(Ninfo-24))-5
            
            # Calculate Ninfo_ based on the formula
            Ninfo_ = max(3840, pow(2,n) * np.round((Ninfo - 24)/(pow(2,n))) )
            
            # If r is less than or equal to 1/4, calculate TBS value based on the given formula
            if r <= 1/4:
                C = np.ceil((Ninfo_/+24)/3816)
                TBS = 8*C*np.ceil((Ninfo_+24)/(8*C)) - 24
                
            # If r is greater than 1/4, calculate TBS value based on the given formula
            else:
                # If Ninfo_ is greater than 8424, calculate C based on the given formula
                if Ninfo_ > 8424:
                    C = np.ceil((Ninfo_+24)/8424)
                    TBS = 8*C*np.ceil((Ninfo_+24)/(8*C)) - 24
                # If Ninfo_ is less than or equal to 8424, calculate TBS value based on the given formula
                else:
                    TBS = 8*np.ceil((Ninfo_+24)/8) - 24
    
        # Return the calculated TBS value
        return TBS
    
    
    ### esto hay que ver como usarlo
    def update_robustMCS_QoS_based(self,reqAvailability):
        if reqAvailability == "High":
            self.robustMCS = True


    def __str__(self):
        '''For pretty printing.
        '''

        msg = "Transport block 5G  "
        msg += "{:s}, band {:s}, fr {:d}, tdd {:d}".\
            format(self.id_object, self.band,self.fr, self.tdd )
        return msg

