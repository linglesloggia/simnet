#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# qa_massive : tests PyWiCh channel models

'''Tests PyWiCh channel models.
'''

# import classes from the main library
from libsimnet.simulator import Setup, Simulation, mk_imports
from libsimnet.results import Statistics
from libsimnet.libutils import run_qa_tests

# import concrete classes overwritten from the main library abstract classes
from models.transpblock.simpletrblk.simple_trblk import TransportBlock 
from models.trafficgen.simpletrfgen.simple_trfgen import TrafficGenerator
from models.scheduler.simplesched.simple_sched import Scheduler
from file_channel_vgb import Channel
from file_chenv import ChannelEnvironment

# import functions for a simple simulation scenery
from extensions.simplesim.mk_simplescene import setup_sim, run_sim

# Python imports
import sys


def test_mass_vrfy():
    '''Setup and run a simulation test with many users, variable time.
    '''
    time_sim = 5                # simulation duration
    nr_usreqs = [3, 0, 0]     # user eqs qty per user group
    loss_prob = 0.0             # probability of losing packets
    gen_size = [40, 0, 0]       # packet size 
    gen_delay = [1, 1, 1]       # packet size and generation delay
    trans_delay = 1             # transmission steps
    ug_maxlen = [0, 0, 0]       # queue max length per user group
    ug_keep = [False, False, False]  # keep packets after transmission
    ug_pkts = [5, 0, 0]         # packets to generate on each round
    nr_res = [0, 0, 1]         # quantity of resources Poor, Fair, Good
    assgn_mat = [
        ["SL-1", "Good", 1]]   # Good is 11*8*1=88, other: 14*1*12=168
    dbg_setup = [True, True, True]  # setup debug
    dbg_run = True
    # change simulation parameters and run with help function in mk_simplescene
    run_sim(time_sim, nr_usreqs, loss_prob, 
        gen_size, gen_delay, ug_pkts,
        ug_maxlen, ug_keep,
        nr_res, assgn_mat,
        trans_delay, 
        dbg_setup, dbg_run)
    return


def test_mass_run():
    '''Test with many users and long time, run.
    '''
    time_sim = 103              # simulation duration
    nr_usreqs = [100, 0, 0]     # user eqs qty per user group
    loss_prob = 0.0             # probability of losing packets
    gen_size = [40, 0, 0]       # packet size 
    gen_delay = [1, 1, 1]       # packet size and generation delay
    trans_delay = 1             # transmission steps
    ug_maxlen = [0, 0, 0]       # queue max length per user group
    ug_keep = [False, False, False]  # keep packets after transmission
    ug_pkts = [3, 0, 0]         # packets to generate on each round
    nr_res = [0, 0, 50]         # quantity of resources Poor, Fair, Good
    assgn_mat = [
        ["SL-1", "Good", 50]]   # Good is 11*8*1=88, other: 14*1*12=168
    dbg_setup = [False, False, False]  # setup debug
    dbg_run = False
    # change simulation parameters and run with help function in mk_simplescene
    run_sim(time_sim, nr_usreqs, loss_prob, 
        gen_size, gen_delay, ug_pkts,
        ug_maxlen, ug_keep,
        nr_res, assgn_mat,
        trans_delay, 
        dbg_setup, dbg_run)
    return


# docstring for pydoctor out of function
ls_tests = []
'''List of tests to show in menu.'''

if __name__ == "__main__":

    mk_imports(nm_channel=Channel, nm_chanenv=ChannelEnvironment, \
        nm_trblk=TransportBlock, nm_trfgen=TrafficGenerator, \
        nm_scheduler=Scheduler)

    ls_tests = [ \
        ["Massive test, verify.", test_mass_vrfy], \
        ["Massive test, run.", test_mass_run], \
        ] 
    print("Choose option, 'q' to quit:")
    run_qa_tests(ls_tests)



