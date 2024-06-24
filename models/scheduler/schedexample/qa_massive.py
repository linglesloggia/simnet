#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# qa_massive : tests a simple scheduler example

'''Tests a simple scheduler example.
'''

# import classes from the main library
from libsimnet.simulator import Setup, Simulation, mk_imports
from libsimnet.results import Statistics
from libsimnet.libutils import run_qa_tests

# import concrete classes overwritten from the main library abstract classes
from models.channel.randfixchan.randfix_chan import Channel, ChannelEnvironment
from models.transpblock.simpletrblk.simple_trblk import TransportBlock 
from models.trafficgen.simpletrfgen.simple_trfgen import TrafficGenerator
from sched_example import Scheduler  # scheduler under test
from sched_example import Slice      # slice rewrite for data collection

# import functions for a simple simulation scenery
from extensions.simplesim.mk_simplescene import setup_sim, run_sim

# Python imports
import sys


def test_mass_vrfy():
    '''Setup and run a simulation test with many users, variable time.
    '''
    time_sim = 5                # simulation duration
    nr_usreqs = [2, 0, 0]     # user eqs qty per user group
    loss_prob = 0.0             # probability of losing packets
    gen_size = [40, 0, 0]       # packet size 
    gen_delay = [1, 1, 1]       # time units between packet generation
    trans_delay = 1             # time units between transmissions
    ug_maxlen = [0, 0, 0]       # queue max length per user group
    #ug_keep = [False, False, False]  # keep packets after transmission
    ug_keep = [True, True, True]  # keep packets after transmission
    ug_pkts = [3, 0, 0]         # packets to generate each time
    nr_res = [0, 0, 3]         # quantity of resources Poor, Fair, Good
    assgn_mat = [
        ["SL-1", "Good", 3]]   # Good is 11*8*1=88, other: 14*1*12=168
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
    gen_delay = [1, 1, 1]       # time units between packet generation
    trans_delay = 1             # time units between transmissions
    ug_maxlen = [0, 0, 0]       # queue max length per user group
    ug_keep = [False, False, False]  # keep packets after transmission
    ug_pkts = [3, 0, 0]         # packets to generate each time
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

    print("Example scheduler, testing.")
    print("Creates a scenery of 1 BaseStation, 2 Slices and 3 UserGroups;")
    print("  number o UserEquipments and entities attributes are configurable.")
    # replace initially imported classes with overwritten classes
    mk_imports(nm_channel=Channel, nm_chanenv=ChannelEnvironment, \
        nm_trblk=TransportBlock, nm_trfgen=TrafficGenerator, \
        nm_scheduler=Scheduler, nm_slice=Slice)
    ls_tests = [ \
        ["Massive test, verify.", test_mass_vrfy], \
        ["Massive test, run.", test_mass_run], \
        ] 
    print("Choose option, 'q' to quit:")
    op = run_qa_tests(ls_tests)



