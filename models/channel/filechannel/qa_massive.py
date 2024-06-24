#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# qa_massive : tests channel from file model

'''Tests channel from file model.
'''

# import classes from the main library
from libsimnet.simulator import Setup, Simulation, mk_imports
from libsimnet.results import Statistics
from libsimnet.libutils import run_qa_tests

# import concrete classes overwritten from the main library abstract classes
from models.transpblock.simpletrblk.simple_trblk import TransportBlock 
from models.trafficgen.simpletrfgen.simple_trfgen import TrafficGenerator
from models.scheduler.simplesched.simple_sched import Scheduler
from file_channel import Channel
from file_chenv import ChannelEnvironment

# import functions for a simple simulation scenery
from extensions.simplesim.mk_simplescene import setup_sim, run_sim

# Python imports
import sys


def channel_state_vrfy():
    '''Shows some channel states obtained from file.
    '''
    # make dictionary of channel states in class Channel
    #f_name = ".././models/channel/filechannel/snr_4ues_1000_5gUmi.csv"
    f_name = "snr_4ues_1000_5gUmi.csv"
    ch_env_obj = ChannelEnvironment()
    chan_obj = Channel(ch_env_obj, interpol=False, debug=True, f_name=f_name)
    chan_obj.id_usreq = "UE-1"
    pos = [0,0,0]
    vel = [0,0,0]
    print("\n=== Verify with exact values")
    for time_t in range(1, 7):
        t_chan = time_t * 0.001
        chan_state = chan_obj.get_chan_state(t_chan, pos, vel)
        print("{} : time: {:6.5f}, channel state: {:6.5f}".\
            format(chan_obj.id_usreq, t_chan, chan_state) )
    print("\n=== Verify interpolation")
    chan_obj.index = 0          # reposition at initial time
    chan_obj.interpol = True    # interpolate
    for time_t in range(1, 7):
        t_chan = time_t * 0.00153
        #t_chan = time_t * 0.001
        chan_state = chan_obj.get_chan_state(t_chan, pos, vel)
        print("{} : time: {:6.5f}, channel state: {:6.5f}".\
            format(chan_obj.id_usreq, t_chan, chan_state) )
    return


def test_mass_vrfy():
    '''Setup and run a simulation test with many users, variable time.
    '''

    # make dictionary of channel states in class Channel, a class variable
    f_name = "snr_4ues_1000_5gUmi.csv"
    ch_env_obj = ChannelEnvironment()
    chan_obj = Channel(ch_env=ch_env_obj, interpol=False, debug=True, f_name=f_name)

    time_sim = 3                # simulation duration
    nr_usreqs = [3, 0, 0]       # user eqs qty per user group
    loss_prob = 0.0             # probability of losing packets
    gen_size = [80, 0, 0]       # packet size 
    gen_delay = [1, 1, 1]       # packet size and generation delay
    trans_delay = 1             # transmission steps
    ug_maxlen = [0, 0, 0]       # queue max length per user group
    ug_keep = [False, False, False]  # keep packets after transmission
    ug_pkts = [2, 0, 0]         # packets to generate on each round
    nr_res = [0, 0, 3]          # quantity of resources Poor, Fair, Good
    assgn_mat = [
        ["SL-1", "Good", 3]]    # Good is 11*8*1=88, other: 14*1*12=168
    dbg_setup = [True, True, True]  # setup debug
    dbg_run = True

    # make Setup object and entities first to change attributes
    setup_obj, ls_setup = setup_sim(nr_usreqs, nr_res, assgn_mat)
    ls_slices, ls_trfgen, dc_objs = setup_obj.mk_entities(debug=False)
    # add attribute usreq to Channel objects to get channel state from file
    for ugr in ["UG-1", "UG-2", "UG-3"]:
        pt_obj = setup_obj.dc_objs[ugr]  # UserGroup object
        for usreq in pt_obj.ls_usreqs:   # UserEquipment in this UserGroup
            # add attributes to channel from file
            usreq.chan.id_usreq = usreq.id_object  # adds UserEquipment id
            usreq.chan.f_name = f_name             # adds channel state file
    #setup_obj.show_slices("BS-1")

    # change simulation parameters and run with help function in mk_simplescene
    run_sim(time_sim, nr_usreqs, loss_prob, 
        gen_size, gen_delay, ug_pkts,
        ug_maxlen, ug_keep,
        nr_res, assgn_mat,
        trans_delay, 
        dbg_setup, dbg_run,
        ls_scene=[setup_obj, ls_setup, ls_slices, ls_trfgen, dc_objs])
    return


def test_mass_run():
    '''Test with many users and long time, run.
    '''
    print("Work in progress; may be same as verify with higher numbers.")
    """
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
    """
    return


# docstring for pydoctor out of function
ls_tests = []
'''List of tests to show in menu.'''

if __name__ == "__main__":

    mk_imports(nm_channel=Channel, nm_chanenv=ChannelEnvironment, \
        nm_trblk=TransportBlock, nm_trfgen=TrafficGenerator, \
        nm_scheduler=Scheduler)

    ls_tests = [ \
        ["Channel states, verify.", channel_state_vrfy], \
        ["Massive test, verify.", test_mass_vrfy], \
        ["Massive test, run.", test_mass_run], \
        ] 
    print("Choose option, 'q' to quit:")
    run_qa_tests(ls_tests)



