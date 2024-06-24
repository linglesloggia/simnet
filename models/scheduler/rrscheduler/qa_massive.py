#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# qa_massive : tests a round robin scheduler example

'''Tests a round robin scheduler.
'''

# import classes from the main library
from libsimnet.simulator import Setup, Simulation, mk_imports
from libsimnet.results import Statistics
from libsimnet.libutils import run_qa_tests

# import concrete classes overwritten from the main library abstract classes
from models.channel.randfixchan.randfix_chan import Channel, ChannelEnvironment
from models.trafficgen.simpletrfgen.simple_trfgen import TrafficGenerator
from models.transpblock.simpletrblk.simple_trblk import TransportBlock 
from rr_scheduler import Scheduler      # the scheduler under test

# import functions for a simple simulation scenery
def run_sim(time_sim=5, nr_usreqs=[1,1], loss_prob=0.2, 
        gen_size=[40,40], gen_delay=[1,1], ug_pkts=[2,3],
        ug_maxlen=[3,3], ug_keep=[True,True],
        nr_res=[4,4], assgn_mat=[], 
        trans_delay=1,
        dbg_setup=[True,True,True], dbg_run=True,
        ls_scene=None):

    '''Runs simulation configured according to received parameters. If parameter ls_scene is not given (ls_scene=None), a Setup object is created with the parameters given. If values of attributes not present in the parameters list need to be changed, a Setup object may be created first, its attribute values adjusted as necessary using the Setup function change_attrs(), and then invoke this function with the ls_scene parameter list. In this case, parameters nr_usreqs, nr_res, and assgn_mat are irrelevant, sin ce they were previously used in the creation of the Setup object. Parameter ls_scene is a list which contains the objects produced by the Setup() constructor and the Setup method mk_entities().
    @param time_sim: number of times to run simulation.
    @param nr_usreqs: quantity of user equipments on each user group.
    @param loss_prob: probability that a transport block is lost.
    @param gen_size: size in bits of data to include in packets to generate.
    @param gen_delay: delay between packet generation.
    @param ug_pkts: quantity of packets to generate on each round, same for each user group.
    @param ug_maxlen: user equipment queue maximum length, same for each user group.
    @param ug_keep: whether to keep packets after transmission, same for each user group.
    @param nr_res: quantity of resources to generate, of qualities Poor, Fair, Good.
    @param assgn_mat: resource to slice assign matrix.
    @param trans_delay: delay between transmissions.
    @param dbg_setup: a list, debug for slice, user equipments, traffic generators.
    @param dbg_run: debug options for simulation run.
    @param ls_scene: a list [setup_obj, ls_setup, ls_slices, ls_trfgen, dc_objs] to use a Setup object with entities already created.
    '''

    # setup scenery
    if ls_scene:    # a Setup object has already been created
        setup_obj, ls_setup, ls_slices, ls_trfgen, dc_objs = ls_scene
    else:
        setup_obj, ls_setup = setup_sim(nr_usreqs, nr_res, assgn_mat)
        ls_slices, ls_trfgen, dc_objs = setup_obj.mk_entities(debug=False)

    # channel configuration
    dc_pars = {"loss_prob":loss_prob}
    setup_obj.change_attrs("Channel", "SL-1", dc_pars)

    # traffic generator configuration
    dc_pars = {"gen_size":gen_size[0], "gen_delay":gen_delay[0],
        "nr_pkts":ug_pkts[0]}
    setup_obj.change_attrs("TrafficGenerator", "UG-1", dc_pars)
    dc_pars = {"gen_size":gen_size[1], "gen_delay":gen_delay[1],
        "nr_pkts":ug_pkts[1]}
    setup_obj.change_attrs("TrafficGenerator", "UG-2", dc_pars)
   
    # packet queue configuration
    dc_pars = {"max_len":ug_maxlen[0], "keep_pkts":ug_keep[0]}
    setup_obj.change_attrs("PacketQueue", "UG-1", dc_pars)
    dc_pars = {"max_len":ug_maxlen[1], "keep_pkts":ug_keep[1]}
    setup_obj.change_attrs("PacketQueue", "UG-2", dc_pars)

    # TODO trans_delay

    # debug options
    dbg_default = [False, False, False]
    dbg_slc, dbg_usreq, dbg_trfgen = dbg_setup if dbg_setup else dbg_default

    # verification of simulation scenary
    if dbg_slc or dbg_usreq or dbg_trfgen:
        print("\n=== Simulation setup")
    if dbg_slc:
        print("--- Base stations and slices")
        setup_obj.show_slices("BsStat-0")
    if dbg_usreq:
        setup_obj.show_dc_objs()
        print("--- User equipments")
        print("    User equipments {:d}, duration {}\n".
            format(len(ls_trfgen), time_sim))
    if dbg_trfgen:
        print("--- Traffic generators")
        setup_obj.show_trfgens()

    # run simulation
    sim_obj = Simulation(time_sim, setup_obj=setup_obj)
    start_time, end_time = sim_obj.simnet_run(debug=dbg_run)
    stat_obj = Statistics(sim_obj)  # create Statistics object

    if dbg_run:
        print("\n=== Simulation results, queues final state")
        stat_obj.show_sent_rec()
    print("\n=== Simulation results, total counters and queues throughput")
    print("    Duration: simulation time {}, real time {:.3f} s".\
        format(time_sim, end_time - start_time))
    stat_obj.sim_totals(show=True)
    stat_obj.show_usereq_stats()

    return

def setup_sim(nr_usreqs=[1,1,2], nr_res=[5,4,4], rs_assgn_mat=[]):
    '''Creates an example of a simulation scenery with default values.

    Entities are created with their default parameter values, but may be changed later, using the method change_attrs() of the Setup class.
    @param nr_usreqs: a list, numbers of user equipments to create for each slice.
    @param nr_res: a list, numbers of resources to create of types Poor, Fair, Good.
    @param rs_assgn_mat: resource assignment matrix.
    @return: a setup object and a setup list.
    '''

    ls_setup = []

    # base station
    ls_setup += [ ["BaseStation", 1, None, "BsStat-"] ]

    # slices, assigned to base station
    ls_setup += [ ["Slice", 1, "BsStat-0", "Bs0Slice-"] ]

    # user groups, assigned to slices
    nr_ues_ug1, nr_ues_ug2 = nr_usreqs  # UserEqs per UserGroup
    # group 1
    ls_setup += [ ["UserGroup", 1, "Bs0Slice-0", "Bs0Sl0UG1", nr_ues_ug1] ]
    # group 2 
    ls_setup += [ ["UserGroup", 1, "Bs0Slice-0", "Bs0Sl0UG2", nr_ues_ug2] ]
 
    # resources for base station; different syms_slot, nr_slots, nr_sbands 
    nr_res_poor, nr_res_fair, nr_res_good = nr_res
    ls_setup += \
        [ ["Resource", nr_res_poor, "BsStat-0", "Bs0Res-", "Poor", 11, 2, 1] ]
    ls_setup += \
        [ ["Resource", nr_res_fair, "BsStat-0", "Bs0Res-", "Fair", 11, 4, 1] ]
    ls_setup += \
        [ ["Resource", nr_res_good, "BsStat-0", "Bs0Res-", "Good", 11, 8, 1] ]
    # resource assignment matrix
    rs_assgn_mat_0 = [
        ["SL-1", "Poor", 5], ["SL-1", "Good", 2],
        ["SL-2", "Fair", 4], ["SL-2", "Good", 2]
        ]
    rs_assgn_mat = rs_assgn_mat if rs_assgn_mat else rs_assgn_mat_0
    # assign base station resources to slices with assignment matrix
    ls_setup += [ ["ResAssign", 0, "BsStat-0", rs_assgn_mat] ]

    setup_obj = Setup(ls_setup)
    # use setup_obj.change_attrs to change attributes and configure scenery

    return setup_obj, ls_setup


def test_mass_vrfy():
    '''Setup and run a long simulation with many users, verify.

    This test creates the Setup object and makes its entitities to change attributes not included in the parameters list of the run_sim() function.
    '''
    time_sim = 5              # simulation duration
    nr_usreqs = [2, 0]       # user eqs qty per user group
    loss_prob = 0.0             # probability of losing packets
    gen_size = [500, 500]  # packet size 
    gen_delay = [0.5, 0.5]       #  generation delay
    trans_delay = 1             # transmission steps
    ug_maxlen = [500, 300]   # queue max length per user group
    ug_keep = [False, False]  # keep packets after transmission
    ug_pkts = [2, 2]        # packets to generate on each round
    nr_res = [0, 0, 3]          # quantity of resources Poor, Fair, Good
    assgn_mat = [
        ["SL-1", "Good", 3]]    # Good is 11*8*1=88, other: 14*1*12=168
    dbg_setup = [True,True, True]  # setup debug
    dbg_run = False

    # make Setup object and entities first to change attributes
    setup_obj, ls_setup = setup_sim(nr_usreqs, nr_res, assgn_mat)
    ls_slices, ls_trfgen, dc_objs = setup_obj.mk_entities(debug=True)
    # add dc_profile to UserGroup objects, change Channel attributes
    dc_ugr_prof = {"lim_res":3, "res_type":"Good"}
    dc_chan_pars = {"chan_mode":"Fixed", "val_1":100}
    for ugr in ["UG-1", "UG-2"]:
        setup_obj.change_attrs("UserGroup", ugr, {"dc_profile":dc_ugr_prof} )
        setup_obj.change_attrs("Channel", ugr, dc_chan_pars)
    # change Scheduler debug attribute to print messages
    dc_sched_pars = {"ul_dl":"DL","debug":False}
    for slc in ["SL-1"]:
        setup_obj.change_attrs("Scheduler", slc, dc_sched_pars)
    # run simulation
    run_sim(time_sim, nr_usreqs, loss_prob, 
        gen_size, gen_delay, ug_pkts,
        ug_maxlen, ug_keep,
        nr_res, assgn_mat,
        trans_delay, 
        dbg_setup, dbg_run, 
        ls_scene=[setup_obj, ls_setup, ls_slices, ls_trfgen, dc_objs])
    return


def test_mass_run():
    '''Setup and run a long simulation with many users, run.

    This test creates the Setup object and makes its entitities to change attributes not included in the parameters list of the run_sim() function.
    '''
    time_sim = 50           # simulation duration
    nr_usreqs = [2, 0]       # user eqs qty per user group
    loss_prob = 0.0             # probability of losing packets
    gen_size = [500, 500]  # packet size 
    gen_delay = [0.5, 0.5]       #  generation delay
    trans_delay = 1             # transmission steps
    ug_maxlen = [500, 300]   # queue max length per user group
    ug_keep = [False, False]  # keep packets after transmission
    ug_pkts = [2, 2]        # packets to generate on each round
    nr_res = [0, 0, 3]          # quantity of resources Poor, Fair, Good
    assgn_mat = [
        ["SL-1", "Good", 3]]    # Good is 11*8*1=88, other: 14*1*12=168
    dbg_setup = [False,False, False]  # setup debug
    dbg_run = False
    # make Setup object and entities to change attributes
    setup_obj, ls_setup = setup_sim(nr_usreqs, nr_res, assgn_mat)
    ls_slices, ls_trfgen, dc_objs = setup_obj.mk_entities(debug=False)
    # add dc_profile to UserGroup objects, change Channel attributes
    dc_profile = {"lim_res":3, "res_type":"Good"}
    dc_pars = {"chan_mode":"Fixed", "val_1":100}
    for ugr in ["UG-1", "UG-2"]:
        setup_obj.change_attrs("UserGroup", ugr, {"dc_profile":dc_profile} )
        setup_obj.change_attrs("Channel", ugr, dc_pars)
    #uncomment to print messages, change Scheduler debug attribute
    dc_sched_pars = {"ul_dl":"DL","debug":False}
    for slc in ["SL-1"]:
        setup_obj.change_attrs("Scheduler", slc, dc_sched_pars)
    #run simulation
    run_sim(time_sim, nr_usreqs, loss_prob, 
        gen_size, gen_delay, ug_pkts,
        ug_maxlen, ug_keep,
        nr_res, assgn_mat,
        trans_delay, 
        dbg_setup, dbg_run, 
        ls_scene=[setup_obj, ls_setup, ls_slices, ls_trfgen, dc_objs])

 
    return


# docstring for pydoctor out of function
ls_tests = []
'''List of tests to show in menu.'''

if __name__ == "__main__":

    print("Round robin scheduler, testing.")
    print("Creates a scenery of 1 BaseStation, 2 Slices and 3 UserGroups;")
    print("  number o UserEquipments and entities attributes are configurable.")
    # replace initially imported classes with overwritten classes
    mk_imports(nm_channel=Channel, nm_chanenv=ChannelEnvironment, \
        nm_trblk=TransportBlock, nm_trfgen=TrafficGenerator, \
        nm_scheduler=Scheduler)
    ls_tests = [ \
        ["Massive test, verify.", test_mass_vrfy], \
        ["Massive test, run.", test_mass_run], \
        ] 
    print("Choose option, 'q' to quit:")
    op = run_qa_tests(ls_tests)



