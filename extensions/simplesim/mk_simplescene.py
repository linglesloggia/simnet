#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# mk_simplescene : make a simple simulation scenery

'''Simple scenery with a base station, two slices, three user groups.

Creates a simple simulation scenery with one base station, two slices and three user groups; the number of user equipments in each user group is configurable, as well as the rest of the attributes of the different objects which comprise the simulation scenery.

This simple scenery allows for running simulations with different configurations. It also provides an example code which may be modified for a different number of base stations, slices, or user groups.

Transport block size, and hence number of packets it can contain, are determined from the following default values:
    - resources categories are Poor, Fair or Good, providing 22, 44, or 88 symbols respectively;
    - channel state determines bits per symbol 0, 1, 2, 4 or 8 in a random sort of equal probability for each size;
    - transport block size is bits per symbol * quantity of symbols;
    - these allow for transport block sizes from 0 to 88*8=704 bits;
    - for a packet size of 40 bits, a transport block may contain 0 to 17 packets.
'''

# import classes from the main library
from libsimnet.simulator import Setup, Simulation, mk_imports
from libsimnet.results import Statistics


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
    ls_setup += [ ["Slice", 2, "BsStat-0", "Bs0Slice-"] ]

    # user groups, assigned to slices
    nr_ues_ug1, nr_ues_ug2, nr_ues_ug3 = nr_usreqs  # UserEqs per UserGroup
    # group 1
    ls_setup += [ ["UserGroup", 1, "Bs0Slice-0", "Bs0Sl0UG1", nr_ues_ug1] ]
    # group 2 
    ls_setup += [ ["UserGroup", 1, "Bs0Slice-0", "Bs0Sl0UG2", nr_ues_ug2] ]
    # group 3
    ls_setup += [ ["UserGroup", 1, "Bs0Slice-1", "Bs0Sl1UG1", nr_ues_ug3] ]

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


def run_sim(time_sim=5, nr_usreqs=[1,1,2], loss_prob=0.2, 
        gen_size=[40,40,40], gen_delay=[1,1,1], ug_pkts=[2,3,4],
        ug_maxlen=[3,3,0], ug_keep=[True,True,True],
        nr_res=[4,4,4], assgn_mat=[], 
        trans_delay=1,
        dbg_setup=[True,True,True], dbg_run=True,
        ls_scene=None):

    '''Runs simulation configured according to received parameters. If parameter ls_scene is not given (ls_scene=None), a Setup object is created with the parameters given. If values of attributes not present in the parameters list need to be changed, a Setup object may be created first, its attribute values adjusted as necessary using the Setup function change_attrs(), and then invoke this function with the ls_scene parameter list. In this case, parameters nr_usreqs, nr_res, and assgn_mat are irrelevant, sin ce they were previously used in the creation of the Setup object. Parameter ls_scene is a list which contains the objects produced by the Setup() constructor and the Setup method mk_entities().
    @param time_sim: number of times to run simulation.
    @param nr_usreqs: quantity of user equipments on each user group.
    @param loss_prob: probability that a transport block is lost.
    @param gen_size: size in bits of data to include in packets to generate.
    @param gen_delay: delay between packet generation.
    @param ug_pkts: quantity of packets to generate each time, same for each user group.
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
    dc_pars = {"gen_size":gen_size[2], "gen_delay":gen_delay[2],
        "nr_pkts":ug_pkts[2]}
    setup_obj.change_attrs("TrafficGenerator", "UG-3", dc_pars)

    # packet queue configuration
    dc_pars = {"max_len":ug_maxlen[0], "keep_pkts":ug_keep[0]}
    setup_obj.change_attrs("PacketQueue", "UG-1", dc_pars)
    dc_pars = {"max_len":ug_maxlen[1], "keep_pkts":ug_keep[1]}
    setup_obj.change_attrs("PacketQueue", "UG-2", dc_pars)
    dc_pars = {"max_len":ug_maxlen[2], "keep_pkts":ug_keep[2]}
    setup_obj.change_attrs("PacketQueue", "UG-3", dc_pars)

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
