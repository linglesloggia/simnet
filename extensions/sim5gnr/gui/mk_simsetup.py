#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# libsimnet.mk_simsetup : example template to setup a simulation scenery

'''SimNet template to setup a simulation scenery.

Creates the lists of objects necessary to run the simulation; these lists are passed as parameters to a Simulation object, which runs the simulation. 

This example template creates only one object of each class; it is just an example of object creation, their required parameters, and how to attach each new object to previuosly created objects to define a complete simulation structure.

This code is not optimized for execution, but for showing how to manually build a simulation scenery. To this end, some redundancy may be detected, e.g. by getting pointers to objects which in this context are already available, but may be necessary in the general case.
'''

### imports to access classes which define the simulation scenery
# import classes from the main library
import sys,io,os,pickle


#sys.path.append('/home/lingles/ownCloud/Summer_school/simnet')
ruta_libsimnet = "/home/lingles/ownCloud/Summer_school/simnet"
if ruta_libsimnet not in sys.path:
    sys.path.append(ruta_libsimnet)

from libsimnet.simulator import Simulation
from libsimnet.results import Statistics
from libsimnet.pktqueue import PacketQueue
from libsimnet.basenode import BaseStation, InterSliceSched, UserGroup 
from libsimnet.usernode import UserEquipment

# import classes from 5g library

from extensions.sim5gnr.basenode import Resource,Slice
from extensions.sim5gnr.usernode import TransportBlock

# import concrete classes overwritten from abstract classes
import models.channel.randfixchan.randfix_chan as fix_chan
import models.channel.randfixchan.randfix_chan  as fix_chenv
import models.channel.filechannel.file_channel  as file_channel
import models.channel.filechannel.file_chenv  as file_chenv

import models.trafficgen.simpletrfgen.simple_trfgen  as trf_simple
import models.trafficgen.poissontrfgen.poisson_trfgen  as trf_poisson

import models.scheduler.simplesched.simple_sched as sch_simple
import models.scheduler.rrscheduler.rr_scheduler as sch_rr
import models.scheduler.maxcqisched.maxcqischeduler as sch_maxcqi
import models.scheduler.pfscheduler.pfscheduler as sch_pf
import models.scheduler.dqnscheduler.dqnschedulerlearner as sch_dqnlearn
import models.scheduler.dqnscheduler.dqnscheduler as sch_dqn
import models.scheduler.dr.dqnschedulerlearner as sch_drlearn
import models.scheduler.dr.dqnscheduler as sch_dr

from extensions.sim5gnr.gui.AppGui import ConfigScenary
import extensions.sim5gnr.gui.AppGui as gui
# Python imports



 #        self.num_ues = [2]

class GuiConfigRunSimulation:

    def __init__(self):
        self.conf = ConfigScenary()
        self.output_config = open(gui.file_log_config,'w')
        self.set_config()
        
        #output = io.StringIO()

    def set_config(self):
        
        self.config_basenode(band = self.conf.band ,robustMCS=self.conf.rMCS  ,long_cp = self.conf.longcp,ul = self.conf.ul,dl = self.conf.dl,mimo_mod =self.conf.mimo,n_slices =self.conf.num_slices ,n_user_groups = self.conf.nuegroups  ,id_base = self.conf.bs_name,time_sim = self.conf.time_sim)  
        self.config_slices(idc_slc = self.conf.name_slice,numerology = self.conf.numerology)        
        self.config_resources(res_type = self.conf.name_resdl,nr_items = self.conf.nresdl,syms_slot =self.conf.sym_slot )
        self.config_sched(self.conf.name_sched) 
        self.config_channel(channel_name = self.conf.channel_type,f_name = self.conf.file_channel,channel_mode = self.conf.channel_mode,val_1 = self.conf.val_1,val_2 = self.conf.val_2,loss=self.conf.loss_prob)       
        sl_ue_id = [[]]
        sl_ue_profile = [[]]
        n_ues = [[]]
        for i in range(self.conf.num_slices):
            sl_ue_id.append([])
            sl_ue_profile.append([])
            n_ues.append([])
            for j in range(len(self.conf.slice_uegr[i])):
                sl_ue_id[i].append(self.conf.name_uegroup[self.conf.slice_uegr[i][j]])
                sl_ue_profile[i].append({"lim_res":self.conf.uegr_par1[self.conf.slice_uegr[i][j]], "res_type":"PRB"})
                n_ues[i].append(self.conf.num_ues[self.conf.slice_uegr[i][j]])
        self.config_uegr_slices(sl_ue_id = sl_ue_id ,sl_ue_profile= sl_ue_profile ,n_ues = n_ues)
        self.config_trfgen_uegr(trfgen_type =self.conf.trgen_type,gen_size = self.conf.pkt_size,gen_delay =self.conf.inter_arrival,nr_pkts = self.conf.burst_size ,size_dist =self.conf.size_dist,max_len =self.conf.max_len, keep_pkts= self.conf.keep_pkts,last_k= self.conf.last_k)

    def config_basenode(self,band = "n258" ,robustMCS=False,long_cp = False,ul = False,dl = True,mimo_mod ="SU",n_slices =1,n_user_groups = 1,id_base = "BS-1",time_sim = 15):
        self.band = band
        self.robustMCS = robustMCS
        self.long_cp = long_cp
        self.ul = ul
        self.dl = dl
        self.uldl="DL"
        self.mimo_mod = mimo_mod
        self.nlayers=1
        #snr =15
        self.n_slices = n_slices
        self.n_user_groups = n_user_groups
        self.id_base = id_base
        ### set some simulation variables
        self.time_sim = time_sim

    def config_resources(self,res_type = ["PRB"],nr_items = [100],syms_slot =[14]):
        self.res_type = res_type
        self.n_res_types = len(res_type)
        self.nr_items = nr_items
        self.syms_slot =syms_slot
        self.nr_slots = []
        self.nr_sbands = []
        for i in range(len(self.syms_slot)):
            self.nr_slots.append(1)   ### fixed in 5g
            self.nr_sbands.append(12) #### fixed in 5g

    def config_slices(self,idc_slc= ["SL-1"],numerology=[1]):        
        self.id_slc = idc_slc 
        self.numerology = numerology

    def config_uegr_slices(self,sl_ue_id = ["UG-1"],sl_ue_profile= [{"lim_res":50, "res_type":"PRB"}],n_ues = [5]):

        self.sl_ue_id = sl_ue_id
        self.sl_ue_profile = sl_ue_profile
        self.n_ues = n_ues

    def config_trfgen_uegr(self,trfgen_type =["poisson"],gen_size = [300],gen_delay = [1],nr_pkts = [1],size_dist =["Exponential"],max_len =[0], keep_pkts= [False],last_k=[10]):

        self.trfgen_type = trfgen_type
        self.gen_size  = gen_size   # packet length in bits
        self.gen_delay =  gen_delay     # time between successive packet generation
        self.nr_pkts = nr_pkts
        self.size_dist = size_dist  # number or packets to generate on each  instant time
        self.max_len = max_len       # maximum queue lenght, then drop packets
        self.keep_pkts = keep_pkts 
        self.last_k = last_k           # time to store the data 

    def config_channel(self, channel_name = "random or fixed",f_name = "../../../models/channel/filechannel/snr_4ues_1000_5gUmi.csv",channel_mode = "Fixed",val_1 = -10,val_2 = 100,loss =0):       
        ###############channel
        #channel_name = "random or fixed"
        self.channel_name = channel_name
        ########Channel mode file
        self.f_name = f_name
        self.interpol = True
        self.chan_mode = channel_mode
        self.val_1 = val_1
        self.val_2 = val_2
        self.loss_prob = loss
        
    def config_sched(self, sched_type="round robin"): 
        self.sched_type = sched_type

        
        ######################################################################
    def create_simulation_scenary(self, file_config,debug):
        ### collections of objects required to setup the simulation scenery
        self.ls_trfgen = []
        '''List of traffic generators.'''
        self.ls_slices = []
        '''List of Slices.'''
        self.dc_objs = {}
        

        '''Container entities by identificator.'''
        with open(file_config,"rb") as fp:
            self.conf = pickle.load(fp)
            self.set_config()


        assgn_mat = []
        ### BaseStation
        # create a BaseStation object
            # BaseStation name
        pt_bs = BaseStation(self.id_base)   # creates with unique identifier BS-1
        self.dc_objs["BS-1"] = pt_bs        # add to entities dictionary
        
        
        ### InterSliceScheduler
        # create an InterSliceScheduler, unique for each BaseStation
        priority = 4    # order of event execution in a certain instant time
        pt_inter_sl_sched = InterSliceSched(priority, debug)
        # attach to BaseStation 1, BS-1
        pt_bs = self.dc_objs["BS-1"]    # get pointer to BaseStation to attach to
        pt_bs.inter_sl_sched = pt_inter_sl_sched   # attach to BaseStation
        
        
        ### Resource
        # repeat for all Resources to create for a BaseStation
        for i in range(self.n_res_types):
            attach_to = "BS-1"      # BaseStation unique identifier
            id_pref = "BS-1:Res-"   # prefix for a name; unique identifier RS-1      # Resource type
            self.dc_objs[attach_to].mk_dc_res(self.res_type[i],self.nr_items[i],  
            res_pars=[self.syms_slot[i], self.nr_slots[i], self.nr_sbands[i],self.band,self.dl,self.ul,self.long_cp],
            res_class=Resource )  # makes Resources, attach to BaseStation
        
        if debug:
            print("=== Simulation scenery ===\n",file=self.output_config)
            print("--- Base station and associated entities",file=self.output_config)
            print(pt_bs,file=self.output_config)
            print("        ", pt_bs.inter_sl_sched,file=self.output_config)
            for key in pt_bs.dc_res:
                print("    5g res    ", pt_bs.dc_res[key][0][0],file=self.output_config)
        
        
        ### Slice
        # create a Slice, attach to a BaseStation
        # repeat for all Slices attached to the BaseStation
        for i in range(self.n_slices):
            #id_slc = "Slc-1:BS-1"   # Slice name; unique identifier "SL-1"
            #trans_delay = 1         # time between successive transmisions
            #priority = 2            # order of event execution in a certain instant time
            pt_slc = Slice(self.id_slc[i], 1, 1,self.numerology[i])  # pointer to Slice
            self.dc_objs[self.id_slc[i]] = pt_slc    # add to entities dictionary
            self.ls_slices += [pt_slc]       # list of all Slices
            # attach Slice SL-1 to BaseStation BS-1
            pt_bs = self.dc_objs["BS-1"]     # get pointer to BaseStation to attach to
            pt_bs.dc_slc[pt_slc.id_object] = pt_slc     # attach to BaseStation
            
            if debug:
                print("    ", pt_slc,file=self.output_config) #, pt_obj.id_slc)
        
        
            ### Scheduler
            # create a Scheduler, attach to Slice
            # repeat for all Slices attached to the BaseStation
            if self.sched_type == "simple":
                pt_sched = sch_simple.Scheduler()      # only 1 scheduler per Slice, unique id SL-1
            if self.sched_type == "round robin":
                pt_sched = sch_rr.Scheduler(ul_dl = "DL") 
            if self.sched_type == "maxcqi":
                pt_sched = sch_maxcqi.Scheduler(ul_dl = "DL") 
            if self.sched_type == "proportional fair":
                pt_sched = sch_pf.Scheduler(ul_dl = "DL")
            if self.sched_type == "dqn":
                pt_sched = sch_dqn.Scheduler(ul_dl = "DL")
            if self.sched_type == "dqnlearn":
                pt_sched = sch_dqnlearn.Scheduler(ul_dl = "DL")
            if self.sched_type == "drlearn":
                pt_sched = sch_drlearn.Scheduler(ul_dl = "DL")
            if self.sched_type == "dr":
                pt_sched = sch_dr.Scheduler(ul_dl = "DL")
                
            pt_slc.sched_dl = pt_sched     # attach scheduler to Slice
        
            if debug:
                print("        ", pt_sched,file=self.output_config)
        
        
            ### UserGroup
            # create a UserGroup, attach to a Slice
            # repeat for all UserGroups in the Slice
            for uegr in range(len(self.sl_ue_id[i])):
                id_usr_grp = self.sl_ue_id[i][uegr]   # UserGroup name; unique identifier UG-1
                profile = self.sl_ue_profile[i][uegr]         # UserGroup profile
                pt_ugr = UserGroup(id_usr_grp, profile)     # pointer to UserGroup
                self.dc_objs[self.sl_ue_id[i][uegr]] = pt_ugr        # add to entities dictionary
                self.dc_objs[self.id_slc[i]].ls_usrgrps += [pt_ugr]  # add to list of UserGroup in Slice
            
                if debug:
                   print("    ", pt_ugr,file=self.output_config) #, pt_obj.id_slc)
                    
                      
                ####dc_ugr_prof = {"lim_res":2, "res_type":"Good"}
            
                ### UserEquipment
                # create a UserEquipment, attach to a UserGroup
                # repeat for all UserEquipments in the UserGroup
                for j in range(self.n_ues[i][uegr]):
                    pt_ugr = self.dc_objs[self.sl_ue_id[i][uegr]]    # UserGroup to attach UserEquipment to
                    v_pos = [0, 0, 0]           # UserEquipment position vector
                    v_vel = [0, 0, 0]           # UserEquipment velocity vector
                    usreq = UserEquipment(pt_ugr, v_pos, v_vel,make_tb="TBbyNrRes", debug=debug)    # id UE-1
                    #"TBbyNrRes" "OneTBbyRes" "OneTBallRes"
                    self.dc_objs[self.sl_ue_id[i][uegr] ].ls_usreqs += [usreq]    # add to list in UserGroup
                    self.dc_objs[self.id_slc[i]].ls_usreqs += [usreq]    # add to list in Slice
                    if debug:
                            print("    ", usreq) #, pt_obj.id_slc)
                    ### ChannelEnvironment
                    # create a channel environment to which channels will be attached
                            
                    ### Channel
                    # create Channel, assign ChannelEnvironment to Channel, Channel to UserEquipment
                    # repeat for each UserEquipment
                    
                    if self.channel_name == "random or fixed":
                        pt_ch_env = fix_chenv.ChannelEnvironment()
                        usreq.chan = fix_chan.Channel(pt_ch_env, self.loss_prob,chan_mode=self.chan_mode,val_1=self.val_1,val_2=self.val_2) 
                    if self.channel_name == "file": 
                        pt_ch_env = file_chenv.ChannelEnvironment()
                        usreq.chan = file_channel.Channel(pt_ch_env, interpol=self.interpol, debug=debug, f_name=self.f_name,loss_prob=self.loss_prob)
                        usreq.chan.id_usreq = usreq.id_object
                    
                    
                    ### TransportBlock
                    # create transport block
                    # repeat for each UserEquipment
                    usreq.tr_blk = TransportBlock(self.band,self.robustMCS,self.uldl,self.mimo_mod,self.nlayers)
                    # creates and assigns TransportBlock
                    
                    
                    ### PacketQueue
                    # create PacketQueue for UserEquipment, attach to UserEquipment
                    # repeat for each UserEquipment in all UserGroups
                    qu_prefix="PQ-"     # PacketQueue prefix for unique identifier
                    pkt_prefix="Pkt-"   # packet prefix for this queue unique packet identifier
                    max_len=self.max_len[i]         # maximum queue lenght, then drop packets
                    keep_pkts=self.keep_pkts[i]  # keep packets after successfully transmitted
                    last_k = self.last_k[i]
                    usreq.pktque_dl = PacketQueue(qu_prefix, pkt_prefix, max_len, keep_pkts,last_k)
                    
                    ### TrafficGenerator
                    # create TrafficGenerator, attach to UserEquipment data PacketQueue
                    # repeat for each PacketQueue in each UserEquipment
                    priority = 1        # order of event execution in a certain instant time

                    if self.trfgen_type[uegr] =="fixed":
                        pt_tg_obj = trf_simple.TrafficGenerator(usreq.pktque_dl, self.gen_size[uegr], \
                        self.gen_delay[uegr], self.nr_pkts[uegr], priority, debug)
                    if self.trfgen_type[uegr] =="poisson":
                        pt_tg_obj = trf_poisson.TrafficGenerator(usreq.pktque_dl, gen_size=self.gen_size[uegr], \
                        gen_delay=self.gen_delay[uegr], nr_pkts=self.nr_pkts[uegr], priority=priority,size_dist=self.size_dist[uegr], debug=debug)      
                        
                    # add traffic generator to list of traffic generators
                    usreq.trf_gen = pt_tg_obj
                    self.ls_trfgen += [pt_tg_obj] # list of all traffic generators
                    if debug:
                            print("    ", pt_tg_obj) #, pt_obj.id_slc)
            
                print(" "*8, pt_ugr,file=self.output_config)
                for ueq in pt_ugr.ls_usreqs:
                    print(" "*12, ueq,file=self.output_config)
                    print(" "*16, ueq.chan,file=self.output_config)
                    print(" "*20, ueq.chan.ch_env,file=self.output_config)
                    print(" "*16, ueq.tr_blk,file=self.output_config)
                    print(" "*16, ueq.pktque_dl,file=self.output_config)
                    print(" "*16, ueq.trf_gen,file=self.output_config)
                ### assign Resources to Slices
            # the assign matrix assigns all available Resources to Slices
            #    [ Slice id, Resource type, quantity to assign ]
            assgn_mat.append([ self.id_slc[i],self.res_type[i], self.nr_items[i]] )    # Resource assignment matrix
       
        attach_to = "BS-1"              # BaseStation for Resource assignment 
        bs = self.dc_objs[attach_to]         # BaseStation object
        bs.inter_sl_sched.assign_slc_res(bs.dc_slc, bs.dc_res, assgn_mat)
                # assigns Resources to Slices according to the assign matrix
        

        print("--- Traffic generators list",file=self.output_config)
        for trf_gen in self.ls_trfgen:
            print(" "*4, trf_gen, "Queue ", trf_gen.pktque.id_object,file=self.output_config)
        print("--- Slices list",file=self.output_config)
        for slc in self.ls_slices:
            print("    ", slc,file=self.output_config)
        print("--- Dictionary of objects",file=self.output_config)
        for key in self.dc_objs:
            print("    ", key, self.dc_objs[key],file=self.output_config) 

        self.output_config.close()
        return 

        
    def run_simulation(self,debug):
        ### run simulation
        # self.output_config = open('config_log.txt','w')

        # with open("config.pickle","rb") as fp:
        #     self.cf = pickle.load(fp)
        # self.set_config()
        orig_stdout = sys.stdout
        f = open(gui.file_log_run, 'w')
        
        sys.stdout = f
        

        self.output_run_rec = open(gui.output_run_rec  ,'w')
        self.output_run_tr = open(gui.output_run_tr ,'w')
        self.output_run_res = open(gui.output_run_res,'w')
        self.output_run_ch_st = open(gui.output_run_ch_st,'w')
        self.output_run_queue = open(gui.output_run_queue,'w')
        self.output_run_names = open(gui.output_run_names,'w')

        print("-------------------simulation starts-------------------")
        dbg_run = debug
        #gen_delay, trans_delay = 2, 1
        #sim_obj = Simulation(time_sim, gen_delay, trans_delay, \
        sim_obj = Simulation(self.time_sim, ls_trfgen=self.ls_trfgen, ls_slices=self.ls_slices)
        start_time, end_time = sim_obj.simnet_run(debug=dbg_run)
        stat_obj = Statistics(sim_obj)   # create Statistics object
        if dbg_run:
            print("\n=== Simulation results, queues final state ===")
            stat_obj.show_sent_rec()
        print("    Simulation duration {:.3f} s".format(end_time - start_time))
        stat_obj.sim_totals(show=True)
        print("---------------------------------------------")
        for  slc in self.ls_slices:
            for ueq in slc.ls_usreqs:
                self.output_run_res.write(str(slc.dc_usreq_traf[ueq.id_object]))
                self.output_run_res.write('\n')
                self.output_run_queue.write(str(slc.dc_usreq_q_delay[ueq.id_object]))
                self.output_run_queue.write('\n')

                self.output_run_ch_st.write(str(slc.dc_usreq_channel[ueq.id_object]))
                self.output_run_ch_st.write('\n')

                queue = ueq.pktque_dl
                self.output_run_rec.write(str(queue.last_lst_rec))
                self.output_run_rec.write('\n')
                self.output_run_tr.write(str(queue.last_lst_snt))
                self.output_run_tr.write('\n')
                self.output_run_names.write(slc.id_object+"/"+ueq.usr_grp.id_object+"/"+ueq.id_object)
                self.output_run_names.write('\n')

        self.output_run_rec.close()
        self.output_run_tr.close()
        self.output_run_res.close()
        self.output_run_ch_st.close()
        self.output_run_queue.close()

        sys.stdout = orig_stdout
        f.close()
        # self.output_config.close()


if __name__ == "__main__":
    
    file_config = sys.argv[1]
    run = eval(sys.argv[2])
    debug = eval(sys.argv[3])
    
    gf = GuiConfigRunSimulation()
    gf.create_simulation_scenary(file_config,debug)

    if run == True:
        gf.run_simulation(debug)

