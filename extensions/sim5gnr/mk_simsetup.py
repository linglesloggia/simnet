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
import matplotlib.pyplot as plt
from pylab import rcParams




rcParams['figure.figsize'] = 11, 6
#sns.set_theme(style = "whitegrid")



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


class GuiConfigRunSimulation:

    def __init__(self):
        # determine current script path and data subdirectory path
        script_path = os.path.abspath(__file__)
        path_list = script_path.split(os.sep)
        script_directory = path_list[0:len(path_list)-1] 
        self.directory = "/".join(script_directory) + "/"
        # 
      

        #self.directory = "./extensions/sim5gnr/data/"
        self.file_log_config = "/content/drive/MyDrive/simnet/extensions/sim5gnr/data/config_log.txt"
        self.file_log_run = "/content/drive/MyDrive/simnet/extensions/sim5gnr/data/run_log.txt"
        self.file_output_run_rec = "/content/drive/MyDrive/simnet/extensions/sim5gnr/data/run_results_rec.txt"
        self.file_output_run_tr = "/content/drive/MyDrive/simnet/extensions/sim5gnr/data/data/run_results_tr.txt"
        self.file_output_run_res = "/content/drive/MyDrive/simnet/extensions/sim5gnr/data/run_results_res.txt"

        self.output_config = open(self.file_log_config,'w')

   
    def config_basenode(self,band = "n258" ,robustMCS=False,long_cp = False,ul = False,dl = True,mimo_mod ="SU",n_slices =1,n_user_groups = 1,id_base = "BS-1"):
        self.band = band
        self.robustMCS = robustMCS
        self.long_cp = long_cp
        self.ul = ul
        self.dl = dl
        self.ul_dl="DL"
        self.mimo_mod = mimo_mod
        self.nlayers=1
        #snr =15
        self.n_slices = n_slices
        self.n_user_groups = n_user_groups
        self.id_base = id_base
        ### set some simulation variables
        self.time_sim = 15

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
        print(" Resources configured ", self.res_type,self.n_res_types,self.nr_items,self.syms_slot,file=self.output_config)

    def config_slices(self,idc_slc= ["SL-1"],numerology=[1]):        
        self.id_slc = idc_slc 
        self.numerology = numerology
        print("slices configured", self.id_slc,self.numerology)

    def config_uegr_slices(self,sl_ue_id = ["UG-1"],sl_ue_profile= [{"lim_res":60, "res_type":"PRB"}],n_ues = [4]):

        self.sl_ue_id = sl_ue_id
        self.sl_ue_profile = sl_ue_profile
        self.n_ues = n_ues
        print("ues-slices configured", self.sl_ue_id,self.sl_ue_profile,self.n_ues,file=self.output_config)


    def config_trfgen_uegr(self,trfgen_type =["poisson"],gen_size = [300],gen_delay = [1],nr_pkts = [1],size_dist =["Exponential"]    ):

        self.trfgen_type = trfgen_type
        self.gen_size  = gen_size   # packet length in bits
        self.gen_delay =  gen_delay     # time between successive packet generation
        self.nr_pkts = nr_pkts
        self.size_dist = size_dist        # number or packets to generate on each  instant time
        print("Traffic Generators - UEgroups configured ", self.trfgen_type,self.gen_size,self.gen_delay,self.nr_pkts,self.size_dist,file=self.output_config)
        
    def config_channel(self, channel_name = "random or fixed",f_name = "../../../models/channel/filechannel/snr_4ues_1000_5gUmi.csv",channel_mode = "Fixed",val_1 = -10,val_2 = 100):       
        ###############channel
        #channel_name = "random or fixed"
        self.channel_name = channel_name
        ########Channel mode file
        self.f_name = f_name
        self.interpol = True
        self.chan_mode = channel_mode
        self.val_1 = val_1
        self.val_2 = val_2
        print("channel configured ", self.channel_name,self.f_name,self.chan_mode,self.val_1,self.val_2,file=self.output_config)
        
    def config_sched(self, sched_type="round robin"): 
        self.sched_type = sched_type
        print("Scheduler configured", self.sched_type,file=self.output_config)

        
        ######################################################################
    def create_simulation_scenary(self):
        ### collections of objects required to setup the simulation scenery
        self.ls_trfgen = []
        '''List of traffic generators.'''
        self.ls_slices = []
        '''List of Slices.'''
        self.dc_objs = {}
        '''Container entities by identificator.'''
        ## Set default configuration of the different simulation scenary
        self.config_basenode()  
        self.config_slices()        
        self.config_resources()
        self.config_sched() 
        self.config_channel()       
        self.config_uegr_slices()
        self.config_trfgen_uegr()


        assgn_mat = []
        ### BaseStation
        # create a BaseStation object
            # BaseStation name
        pt_bs = BaseStation(self.id_base)   # creates with unique identifier BS-1
        self.dc_objs["BS-1"] = pt_bs        # add to entities dictionary
        
        
        ### InterSliceScheduler
        # create an InterSliceScheduler, unique for each BaseStation
        priority = 4    # order of event execution in a certain instant time
        debug = True    # debug for InterSliceScheduler
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
            print("=== Simulation scenery ===\n")
            print("--- Base station and associated entities")
            print(pt_bs)
            print("        ", pt_bs.inter_sl_sched)
            for key in pt_bs.dc_res:
                print("    5g res    ", pt_bs.dc_res[key][0][0])
        
        
        ### Slice
        # create a Slice, attach to a BaseStation
        # repeat for all Slices attached to the BaseStation
        for i in range(self.n_slices):
            pt_slc = Slice(self.id_slc[i], 1, 1,self.numerology[i])  # pointer to Slice
            self.dc_objs[self.id_slc[i]] = pt_slc    # add to entities dictionary
            self.ls_slices += [pt_slc]       # list of all Slices
            # attach Slice SL-1 to BaseStation BS-1
            pt_bs = self.dc_objs["BS-1"]     # get pointer to BaseStation to attach to
            pt_bs.dc_slc[pt_slc.id_object] = pt_slc     # attach to BaseStation
            
            if debug:
                print("    ", pt_slc) #, pt_obj.id_slc)
        
        
            ### Scheduler
            # create a Scheduler, attach to Slice
            # repeat for all Slices attached to the BaseStation
            if self.sched_type == "simple":
                pt_sched = sch_simple.Scheduler()      # only 1 scheduler per Slice, unique id SL-1
            if self.sched_type == "round robin":
                pt_sched = sch_rr.Scheduler(ul_dl ="DL")  
            pt_slc.sched_dl = pt_sched     # attach scheduler to Slice
        
            if debug:
                print("        ", pt_sched)
        
        
            ### UserGroup
            # create a UserGroup, attach to a Slice
            # repeat for all UserGroups in the Slice
            id_usr_grp = self.sl_ue_id[i]   # UserGroup name; unique identifier UG-1
            profile = self.sl_ue_profile[i]         # UserGroup profile
            pt_ugr = UserGroup(id_usr_grp, profile)     # pointer to UserGroup
            self.dc_objs[self.sl_ue_id[i]] = pt_ugr        # add to entities dictionary
            self.dc_objs[self.id_slc[i]].ls_usrgrps += [pt_ugr]  # add to list of UserGroup in Slice
        
            if debug:
               print("    ", pt_ugr) #, pt_obj.id_slc)
                
               
            ### UserEquipment
            # create a UserEquipment, attach to a UserGroup
            # repeat for all UserEquipments in the UserGroup
            for j in range(self.n_ues[i]):
                pt_ugr = self.dc_objs[self.sl_ue_id[i]]    # UserGroup to attach UserEquipment to
                v_pos = [0, 0, 0]           # UserEquipment position vector
                v_vel = [0, 0, 0]           # UserEquipment velocity vector
                usreq = UserEquipment(pt_ugr, v_pos, v_vel,make_tb="TBbyNrRes", debug=debug)    # id UE-1
                #"TBbyNrRes" "OneTBbyRes" "OneTBallRes"
                self.dc_objs[self.sl_ue_id[i] ].ls_usreqs += [usreq]    # add to list in UserGroup
                self.dc_objs[self.id_slc[i]].ls_usreqs += [usreq]    # add to list in Slice
                if debug:
                        print("    ", usreq) #, pt_obj.id_slc)
                ### ChannelEnvironment
                # create a channel environment to which channels will be attached
                        
                ### Channel
                # create Channel, assign ChannelEnvironment to Channel, Channel to UserEquipment
                # repeat for each UserEquipment
                
                loss_prob = 0.0             # channel probability loss
                if self.channel_name == "random or fixed":
                    pt_ch_env = fix_chenv.ChannelEnvironment()
                    usreq.chan = fix_chan.Channel(pt_ch_env, loss_prob,chan_mode=self.chan_mode,val_1=self.val_1,val_2=self.val_2) 
                if self.channel_name == "file": 
                    pt_ch_env = file_chenv.ChannelEnvironment()
                    usreq.chan = file_channel.Channel(pt_ch_env, interpol=self.interpol, debug=True, f_name=self.f_name)
                    usreq.chan.id_usreq = usreq.id_object
                
                
                ### TransportBlock
                # create transport block
                # repeat for each UserEquipment
                usreq.tr_blk = TransportBlock(self.band,self.robustMCS,self.ul_dl,self.mimo_mod,self.nlayers)
                # creates and assigns TransportBlock
                
                
                ### PacketQueue
                # create PacketQueue for UserEquipment, attach to UserEquipment
                # repeat for each UserEquipment in all UserGroups
                qu_prefix="PQ-"     # PacketQueue prefix for unique identifier
                pkt_prefix="Pkt-"   # packet prefix for this queue unique packet identifier
                max_len=0           # maximum queue lenght, then drop packets
                keep_pkts=True      # keep packets after successfully transmitted
                last_k=50           # time to store the data 
                usreq.pktque_dl = PacketQueue(qu_prefix, pkt_prefix, max_len, keep_pkts,last_k)
                
                
                ### TrafficGenerator
                # create TrafficGenerator, attach to UserEquipment data PacketQueue
                # repeat for each PacketQueue in each UserEquipment
                priority = 1        # order of event execution in a certain instant time
                if self.trfgen_type[i] =="fixed":
                    pt_tg_obj = trf_simple.TrafficGenerator(usreq.pktque_dl, self.gen_size[i], \
                            self.gen_delay[i], self.nr_pkts[i], priority, debug)
                if self.trfgen_type[i] =="poisson":
                    pt_tg_obj = trf_poisson.TrafficGenerator(usreq.pktque_dl, gen_size=self.gen_size[i], \
                            gen_delay=self.gen_delay[i], nr_pkts=self.nr_pkts[i], priority=priority,size_dist=self.size_dist[i], debug=debug)        
                # add traffic generator to list of traffic generators
                usreq.trf_gen = pt_tg_obj
                self.ls_trfgen += [pt_tg_obj] # list of all traffic generators
                if debug:
                        print("    ", pt_tg_obj) #, pt_obj.id_slc)
        
            ### assign Resources to Slices
            # the assign matrix assigns all available Resources to Slices
            #    [ Slice id, Resource type, quantity to assign ]
            print("slice ",self.id_slc[i],self.res_type[i], self.nr_items[i],file=self.output_config)
            assgn_mat.append([ self.id_slc[i],self.res_type[i], self.nr_items[i]] )    # Resource assignment matrix
            print(" "*8, pt_ugr,file=self.output_config)
            for ueq in pt_ugr.ls_usreqs:
                print(" "*12, ueq,file=self.output_config)
                print(" "*16, ueq.chan,file=self.output_config)
                print(" "*20, ueq.chan.ch_env,file=self.output_config)
                print(" "*16, ueq.tr_blk,file=self.output_config)
                print(" "*16, ueq.pktque_dl,file=self.output_config)
                print(" "*16, ueq.trf_gen,file=self.output_config)
        
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

        
    def run_simulation(self):
        ### run simulation

        # with open("config.pickle","rb") as fp:
        #     self.cf = pickle.load(fp)
        # self.set_config()
        
        orig_stdout = sys.stdout
        f = open(self.file_log_run, 'w')
        sys.stdout = f
        self.output_run_rec = open(self.file_output_run_rec,'w')
        self.output_run_tr = open(self.file_output_run_tr,'w')
        self.output_run_res = open(self.file_output_run_res,'w')


        print("-------------------simulation starts-------------------")
        dbg_run = True
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
            print("slice ", slc.id_object)
            for ueq in slc.ls_usreqs:
                print("ueq ", ueq.id_object)
                self.output_run_res.write(str(slc.dc_usreq_traf[ueq.id_object]))
                self.output_run_res.write('\n')
                queue = ueq.pktque_dl
                print("queue ",queue)
                #print(queue.dc_traf["Transport Blocks"])
                self.output_run_rec.write(str(queue.last_lst_rec))
                self.output_run_rec.write('\n')
                self.output_run_tr.write(str(queue.last_lst_snt))
                self.output_run_tr.write('\n')

        self.output_run_rec.close()
        self.output_run_tr.close()
        self.output_run_res.close()

        sys.stdout = orig_stdout
        f.close()
        # self.output_config.close()


    def process_data(self):

        with  open(self.file_output_run_rec,'r') as fp:
            ue = 1
            times = [[]]
            bits_rec = [[]]
            bits_drop = [[]]
            for line in fp:
                data = eval(line)
                times.append([]) 
                bits_rec.append([])
                bits_drop.append([])
                for i in range(len(data)):
                    times[ue-1].append(data[i][0])
                    bits_rec[ue-1].append(data[i][1])
                    bits_drop[ue-1].append(data[i][2])
                #self.graph("User Equipment "+str(ue),"Bits recieved",len(data),bits_rec,times)
                #self.graph("User Equipment "+str(ue),"Bits dropped",len(data),bits_drop,times)
                ue = ue +1
            self.graph_subplots(bits_rec,times,"Bits recieved", "Bits")    

        with  open(self.file_output_run_tr,'r') as fp:
            ue = 1
            times = [[]]
            bits_sent = [[]]
            for line in fp:
                data = eval(line)
                times.append([])
                bits_sent.append([]) 
                for i in range(len(data)):
                    times[ue-1].append(data[i][0])
                    bits_sent[ue-1].append(data[i][1])
                #self.graph("User Equipment "+str(ue),"Bits sent",len(data),bits_sent,times)
                #self.graph("User Equipment "+str(ue),"Bits lost",len(data),bits_lost,times)
                #self.graph("User Equipment "+str(ue),"TB(bits)",len(data),tbits,times)
                ue = ue +1   
            self.graph_subplots(bits_sent,times,"Bits sent", "Bits")    
        with  open(self.file_output_run_res,'r') as fp:
            ue = 1
            tbits =[[]]
            resources = [[]]
            times = [[]]
            for line in fp:
                data = eval(line)
                times.append([])
                tbits.append([])
                resources.append([])                
                for i in range(len(data)):
                    tbits[ue-1].append(data[i][2])
                    times[ue-1].append(data[i][0])
                    resources[ue-1].append(data[i][1])
                #self.graph("User Equipment "+str(ue),"Transport Blocks",len(data),tbits[ue-1],times)
                #self.graph("User Equipment "+str(ue),"PRBs",len(data),resources,times)
                #self.graph("User Equipment "+str(ue),"TB(bits)",len(data),tbits,times)
                ue = ue +1   

            self.graph_subplots(tbits,times,"TB bits per TTI", "TB bits")
            self.graph_subplots(resources,times,"Resources per TTI", "PRB")
        plt.show()
                
    def graph(self,title,ylabel,n,param,times):
        """ This method plots one of the short scale parameters (ssp) for each cluster.
        
        @type title: string
        @param title: The title of the plot.
        @type ylabel: string
        @param ylabel: The name of the ssp parameter to plot.
        @type n: int.
        @param n: number of clusters.
        @type param: array.
        @param param: An array with ssp value for each cluster.    
        """ 
        plot2 = plt.figure()
        plt.title(title)
        my_colors = list('rgbkymc')
        plt.bar(times,param,color = my_colors)
        plt.grid(color='r', axis = 'y',linestyle='-', linewidth=2,which = 'major')
    
        plt.xlabel("TTI" )
        plt.ylabel(ylabel )
        plt.draw()
        #plt.show(block=True)

    def graph_subplots(self,data,times,title,ylabel):
        fig, axs = plt.subplots(2, 2)
        fig.suptitle(title)
        my_colors = list('rgbkymc')
        axs[0, 0].bar(times[0],data[0],color = my_colors )
        axs[0, 0].set_title('UE 1')
        axs[0, 1].bar(times[1], data[1],color = my_colors)
        axs[0, 1].set_title('UE 2')
        axs[1, 0].bar(times[2], data[2],color = my_colors)
        axs[1, 0].set_title('UE 3')
        axs[1, 1].bar(times[3], data[3],color = my_colors)
        axs[1, 1].set_title('UE 4')
        
        for ax in axs.flat:
             ax.set(ylabel=ylabel)
        for ax in axs.flat:
            ax.label_outer()
        plt.draw()

if __name__ == "__main__":
    
    
    gf = GuiConfigRunSimulation()
    # gf.config_basenode()  
    # gf.config_channel()
    # gf.config_resources()
    # gf.config_slices()
    # gf.config_trfgen_uegr()
    # gf.config_uegr_slices()
    # gf.config_sched()
    gf.create_simulation_scenary()

    gf.run_simulation()
    gf.process_data()

