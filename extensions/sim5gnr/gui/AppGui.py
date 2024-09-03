#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:05:34 2024


This module is the main Graphical User interface of the Python Wireless Simulator  - 5G extension 

"""
import sys,os

# current_working_directory = os.getcwd()
# print(current_working_directory)
# ###### to run from gui directory
# sys.path.append('../')
# sys.path.append('../../../extensions')
# sys.path.append('../../../libsimnet')
# sys.path.append('../../../models')
# print(sys.path)


### Python and standar libraries imports

import matplotlib.pyplot as plt
from pylab import rcParams

import pickle
import csv

import subprocess
import tkinter as tk
import tkinter.font as tkfont
import tkinter.messagebox
from tkinter import filedialog


### GUI imports
import gui_basestation as gbs
import gui_slice as gsl
import gui_resource as gres
import gui_scheduler as gsched
import gui_channel as gchan
import gui_traffic_generator as gtrgen
import gui_slice_uegr as gsluegr
import gui_user_group as guegr
import gui_uegr_trfgen as guegrtr
import gui_user_message as gum
import gui_ask_options as gao
import gui_select_graph as gsg

##### to run from the 5G project directory and PyWiSim directory
# sys.path.append('../')
# sys.path.append('./src')
# sys.path.append('./src/gui')
# sys.path.append('./src/graph')

rcParams['figure.figsize'] = 11, 6

script_path = os.path.abspath(__file__)
path_list = script_path.split(os.sep)
script_directory = path_list[0:len(path_list)-2] 
directory = '/content/drive/MyDrive/simnet/extensions/sim5gnr'

file_config = "/home/lingles/ownCloud/Summer_school/simnet/extensions/sim5gnr/data/config.pickle"
file_log_config = "/home/lingles/ownCloud/Summer_school/simnet/extensions/sim5gnr/data/config_log.txt"
file_log_run = "/home/lingles/ownCloud/Summer_school/simnet/extensions/sim5gnr/data/run_log.txt"
output_run_rec = "/home/lingles/ownCloud/Summer_school/simnet/extensions/sim5gnr/data/run_results_rec.txt"
output_run_tr = "/home/lingles/ownCloud/Summer_school/simnet/extensions/sim5gnr/data/run_results_tr.txt"
output_run_res = "/home/lingles/ownCloud/Summer_school/simnet/extensions/sim5gnr/data/run_results_res.txt"
output_run_ch_st = "/home/lingles/ownCloud/Summer_school/simnet/extensions/sim5gnr/data/run_results_ch_st.txt"
output_run_queue = "/home/lingles/ownCloud/Summer_school/simnet/extensions/sim5gnr/data/run_results_queue.txt"
output_run_names = "/home/lingles/ownCloud/Summer_school/simnet/extensions/sim5gnr/data/run_results_names.txt"


class ConfigScenary():
    ''' The Scenary configuration.
    This class has all properties to set the Scenary
    '''
    
    def __init__(self):
        """The constructor of the Scenary Configuration     
        """ 

        
        
        ####Base Station configuration
        self.num_slices = 1
        self.mimo = "SU"
        self.bs_name = "BS-1"
        self.name_sched = "round robin"
        self.rMCS = False
        self.longcp = False
        self.band ="n258"
        self.ul = False
        self.dl = True
        self.time_sim = 100
    
        #######Slice Configuration. Default one slice
        self.name_slice = ["SL-1"]
        self.numerology = [0]
     
        #####Resources configuration. Resources for each slice
        self.sym_slot = [14]
        self.nresdl = [100]
        self.nresul = [0]
        self.name_resdl = ["PRB"]
        self.name_resul = ["PRB"]
        
        ###Slice-UEGroup association
        self.slice_uegr = [[0]]
        self.ugr_not_assigned = []
        
        ##### Channel configuration
        self.channel_type="random or fixed"
        self.file_channel = None
        self.channel_mode ="Random"
        self.val_1=-10
        self.val_2=100
        self.loss_prob = 0
        
        ######Traffic Gnerator- One TrfGen for each UEGroup
        self.trgen_type = ["poisson"]
        self.inter_arrival = [1]
        self.pkt_size = [300]
        self.burst_size = [1]
        self.size_dist = ["Exponential"]
        
        self.max_len = [0]       # maximum queue lenght, then drop packets
        self.keep_pkts = [False] 
        self.last_k = [100]          # time to store the data 

    
        #### User Groups Configuration
        self.nuegroups = 1
        self.name_uegroup = ["UG-1"]
        self.uegr_par1 = [60]
        self.num_ues = [5]

       
    

class AppGui():
    """ This class is the main form of the Python wireless csimulator.
    
    """
    def __init__(self,window,title):
        """The constructor of the AppGui
        
        @type window: tkinter.Tk() window.
        @param window: The main window of this form.
        @type title: string
        @param title: The title of the form.

        """ 
        self.__window  = window
        """ The main window of this form. """ 
        self.__window.title(title)
        """ The title of the main window. """
        self.conf = ConfigScenary()
        """ The default scenary configuration"""
        self.__window_txa = None
        self.root = None
        self.gui_configuration()
        """ Defines the configuration of the main window """

    def gui_configuration(self):
        """This method builds the main form to enter the scenary configuration.
 
            This method defines the characteristics of the main window  and
            builds the buttons for each object  configuration, 
            It also, allows to save and restore an scenary,  to run the simulation, and to analyze the results.
         
        """         
        width, height = window.winfo_screenwidth(), window.winfo_screenheight()
        self.__FONT_SIZE = int(width /1.25/ 1920*25)
        #window.geometry('%dx%d+0+0' % (width/1.1,height/1.1))

        self.__window.rowconfigure((0,1,2,3,4,5,6,7), weight=1)  # make buttons stretch when

        self.__window.columnconfigure((0,1,2,3,4), weight=1)# columnconfigure([0, 1, 2, 3, 4, 5], minsize=50, weight=1) 
             
        font = tkfont.Font(family="Helvetica", size=self.__FONT_SIZE, weight = tkfont.BOLD)
        square_size = int(font.metrics('linespace')/1.)
        
        lbl_config = tk.Label(self.__window, text="Configuration of the \n simulation scenario: ",fg = "dark blue",font = font)#Verdana 18 bold")
        lbl_config.grid(row=0, column=1, columnspan=2, sticky='EWNS')

       
        aux0 = tk.Button(self.__window, text="RadioBase",font=font, compound=tk.CENTER, command=self.cmd_rb_config)
        aux0.grid(row=2, column=0, columnspan=1, sticky='EWNS') #padx=10)
        aux0.config(width=int(square_size*1.1), height=int(square_size/2.5))

        aux0 = tk.Button(self.__window, text="Slices", font=font, compound=tk.CENTER,command=self.cmd_slices)
        aux0.grid(row=2, column=1, columnspan=1, sticky='EWNS')
        aux0.config(width=int(square_size*1.1), height=int(square_size/2.5))
        
        aux0 = tk.Button(self.__window, text="User Groups", font=font, compound=tk.CENTER,command=self.cmd_uegroups)
        aux0.grid(row=2, column=2, columnspan=1, sticky='EWNS')
        aux0.config(width=int(square_size*1.1), height=int(square_size/2.5))


        aux0 = tk.Button(self.__window, text="Resources",font=font, compound=tk.CENTER, command=self.cmd_resources)
        aux0.grid(row=2, column=3, columnspan=1, sticky='EWNS')
        aux0.config(width=int(square_size*1.1), height=int(square_size/2.5))
      
        aux0 = tk.Button(self.__window, text="Scheduler", font=font, compound=tk.CENTER,command=self.cmd_sched)
        aux0.grid(row=3, column=0, columnspan=1, sticky='EWNS')
        aux0.config(width=int(square_size*1.1), height=int(square_size/2.5))
        
        aux0 = tk.Button(self.__window, text="Channel", font=font, compound=tk.CENTER,command=self.cmd_channel)
        aux0.grid(row=3, column=1, columnspan=1, sticky='EWNS')
        aux0.config(width=int(square_size*1.1), height=int(square_size/2.5))

        # aux0 = tk.Button(self.__window, text="Slice - UE Group \nassociation", font=font, compound=tk.CENTER,command=self.cmd_asoc_slice_uegroup)
        # aux0.grid(row=3, column=2, columnspan=1, sticky='EWNS')
        # aux0.config(width=int(square_size*1.1), height=int(square_size/2.5))
 
        aux0 = tk.Button(self.__window, text="Traffic Generators", font=font, compound=tk.CENTER,command=self.cmd_asoc_uegroup_trfgen)
        aux0.grid(row=3, column=2, columnspan=1, sticky='EWNS')
        aux0.config(width=int(square_size*1.1), height=int(square_size/2.5))

        aux0 = tk.Button(self.__window, text="Save Scenary", font=font, compound=tk.CENTER,command=self.cmd_save)
        aux0.grid(row=4, column=0, columnspan=1, sticky='EWNS')
        aux0.config(width=int(square_size*1.1), height=int(square_size/2.5))

        aux0 = tk.Button(self.__window, text="Load Scenary", font=font, compound=tk.CENTER,command=self.cmd_load)
        aux0.grid(row=4, column=1, columnspan=1, sticky='EWNS')
        aux0.config(width=int(square_size*1.1), height=int(square_size/2.5))

        aux0 = tk.Button(self.__window, text="View scenary", font=font, compound=tk.CENTER,command=self.cmd_view)
        aux0.grid(row=4, column=2,columnspan=1, sticky='EWNS')
        aux0.config(width=int(square_size*1.1), height=int(square_size/2.5))


        frm_runsim = tk.Frame(master=self.__window)
        lbl_runsim = tk.Label(master=frm_runsim, text="Run the simulation : ",fg = "dark blue",font = font)
        lbl_runsim.grid(row=0, column=0,columnspan=2, sticky='EWNS')
        frm_runsim.grid(row=5, column=0, padx=10)
        
        aux0 = tk.Button(self.__window, text="Run", font=font, compound=tk.CENTER,command=self.cmd_run)
        aux0.grid(row=6, column=0,columnspan=1, sticky='EWNS')
        aux0.config(width=int(square_size*1.1), height=int(square_size/2.5))
        
        aux0 = tk.Button(self.__window, text="Graph Results ", font=font, compound=tk.CENTER,command=self.cmd_graph)
        aux0.grid(row=6, column=1,columnspan=1, sticky='EWNS')
        aux0.config(width=int(square_size*1.1), height=int(square_size/2.5))
        
 
    ##### comand functions. Each of the following functions are called when
    ##### the user press the corresponding button.
        
    def cmd_run(self):
        """This method runs the simulation calling the PyWiSim mk_setup file.

            First save the current scenary configuration and call the mk_setup.py
            This methods assumes that the simulator GUI is runing from the Project Directory.
        """  
        if self.__window_txa is not None:
            gum.AppUserMsg('Error', 'Another configuration window is open, please close it' )
            return

        self.__window_txa = tk.Tk()
        app = gao.AppAskOptions(self.__window_txa,self.function_run,title="Debug information",ok_title="OK",data_description="Store and show debug information?",list_val=["True","False"],val_defualt= "False")
        self.__window_txa.mainloop()

    def function_run(self,debug):
        self.__window_txa.destroy()
        self.__window_txa = None
        if  os.path.isfile(output_run_rec):
            os.remove(output_run_rec)### delete the recieve data file to be sure 
        if  os.path.isfile(output_run_tr):
            os.remove(output_run_tr)### delete the transmit data file to be sure 
        if  os.path.isfile(output_run_res):
            os.remove(output_run_res)### delete the resources data file to be sure 
        if  os.path.isfile(output_run_ch_st):
            os.remove(output_run_ch_st)### delete the resources data file to be sure 
        if  os.path.isfile(output_run_queue):
            os.remove(output_run_queue)### delete the resources data file to be sure 
        if  os.path.isfile(output_run_names):
            os.remove(output_run_names)### delete the resources data file to be sure 

        with open(file_config,"wb") as fp:
            pickle.dump(self.conf,fp) # save the current configuration

        file_conf = file_config # name of the configuration file
        run = "True" # if run the simulation or not
        proc = subprocess.Popen(['python3', "mk_simsetup.py", file_conf,run,debug],stdout=subprocess.PIPE)
        try:
            outs, errs = proc.communicate(timeout=600)
            #tk.messagebox.showinfo(message="PyWinSim Simulation ends ")
        except subprocess.TimeoutExpired:
            proc.kill()
            tk.messagebox.showinfo(message="PyWinSim Simulation timeout")

        try:    
            with  open(file_log_run,'r') as fp:
                data = fp.read()
            root = tk.Tk()
            root.title("PyWiSim run debug log")
            text_widget = tk.Text(root, wrap="word", width=80, height=100)
            text_widget.pack(pady=10)
            text_widget.delete(1.0, tk.END)  # Clear previous content
            text_widget.insert(tk.END, data)
        except Exception as error:
            gum.AppUserMsg('Exception occurred!','{}'.format(error)+ " try again please!" )

    def process_data(self):
        """ This method loads the files with the results of the simulation and buids arrays with them
        """
        with  open(output_run_rec,'r') as fp:
            ue = 1
            self.times_rec = [[]]
            self.bits_rec = [[]]
            self.bits_drop = [[]]
            for line in fp:
                data = eval(line)
                self.times_rec.append([]) 
                self.bits_rec.append([])
                self.bits_drop.append([])
                for i in range(len(data)):
                    self.times_rec[ue-1].append(data[i][0])
                    self.bits_rec[ue-1].append(data[i][1])
                    self.bits_drop[ue-1].append(data[i][2])
                #self.graph("User Equipment "+str(ue),"Bits recieved",len(data),bits_rec,times)
                #self.graph("User Equipment "+str(ue),"Bits dropped",len(data),bits_drop,times)
                ue = ue +1
            #self.graph_subplots(bits_rec,times,"Bits recieved", "Bits")    

        with  open(output_run_tr,'r') as fp:
            ue = 1
            self.times_tr = [[]]
            self.bits_sent = [[]]
            self.bits_lost = [[]]
            for line in fp:
                data = eval(line)
                self.times_tr.append([])
                self.bits_sent.append([]) 
                self.bits_lost.append([]) 
                for i in range(len(data)):
                    self.times_tr[ue-1].append(data[i][0])
                    self.bits_sent[ue-1].append(data[i][1])
                    self.bits_lost[ue-1].append(data[i][2])

                #self.graph("User Equipment "+str(ue),"Bits sent",len(data),bits_sent,times)
                #self.graph("User Equipment "+str(ue),"Bits lost",len(data),bits_lost,times)
                #self.graph("User Equipment "+str(ue),"TB(bits)",len(data),tbits,times)
                ue = ue +1   
            #self.graph_subplots(bits_sent,times,"Bits sent", "Bits")    
        with  open(output_run_res,'r') as fp:
            ue = 1
            self.tbits =[[]]
            self.resources = [[]]
            self.times_res = [[]]
            for line in fp:
                data = eval(line)
                self.times_res.append([])
                self.tbits.append([])
                self.resources.append([])                
                for i in range(len(data)):
                    self.tbits[ue-1].append(data[i][2])
                    self.times_res[ue-1].append(data[i][0])
                    self.resources[ue-1].append(data[i][1])
                #self.graph("User Equipment "+str(ue),"Transport Blocks",len(data),tbits[ue-1],times)
                #self.graph("User Equipment "+str(ue),"PRBs",len(data),resources,times)
                #self.graph("User Equipment "+str(ue),"TB(bits)",len(data),tbits,times)
                ue = ue +1   

        with  open(output_run_queue,'r') as fp:
            ue = 1
            self.bits =[[]]
            self.pkts = [[]]
            self.delay = [[]]
            self.times_delay = [[]]
            for line in fp:
                data = eval(line)
                self.times_delay.append([])
                self.bits.append([])
                self.pkts.append([])  
                self.delay.append([])                
                for i in range(len(data)):
                    self.bits[ue-1].append(data[i][1])
                    self.times_delay[ue-1].append(data[i][0])
                    self.pkts[ue-1].append(data[i][2])
                    self.delay[ue-1].append(data[i][3])

                #self.graph("User Equipment "+str(ue),"Transport Blocks",len(data),tbits[ue-1],times)
                #self.graph("User Equipment "+str(ue),"PRBs",len(data),resources,times)
                #self.graph("User Equipment "+str(ue),"TB(bits)",len(data),tbits,times)
                ue = ue +1   



        with  open(output_run_ch_st,'r') as fp:
            ue = 1
            self.chst =[[]]
            self.times_ch = [[]]
            for line in fp:
                data = eval(line)
                self.times_ch.append([])
                self.chst.append([])
                for i in range(len(data)):
                    self.chst[ue-1].append(data[i][1])
                    self.times_ch[ue-1].append(data[i][0])
                #self.graph("User Equipment "+str(ue),"Transport Blocks",len(data),tbits[ue-1],times)
                #self.graph("User Equipment "+str(ue),"PRBs",len(data),resources,times)
                #self.graph("User Equipment "+str(ue),"TB(bits)",len(data),tbits,times)
                ue = ue +1   


            #self.graph_subplots(tbits,times,"TB bits per TTI", "TB bits")
            #self.graph_subplots(resources,times,"Resources per TTI", "PRB")
        #plt.show()
        self.ues = ue
        with  open(output_run_names,'r') as fp:
            ue = 1
            self.l_names =[]
            for line in fp:
                self.l_names.append(line)
                ue = ue +1   


    def mk_dc_positions(self, f_name):
        '''Makes a dictionary of channel states by user equipment.

        @param f_name: file name of a CSV file with channel states.
        '''
    
        dc_usreq_x = {}
        dc_usreq_y = {}
        fp = open(f_name, "r", encoding='utf8')
        reader = csv.reader(fp, delimiter=';', quotechar='"', \
            quoting=csv.QUOTE_ALL, lineterminator='\n')
            
        for row in reader:
            if row[0] in dc_usreq_x:
                aux = [float(ele) for ele in row[2].split()]
                dc_usreq_x[row[0]] = dc_usreq_x[row[0]] + [aux[0]]
                dc_usreq_y[row[0]] = dc_usreq_y[row[0]] + [aux[1]]
            else:
                aux = [float(ele) for ele in row[2].split()]
                dc_usreq_x[row[0]] =  [aux[0]]
                dc_usreq_y[row[0]] =  [aux[1]]

        fp.close()
        plt.figure()

        for i in range(len(dc_usreq_x)):
            plt.plot(dc_usreq_x["UE-"+str(i+1)], dc_usreq_y["UE-"+str(i+1)], marker='o', label='UE-'+str(i+1))  # Trayectoria del móvil 1
        plt.scatter(0, 0, color='black', marker='X',label='Base Station')

        plt.title('Mobiles trajectories')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.legend()  # Mostrar leyenda
        plt.grid(True)  # Activar la cuadrícula
        
        # Mostrar la gráfica
        plt.show()
        
        return dc_usreq_x,dc_usreq_y

        
    def cmd_graph(self):
        """This method graph the simulation results.
        
           The sim_setup.py saves the results to three files in sim5gnr/data directory. 
           This method opens these three files and plot the data. The files with the simulation results are
           set by default in the ConfigScenary class. By default are run_results_rec.txt, run_results_tr.txt and run_results_res.txt.
           The first one has information about the packets generated, and dropped of the queue. The second file has the 
           information about the bits sent and the bits lost. And the third ones has information of the 
           resources assigned an the transport blocks gnerated in each TTI.
        """   
        """ This method is called when the user selects the button to configure the BaseStation. 
        
        """ 
        self.process_data()

        self.__window_txa = tk.Tk()
        app = gsg.AppSelectGraph(self.__window_txa,self.function_graph,"Select users","OK",self.l_names)
        self.__window_txa.mainloop()

                  

    def ask_savefilenme(self):
        """ This method is called when the user saves a simulation scenary asking for the  the file name and directory. 
        
        """ 
        self.file_name = ""
        try:
            file_name = filedialog.asksaveasfilename(initialdir="./extensions/sim5gnr/demo",title='Please select the file and directory',defaultextension=".pickle", filetypes=(("pickle file", "*.pickle"),("All Files", "*.*") ))

            if file_name is not None and file_name != "" :
                self.file_name = file_name

            else:
                gum.AppUserMsg('Error!','You must select a Directory. Please try again')

        except BaseException as error:
            gum.AppUserMsg('Exception occurred!','{}'.format(error)+ " try again please!" )
    
    def ask_openfilename(self):
        """ This method is called when the user opens a simulation scenary asking for the  the file name and directory. 
                
        """ 
        self.file_name = ""
        try:
            file_name = filedialog.askopenfilename(initialdir="./extensions/sim5gnr/demo",title='Please select the file and directory',defaultextension=".pickle", filetypes=(("pickle file", "*.pickle"),("All Files", "*.*") ))

            if file_name is not None and file_name != "" :
                self.file_name = file_name
            else:
                gum.AppUserMsg('Error!','You must select a Directory. Please try again')

        except BaseException as error:
            gum.AppUserMsg('Exception occurred!','{}'.format(error)+ " try again please!" )
    

    def cmd_view(self):
        """This method run the mk_simsetup.py to view the scenary but it does not run the simulation.
        
           
        """
        #print("cmd_view")
        if  os.path.isfile(file_log_config):
            os.remove(file_log_config)### delete the log to be sure 
        with open(file_config,"wb") as fp:
            pickle.dump(self.conf,fp)

        arg1 = file_config # name of the configuration file
        run = "False" # if run the simulation or not
        debug = "True"
        proc = subprocess.Popen(['python3', "/content/drive/MyDrive/simnet/extensions/sim5gnr/gui/mk_simsetup.py", arg1,run,debug],stdout=subprocess.PIPE)
        try:
            outs, errs = proc.communicate(timeout=300)
            #tk.messagebox.showinfo(message="PyWinSim Simulation ends ")
        except subprocess.TimeoutExpired:
            proc.kill()
            tk.messagebox.showinfo(message="PyWinSim Simulation timeout")
        try:    
            with  open(file_log_config,'r') as fp:
                data = fp.read()
                
                if self.root is not None:
                       self.root.lift()
                       self.root.attributes('-topmost', True)
                       self.root.attributes('-topmost', False)
                       self.root.focus_set()
                       gum.AppUserMsg('Error', 'Another log window is open, please close it' )

                       return      
                self.root = tk.Tk()
                self.root.protocol("WM_DELETE_WINDOW", self.root_exit)
                self.root.title("PyWiSim Scenary Configuration")
                text_widget = tk.Text(self.root, wrap="word", width=80, height=100)
                text_widget.pack(pady=10)
                text_widget.delete(1.0, tk.END)  # Clear previous content
                text_widget.insert(tk.END, data)
        except Exception as error:
            gum.AppUserMsg('Exception occurred!','{}'.format(error)+ " try again please!" )
  
    def root_exit(self):
        self.root.destroy()
        self.root = None

    
    
    def cmd_save(self):  
        """ This method is called when the user saves a simulation scenary . 
        
        """        

        self.ask_savefilenme()
        if self.file_name is not None and self.file_name != "" :
            with open(self.file_name,"wb") as fp:
                pickle.dump(self.conf,fp)
    
    def cmd_load(self):
        """ This method is called when the user opens a simulation scenary . 
        
        """ 
        self.ask_openfilename()
        if self.file_name is not None and self.file_name != "" :
            with open(self.file_name,"rb") as fp:
                self.conf = pickle.load(fp)

    def window_exit(self):
        close = tk.messagebox.askyesno("Exit?", "Are you sure you want to exit without save?")
        if close:
            self.__window_txa.destroy()
            self.__window_txa = None
           
    def cmd_rb_config(self):
    
        """ This method is called when the user selects the button to configure the BaseStation. 
        
        """ 
        if self.__window_txa is not None:
            gum.AppUserMsg('Error', 'Another configuration window is open, please close it' )
            return
        self.__window_txa = tk.Tk()
        app = gbs.AppBaseStation(self.__window_txa,self.function_basestation,"BaseStation Specification",self.conf.bs_name,self.conf.band ,self.conf.rMCS,self.conf.longcp,self.conf.ul,self.conf.dl,self.conf.mimo,self.conf.nuegroups,self.conf.num_slices,self.conf.time_sim)
        self.__window_txa.protocol("WM_DELETE_WINDOW", self.window_exit)
        self.__window_txa.mainloop()

    def cmd_uegroups(self):
        """ This method is called when the user selects the button to configure the User Groups. 
        
        """ 
        if self.__window_txa is not None:
            gum.AppUserMsg('Error', 'Another configuration window is open, please close it' )
            return
        
        self.__window_txa = tk.Tk()
        app = guegr.AppUserGroup(self.__window_txa,self.function_uegroups,"User Group Specification",self.conf.name_uegroup[0],0,self.conf.uegr_par1[0],self.conf.num_ues[0])
        self.__window_txa.protocol("WM_DELETE_WINDOW", self.window_exit)
        self.__window_txa.mainloop()

    def cmd_sched(self):
        """ This method is called when the user selects the button to configure the scheduler. 
        
        """ 
        if self.__window_txa is not None:
            gum.AppUserMsg('Error', 'Another configuration window is open, please close it' )
            return

        self.__window_txa = tk.Tk()
        app = gsched.AppScheduler(self.__window_txa,self.function_sched,"Scheduler Specification",name =self.conf.name_sched)
        self.__window_txa.protocol("WM_DELETE_WINDOW", self.window_exit)
        self.__window_txa.mainloop()

    def cmd_resources(self):
        """ This method is called when the user selects the button to configure the resources. 
        
        """ 
        if self.__window_txa is not None:
            gum.AppUserMsg('Error', 'Another configuration window is open, please close it' )
            return

        self.__window_txa = tk.Tk()
        app = gres.AppResource(self.__window_txa,self.function_resource,"Resources Specification",self.conf.name_resdl[0],self.conf.name_resul[0],self.conf.nresdl[0],self.conf.nresul[0],self.conf.sym_slot[0],self.conf.ul,self.conf.dl,self.conf.band,0)
        self.__window_txa.protocol("WM_DELETE_WINDOW", self.window_exit)
        self.__window_txa.mainloop()

    def cmd_slices(self):
        """ This method is called when the user selects the button to configure the slices. 
        
        """ 
        if self.__window_txa is not None:
            gum.AppUserMsg('Error', 'Another configuration window is open, please close it' )
            return

        self.__window_txa = tk.Tk()
        app = gsl.AppSlice(self.__window_txa,self.function_slice,"Slice Specification",self.conf.name_slice[0],0,self.conf.numerology,self.conf.slice_uegr,self.conf.name_uegroup,self.conf.num_slices,self.conf.nuegroups,self.conf.ugr_not_assigned)
        self.__window_txa.protocol("WM_DELETE_WINDOW", self.window_exit)
        self.__window_txa.mainloop()

    # def cmd_trfgen(self):
    #     """ This method is called when the user select to configure the Tx antenna. 
        
    #     """ 
    #     if self.__window_txa is not None:
    #         gum.AppUserMsg('Error', 'Another configuration window is open, please close it' )
    #         return

    #     self.__window_txa = tk.Tk()
    #     app = gtrgen.AppTrGen(self.__window_txa,self.function_trgen,"Traffic Generator Specification",name=self.conf.trgen_type,inter_arrival =self.conf.inter_arrival, pkt_size=self.conf.pkt_size, size_dist = self.conf.size_dist, pkt_burst=self.conf.burst_size[0],dl = self.conf.dl,ul =self.conf.ul,keep_pkts=self.conf.keep_pkts[0],max_len=self.conf.max_len,last_k = self.conf.last_k ) 
    #     self.__window_txa.protocol("WM_DELETE_WINDOW", self.window_exit)
    #     self.__window_txa.mainloop()

    def cmd_channel(self):
        """ This method is called when the user selects the button to configure the channel. 
        
        """ 
        if self.__window_txa is not None:
            gum.AppUserMsg('Error', 'Another configuration window is open, please close it' )
            return

        self.__window_txa = tk.Tk()
        app = gchan.AppChannel(self.__window_txa,self.function_channel,"Channel Specification",self.conf.channel_type,self.conf.file_channel,self.conf.channel_mode,self.conf.val_1, self.conf.val_2,self.conf.loss_prob ) 
        self.__window_txa.protocol("WM_DELETE_WINDOW", self.window_exit)
        self.__window_txa.mainloop()

    # def cmd_asoc_slice_uegroup(self):
    #     """ This method is called when the user selects the button to configure the slices-usergroups association. 
        
    #     """ 
    #     self.__window_txa = tk.Tk()
    #     app = gsluegr.AppSliceUegr(self.__window_txa,self.function_slice_uegr,"Slice-UEGroups association",0,self.conf.slice_uegr[0],self.conf.name_uegroup ) 
    #     self.__window_txa.mainloop()
        
    def cmd_asoc_uegroup_trfgen(self):
        """ This method is called when the user selects the button to configure the user group - traffic gnerator association. 
        
        """ 
        if self.__window_txa is not None:
            gum.AppUserMsg('Error', 'Another configuration window is open, please close it' )
            return

        self.__window_txa = tk.Tk()
        app = gtrgen.AppTrGen(self.__window_txa,self.function_uegr_trfgen,"Traffic Generator Specification",name =self.conf.trgen_type,uegr=0,inter_arrival =self.conf.inter_arrival, pkt_size=self.conf.pkt_size, size_dist = self.conf.size_dist, burst_size=self.conf.burst_size,nugr=self.conf.nuegroups,dl = self.conf.dl,ul =self.conf.ul,keep_pkts=self.conf.keep_pkts,max_len=self.conf.max_len,last_k = self.conf.last_k ) 
        self.__window_txa.protocol("WM_DELETE_WINDOW", self.window_exit)
        self.__window_txa.mainloop()

        # self.__window_txa = tk.Tk()
        # app = guegrtr.AppUegrTrfgen(self.__window_txa,self.function_uegr_trfgen,"UEGroup-Traffic Generator association",self.conf.trgen_type[0],0,inter_arrival =self.conf.inter_arrival[0], pkt_size=self.conf.pkt_size[0], size_dist = self.conf.size_dist[0], burst_size=self.conf.burst_size[0],dl = self.conf.dl,ul =self.conf.ul ) 
        # self.__window_txa.mainloop()


    #########################################################
    ####################callback functions###################
    # These functions are called when the configuration forms 
    #are closed.    
    #########################################################

    def function_graph(self,l_users_sel,l_graphs):
        """ This is the callback function of the AppScheduler to configure the scheduler.

        @type name: string.
        @param name: The type of scheduler.
                """ 
        
        self.__window_txa.destroy()
        self.__window_txa = None
        if 0 in l_graphs:
            bits_rec_sel = [self.bits_rec[i] for i in l_users_sel]
            bits_drop_sel = [self.bits_drop[i] for i in l_users_sel]
            times_sel = [self.times_rec[i] for i in l_users_sel]
            self.graph_subplots(bits_rec_sel,times_sel,"Bits received", "bits",l_users_sel)
            self.graph_subplots(bits_drop_sel,times_sel,"Bits dropped", "bits",l_users_sel)
 
        if 1 in l_graphs:
            bits_sent_sel = [self.bits_sent[i] for i in l_users_sel]
            bits_lost_sel = [self.bits_lost[i] for i in l_users_sel]
            times_sel = [self.times_tr[i] for i in l_users_sel]
            self.graph_subplots(bits_sent_sel,times_sel,"Bits sent", "bits",l_users_sel)
            self.graph_subplots(bits_lost_sel,times_sel,"Bits lost", "bits",l_users_sel)

        if 2 in l_graphs:
            t_bits_sel = [self.tbits[i] for i in l_users_sel]
            resources_sel = [self.resources[i] for i in l_users_sel]
            times_sel = [self.times_res[i] for i in l_users_sel]
            self.graph_subplots(t_bits_sel,times_sel,"TB bits per TTI", "TB bits",l_users_sel)
            self.graph_subplots(resources_sel,times_sel,"Resources per TTI", "PRB",l_users_sel)

        if 3 in l_graphs:
            ch_sel = [self.chst[i] for i in l_users_sel]
            times_sel = [self.times_ch[i] for i in l_users_sel]
            self.graph_subplots(ch_sel,times_sel,"SNR", "db",l_users_sel)
            if self.conf.channel_type=="file":
                f_name = self.conf.file_channel.replace('snr','positions')
                self.mk_dc_positions(f_name)
        if 4 in l_graphs:
            bits_sel = [self.bits[i] for i in l_users_sel]
            pkts_sel = [self.pkts[i] for i in l_users_sel]
            delay_sel = [self.delay[i] for i in l_users_sel]  
            times_sel = [self.times_delay[i] for i in l_users_sel]
            self.graph_subplots(bits_sel,times_sel,"Bits in queue per TTI", "bits",l_users_sel)
            self.graph_subplots(pkts_sel,times_sel,"Packets in queue per TTI", "pkts",l_users_sel)
            self.graph_subplots(delay_sel,times_sel,"Average packet delay in queue per TTI", "ms",l_users_sel)
            
    
        plt.show()

   
    def graph_subplots(self,data,times,title,ylabel,users):
        """ This method plots diferents users performance data as a function of the time.
        
        @type data: array.
        @param data: An array with the users performance data. 
        @type times: array.
        @param times: An array with the time values.
        @type title: string
        @param title: The title of the plot.
        @type ylabel: string
        @param ylabel: The name of the data performance index.
        @type users: list.
        @param users: the list of users to plot.

        """ 
        n=len(data)
        if n >= 4:
            fig, axs = plt.subplots(2, 2)
            fig.suptitle(title)
            my_colors = list('rgbkymc')
            axs[0, 0].bar(times[0],data[0],color = my_colors )
            axs[0, 0].set_title('UE-'+str(users[0]+1))
            axs[0, 1].bar(times[1], data[1],color = my_colors)
            axs[0, 1].set_title('UE-'+str(users[1]+1))
            axs[1, 0].bar(times[2], data[2],color = my_colors)
            axs[1, 0].set_title('UE-'+str(users[2]+1))
            axs[1, 1].bar(times[3], data[3],color = my_colors)
            axs[1, 1].set_title('UE-'+str(users[3]+1))
        if n == 3:
            fig, axs = plt.subplots(3, 1)
            fig.suptitle(title)
            my_colors = list('rgbkymc')
            axs[0].bar(times[0],data[0],color = my_colors )
            axs[0].set_title('UE-'+str(users[0]+1))
            axs[1].bar(times[1], data[1],color = my_colors)
            axs[1].set_title('UE-'+str(users[1]+1))
            axs[2].bar(times[2], data[2],color = my_colors)
            axs[2].set_title('UE-'+str(users[2]+1))
        if n == 2:
            fig, axs = plt.subplots(2, 1)
            fig.suptitle(title)
            my_colors = list('rgbkymc')
            axs[0].bar(times[0],data[0],color = my_colors )
            axs[0].set_title('UE-'+str(users[0]+1))
            axs[1].bar(times[1], data[1],color = my_colors)
            axs[1].set_title('UE-'+str(users[1]+1))
        if n == 1:
            fig, axs = plt.subplots(1, 1)
            fig.suptitle(title)
            my_colors = list('rgbkymc')
            axs.bar(times[0],data[0],color = my_colors )
            axs.set_title('UE-'+str(users[0]+1))
       
        # if n>1:   
            # for ax in axs.flat:
            #      ax.set(ylabel=ylabel)
            # for ax in axs.flat:
            #     ax.label_outer()
        plt.draw()


     
        
    def function_basestation(self,mimo,num_slices,band,name,rMCS,longcp,ul,dl,time_sim):
        """ This is the callback function of the AppBaseStation to configure the Base node.

        @type mimo: "MU" or "SU".
        @param mimo: single user o multiuser mimo.
        @type num_slices: int.
        @param num_slices: th number of slices in the base node.
        @type band: string
        @param band: The operation frequency band of the system.
        @type name: string.
        @param nae: The the name of the base node.      
        @type rMCS: Boolean.
        @param rMCS: Roubust Modulation and code scheme or not. 
        @type longcp: Boolean.
        @param longcp: Long cyclic prefix or not. 
        @type ul: Boolean.
        @param ul: If the simulation is for downlink or not. 
        @type dl: Boolean.
        @param dl: If the simulation is for uplink or not. 
        @type time_sim int.
        @param time_sim: simulation time.

          
        """ 
        self.conf.mimo = mimo
        
        self.conf.bs_name = name
        self.conf.band = band
        self.conf.rMCS = eval(rMCS)
        self.conf.longcp = eval(longcp)
        self.conf.ul =eval(ul)
        self.conf.dl = eval(dl)
        self.conf.time_sim = eval(time_sim)
        # if self.conf.nuegroups > int(nuegroups):
        #     # if the number of uegroups is reduced, reset the values
        #     self.conf.name_uegroup = []
        #     self.conf.uegr_par1 = []
        #     self.conf.num_ues =[]
        #     self.conf.trgen_type = []
        #     self.conf.inter_arrival = []
        #     self.conf.pkt_size = []
        #     self.conf.burst_size = []
        #     self.conf.size_dist = []
        #     r1 = 0
        #     r2 = int(nuegroups)
        # else:
        #     r1 =self.conf.nuegroups
        #     r2 = int(nuegroups)
        # self.conf.nuegroups = int(nuegroups)
        # for i in range(r1,r2):
        #     self.conf.name_uegroup.append("UG-"+str(i+1))
        #     self.conf.uegr_par1.append(50)
        #     self.conf.num_ues.append(5)
        #     self.conf.trgen_type.append("poisson")
        #     self.conf.inter_arrival.append(1)
        #     self.conf.pkt_size.append(300)
        #     self.conf.burst_size.append(1)
        #     self.conf.size_dist.append ("Exponential")
        
        # if self.conf.num_slices>int(num_slices):       
        self.conf.name_slice =[]
        self.conf.numerology = []
        self.conf.sym_slot = []
        self.conf.nresdl = []
        self.conf.nresul = []
        self.conf.name_resdl = []
        self.conf.name_resul = []
        self.conf.slice_uegr = [[]]
        for i in range(int(num_slices)-1):
          self.conf.slice_uegr.append([])  
        self.conf.name_uegroup = []
        self.conf.uegr_par1 = []
        self.conf.num_ues =[]
        self.conf.trgen_type = []
        self.conf.inter_arrival = []
        self.conf.pkt_size = []
        self.conf.burst_size = []
        self.conf.size_dist = []
        self.conf.ugr_not_assigned=[]
        self.conf.max_len = []
        self.conf.keep_pkts = []
        self.conf.last_k = []
        r1 = 0
        r2 = int(num_slices)
        # else:
        #     r1 =self.conf.num_slices
        #     r2 = int(num_slices)
 
        self.conf.num_slices = int(num_slices)
        self.conf.nuegroups = int(num_slices)
        for i in range(r1,r2):
            self.conf.name_slice.append("SL-"+str(i+1))
            self.conf.numerology.append(0)
            self.conf.sym_slot.append(14)
            self.conf.nresdl.append(100)
            self.conf.nresul.append(0)
            self.conf.name_resdl.append("PRB")
            self.conf.name_resul.append("PRB")
            #self.conf.slice_uegr.append([])
            self.conf.slice_uegr[i].append(i)
            self.conf.name_uegroup.append("UG-"+str(i+1))
            self.conf.uegr_par1.append(50)
            self.conf.num_ues.append(5)
            self.conf.trgen_type.append("poisson")
            self.conf.inter_arrival.append(1)
            self.conf.pkt_size.append(300)
            self.conf.burst_size.append(1)
            self.conf.size_dist.append ("Exponential")
            self.conf.max_len.append(0)
            self.conf.keep_pkts.append(False)
            self.conf.last_k.append(100)
 

        self.__window_txa.destroy()
        self.__window_txa = None

    def function_slice(self,numerology,num_ugr,sl_ugr,ugr_notassigned):
        """ This is the callback function of the AppSlice to configure the slices.

        @type name: string.
        @param name: The name of the slice.
        @type number: int
        @param number: the slice number.
        @type numerology: int
        @param numerology The numerology used by this slice.
        """ 
        self.conf.numerology = numerology
        if self.conf.nuegroups < num_ugr:
            self.conf.uegr_par1.append(50)
            self.conf.num_ues.append(5)
            self.conf.trgen_type.append("poisson")
            self.conf.inter_arrival.append(1)
            self.conf.pkt_size.append(300)
            self.conf.burst_size.append(1)
            self.conf.size_dist.append ("Exponential")
            self.conf.max_len.append(0)
            self.conf.keep_pkts.append(False)
            self.conf.last_k.append(100)
 
 
        self.conf.nuegroups =num_ugr
        self.conf.slice_uegr = sl_ugr
        self.conf.ugr_not_assigned
        self.__window_txa.destroy()
        self.__window_txa = None
        
    def function_resource(self,namedl,nameul,nresdl,nresul,sym_slot,slice_number):
        """ This is the callback function of the AppAResource to configure the resources.

        @param namedl: name of the resources for downlink.
        @param nameul: name of the resources for uplink.
        @param nresdl: number of rsources for downlink.
        @param nresul: number of resources for uplink.
        @param sym_slot: In TDD the resources are shared for downlik and uplink. This is the number of symbols used for downlik. Uplink uses 14 - sym_slot.
        @param slice_number: The slice number where these resources are initially assigned.      
        
        """ 
        self.conf.sym_slot[slice_number] = int(sym_slot)
        self.conf.nresdl[slice_number] = int(nresdl)
        self.conf.nresul[slice_number] = int(nresul)
        self.conf.name_resdl[slice_number] = namedl
        self.conf.name_resul[slice_number] = nameul
        if slice_number < self.conf.num_slices-1:
            self.__window_txa.destroy()
            self.__window_txa = None
            self.__window_txa = tk.Tk()
            app = gres.AppResource(self.__window_txa,self.function_resource,"Resources Specification",self.conf.name_resdl[slice_number+1],self.conf.name_resul[slice_number+1],self.conf.nresdl[slice_number+1],self.conf.nresul[slice_number+1],self.conf.sym_slot[slice_number+1],self.conf.ul,self.conf.dl,self.conf.band,slice_number+1)
            self.__window_txa.mainloop()
        else:
            self.__window_txa.destroy()
            self.__window_txa = None


    def function_sched(self,name):
        """ This is the callback function of the AppScheduler to configure the scheduler.

        @type name: string.
        @param name: The type of scheduler.
                """ 
        self.conf.name_sched = name
        self.__window_txa.destroy()
        self.__window_txa = None

    def function_channel(self,channel_type,ch_mode,file,val1,val2,loss):
        """ This is the callback function of the AppAntenna to configure the TX antenna.

        @type channel_type: string.
        @param channel_type: The type of channel.
        @type ch_mode: string
        @param ch_mode: the channel mode: fixed or random.
        @type file: string
        @param file: The file in case of file channel.
        @type val1: float
        @param val1: The min value of the snr for fixed or random channel.      
        @type val2: float
        @param val2: The max value of the snr for fixed or random channel.      

        """ 
        self.conf.channel_type = channel_type
        self.conf.channel_mode = ch_mode
        self.conf.file_channel = file
        self.conf.val_1 = val1
        self.conf.val_2 = val2
        self.conf.loss_prob = loss
        self.__window_txa.destroy()
        self.__window_txa = None

 
    def function_uegroups(self,name,nuegr,par1,num_ues):
        """ This is the callback function of the AppUserGroups to configure the user groups.

        @type name: string.
        @param name: The name of the user group..
        @type nuegr: int
        @param nuegr: The number of the user group.
        @type par1: int
        @param par1: The maximum number of resources that can be assigned to a UE.
        @type num_ues: int
        @param num_ues: Number of users in this user group.      
        """ 
        self.conf.name_uegroup[nuegr] = name
        self.conf.uegr_par1[nuegr] = par1
        self.conf.num_ues[nuegr] = num_ues
        
        if nuegr < self.conf.nuegroups -1:
            self.__window_txa.destroy()
            self.__window_txa = None
            self.__window_txa = tk.Tk()
            app = guegr.AppUserGroup(self.__window_txa,self.function_uegroups,"User Group Specification",self.conf.name_uegroup[nuegr+1],nuegr+1,self.conf.uegr_par1[nuegr+1],self.conf.num_ues[nuegr+1])
            self.__window_txa.mainloop()
        else:
            self.__window_txa.destroy()
            self.__window_txa = None

    def function_slice_uegr(self,nslice,nuegr):
        """ This is the callback function of the AppSliceUegr to configure the associattion of slices and user groups.

        @type nslice: int.
        @param nslice: the slice number.
        @type nuegr: int
        @param nuegr: the user group number.

        """ 
      
        self.conf.slice_uegr[nslice] = nuegr
        if nslice < self.conf.num_slices-1:
            self.__window_txa.destroy()
            self.__window_txa = None
            self.__window_txa = tk.Tk()
            """ The tk.TK() window for the AppAntenna Tx form. """ 
            app = gsluegr.AppSliceUegr(self.__window_txa,self.function_slice_uegr,"Slice-UEGroups association",nslice+1,self.conf.slice_uegr[nslice+1],self.conf.name_uegroup ) 
            self.__window_txa.mainloop()
        else:
             self.__window_txa.destroy()
             self.__window_txa = None

    def function_uegr_trfgen(self,trgen_type,inter_arrival,pkt_size,burst_size,size_dist,max_len,last_k,keep_pkts):
        """ This is the callback function of the AppUegrTrfgen to configure the user group -traffic generator association.

        @param nuegr: User group number.
        @param trgen_type: The type of the traffic generator.
        @param inter_arrival: The inter arrival time of the packets.
        @param pkt_size: The packet size.
        @param burst_size: The number of packets in a burst.
        @param size_dist: The size of packets distribution.
        """ 
        self.conf.trgen_type= trgen_type
        self.conf.inter_arriva=inter_arrival
        self.conf.pkt_size = pkt_size
        self.conf.burst_size = burst_size
        self.conf.size_dist = size_dist
        self.conf.max_len = max_len
        self.conf.keep_pkts = keep_pkts
        self.conf.last_k = last_k
        self.__window_txa.destroy()
        self.__window_txa = None

def create_directory_if_not_exists(dire):
    if not os.path.exists(dire):
        os.makedirs(dire)
        print(f"Directory '{dire}' created.")
    else:
        print(f"The directory '{dire}' already exists. ")



if __name__ == "__main__":
    window = tk.Tk()

    cf = ConfigScenary()
    data_dir = directory+'/data'
    create_directory_if_not_exists(data_dir)
    app = AppGui(window,"5G Python Wireless Simulator")

    try:
        window.mainloop()
    except:
        print('Exception!', sys.exc_info()[0],'occurred.' )


