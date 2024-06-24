#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is a gui for the configuration of the base station.

"""
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont
from extensions.sim5gnr.tables import load_bands

class AppBaseStation():
    """ This class is the form for the configuration of the base station. """

    def __init__(self,window,function_cbk,title,name="BaseStation 1",band = "n258" ,robustMCS=False,long_cp = False,ul = False,dl = True,mimo_mod ="SU",nuegroups = 1,nslices=1,time_sim = 15):
        """The constructor of the AppABaseStation Class.
        
        @type window: tkinter.Tk() window.
        @param window: The main window of this form.
        @type function_cbk: python function.
        @param function_cbk: The callback function to return when the user press OK button.
        @type title: string
        @param title: The title of the form.
        @type name: string
        @param name: The name of the base station.
        @type band: string
        @param band: The operation frequency band of the system.
        type rMCS: Boolean.
        @param rMCS: Roubust Modulation and code scheme or not. 
        @type longcp: Boolean.
        @param longcp: Long cyclic prefix or not. 
        @type ul: Boolean.
        @param ul: If the simulation is for downlink or not. 
        @type dl: Boolean.
        @param dl: If the simulation is for uplink or not.         
        @type mimo_mod: "MU" or "SU".
        @param mimo_mod: single user o multiuser mimo.
        @type nuegroups: int.
        @param nuegroups th number of user groups.        
        @type nslices: int.
        @param nslices: th number of slices in the base node.

       """
        self.window  = window
        """ The main window of this form.""" 
        self.window.title(title)
        """ The title of this form. """ 
        self.function_cbk = function_cbk
        """ The callback function to return when the user press OK button. """ 
        self.bs_name = name
        """  The name of the Base Station. """ 
        self.band = band 
        self.robustMCS=robustMCS
        self.long_cp = long_cp
        self.ul = ul
        self.dl = dl
        self.mimo_mod = mimo_mod
        self.nuegroups = nuegroups
        self.nslices = nslices
        self.time_sim = time_sim

        self.gui_configuration()
    
    def bands_selection_changed(self,event):
        """
        This method is called when the combo_bands changes.
        """
        txt = "Type: fr" + str(self.bands[self.combo_bands.get()][0])+ " \nBandwith: "+ str(self.bands[self.combo_bands.get()][1]) + " MHz \nTDD: "+str(self.bands[self.combo_bands.get()][2]) 
        self.lbl_band_selected.config(text = txt)

    def validate_nuegroups(self,event:tk.Event=None):
        """
        This method is called when the nuegroups entry changes.
        """
        nuegr = self.ent_nuegroups.get()
        try:
            int(nuegr)
        except ValueError:
            tk.messagebox.showinfo(message="The number of user groups must be an integer", title="Error")
            self.ent_nuegroups.delete(0, 'end')
            self.ent_nuegroups.insert(0,"1")  
            self.ent_nuegroups.focus()

    def validate_nslices(self,event:tk.Event=None):
        """
        This method is called when the nslices entry changes.
        """
        nslices = self.ent_nslices.get()
        try:
            int(nslices)
        except ValueError:
            tk.messagebox.showinfo(message="The number of slices must be an integer", title="Error")
            self.ent_nslices.delete(0, 'end')
            self.ent_nslices.insert(0,"1")  
            self.ent_nslices.focus()
    
    def validate_time_sim(self,event:tk.Event=None):
        """
        This method is called when the nslices entry changes.
        """
        time_sim = self.ent_time_sim.get()
        try:
            int(time_sim)
        except ValueError:
            tk.messagebox.showinfo(message="The simulation time must be an integer", title="Error")
            self.ent_time_sim.delete(0, 'end')
            self.ent_time_sim.insert(0,"15")  
            self.ent_time_sim.focus()

    
    def save(self):
        """
        This method is called when the sve button is pressed.
        """
        self.function_cbk(self.combo_mimo.get(),self.ent_nslices.get(),self.combo_bands.get(),self.ent_name.get(),self.combo_rMCS.get(),self.combo_longcp.get(),self.combo_ul.get(),self.combo_dl.get(),self.ent_time_sim.get())
            

         
    def gui_configuration(self):
        """This method builds the main form to enter the data
        
        """    
        width, height = self.window.winfo_screenwidth(), self.window.winfo_screenheight()

        self.__FONT_SIZE = int(width /1.25/ 1920*25)
        font = tkfont.Font(family="Helvetica", size=self.__FONT_SIZE, weight = tkfont.BOLD)
        self.window.rowconfigure(10, minsize=50, weight=1)
        self.window.columnconfigure([0, 1, 2, 3], minsize=50, weight=1)       

        #### load bands from 5G tables
        self.bands = load_bands()
        bands_names = list(self.bands.keys())
        
        # label for title of the form
        frm_bs = tk.Frame(master=self.window)
        lbl_bs = tk.Label(master=frm_bs, text="Base Station Configuration",fg = "blue",bg = "white",font = "Verdana 14 bold")
        lbl_bs.grid(row=0, column=0, sticky="w")

        #### label and combobox for bands selection
        frm_bands = tk.Frame(master=self.window)
        self.combo_bands = ttk.Combobox(master=frm_bands, width=12,state="readonly",values=bands_names)
        self.combo_bands.current(bands_names.index(self.band))
        self.combo_bands.bind("<<ComboboxSelected>>", self.bands_selection_changed)
        lbl_bands = tk.Label(master=frm_bands, text="Frequency Band")
        self.combo_bands.grid(row=1, column=0, sticky="w")
        lbl_bands.grid(row=0, column=0, sticky="w")
        ### when the user select one band the following label shows its properties 
        txt = "Type: fr" + str(self.bands[self.combo_bands.get()][0])+ " \nBandwith: "+ str(self.bands[self.combo_bands.get()][1]) + " MHz \nTDD: "+str(self.bands[self.combo_bands.get()][2])
        self.lbl_band_selected =tk.Label(master=frm_bands, text=txt)
        self.lbl_band_selected.grid(row=2, column=0, sticky="w")

        ##Entry for the name of the base station, it is disable the user cannot edit it.
        frm_name = tk.Frame(master=self.window)
        self.ent_name = tk.Entry(master=frm_name, width=12)
        """The tkinter Entry object to enter the name of the BS. """
        lbl_name = tk.Label(master=frm_name, text="Name of the Base Station")
        self.ent_name.grid(row=1, column=0, sticky="e")
        lbl_name.grid(row=0, column=0, sticky="w")
        self.ent_name.insert(tk.END,self.bs_name)
        self.ent_name.configure(state= 'disable', disabledbackground='white', disabledforeground='red')

        ##Combo bosx for select the robust MCS property
        frm_rMCS = tk.Frame(master=self.window)
        self.combo_rMCS = ttk.Combobox(master=frm_rMCS, width=12,state="readonly",values=["False","True"])
        self.combo_rMCS.current(self.robustMCS)
        lbl_rMCS = tk.Label(master=frm_rMCS, text="Robust MCS")
        self.combo_rMCS.grid(row=1, column=0, sticky="e")
        lbl_rMCS.grid(row=0, column=0, sticky="w")
        
        ##Combo box to select the long prefix property
        frm_longcp = tk.Frame(master=self.window)
        self.combo_longcp = ttk.Combobox(master=frm_longcp, width=12,state="readonly",values=["False","True"])
        self.combo_longcp.current(self.long_cp)
        lbl_longcp = tk.Label(master=frm_longcp, text="Long CP")
        self.combo_longcp.grid(row=1, column=0, sticky="e")
        lbl_longcp.grid(row=0, column=0, sticky="w")
        
        ##Combo box to select if the simulation is downlik
        frm_ul = tk.Frame(master=self.window)
        self.combo_ul = ttk.Combobox(master=frm_ul, width=12,state="readonly",values=["False","True"])
        self.combo_ul.current(self.ul)
        lbl_ul = tk.Label(master=frm_ul, text="Uplink Simulation")
        self.combo_ul.grid(row=1, column=0, sticky="e")
        lbl_ul.grid(row=0, column=0, sticky="w")

         ##Combo box to select if the simulation is downlink       
        frm_dl = tk.Frame(master=self.window)
        self.combo_dl = ttk.Combobox(master=frm_dl, width=12,state="readonly",values=["False","True"])
        self.combo_dl.current(self.dl)
        lbl_dl = tk.Label(master=frm_dl, text="Downlink Simulation")
        self.combo_dl.grid(row=1, column=0, sticky="e")
        lbl_dl.grid(row=0, column=0, sticky="w")
        
        ##Combo box to select if mimo multiuser or single user.
        frm_mimo = tk.Frame(master=self.window)
        self.combo_mimo = ttk.Combobox(master=frm_mimo, width=12,state="readonly",values=["SU","MU"])
        if self.mimo_mod == "SU":
            self.combo_mimo.current(0)
        else:
            self.combo_mimo.current(1)
        lbl_mimo = tk.Label(master=frm_mimo, text="MIMO mode")
        self.combo_mimo.grid(row=1, column=0, sticky="e")
        lbl_mimo.grid(row=0, column=0, sticky="w")
        
        ## Entry for the number of slices
        frm_nslices = tk.Frame(master=self.window)
        self.ent_nslices = tk.Entry(master=frm_nslices, width=12)
        self.ent_nslices.bind("<FocusOut>", self.validate_nslices)
        lbl_nslices = tk.Label(master=frm_nslices, text="Number of Slices")
        self.ent_nslices.grid(row=1, column=0, sticky="e")
        lbl_nslices.grid(row=0, column=0, sticky="w")
        self.ent_nslices.insert(tk.END,str(self.nslices))

        # ## Entry for the number of user groups
        # frm_nuegroups = tk.Frame(master=self.window)
        # self.ent_nuegroups = tk.Entry(master=frm_nuegroups, width=12)
        # self.ent_nuegroups.bind("<FocusOut>", self.validate_nuegroups)
        # lbl_nuegroups = tk.Label(master=frm_nuegroups, text="Number of User Groups")
        # self.ent_nuegroups.grid(row=1, column=0, sticky="e")
        # lbl_nuegroups.grid(row=0, column=0, sticky="w")
        # self.ent_nuegroups.insert(tk.END,str(self.nuegroups)) 
        
        ## Entry for the simulation duration
        frm_time_sim = tk.Frame(master=self.window)
        self.ent_time_sim = tk.Entry(master=frm_time_sim, width=12)
        self.ent_time_sim.bind("<FocusOut>", self.validate_time_sim)
        lbl_time_sim = tk.Label(master=frm_time_sim, text="Simulation time (ms)")
        self.ent_time_sim.grid(row=1, column=0, sticky="e")
        lbl_time_sim.grid(row=0, column=0, sticky="w")
        self.ent_time_sim.insert(tk.END,str(self.time_sim)) 
        
        ### button to save the configuration
        frm_save = tk.Frame(master=self.window)
        save = tk.Button(master=frm_save, text="Save \nConfiguration",font=font, compound=tk.CENTER, command=self.save)
        save.grid(row=0, column=0, columnspan=1, sticky='EWNS') #padx=10)
 
        #### position of the different objects in the form
        frm_bs.grid(row=0, column=1, padx=10)
        frm_bands.grid(row=2, column=0, padx=10)
        frm_name.grid(row=1, column=0, padx=10)
        frm_rMCS.grid(row=1, column=1, padx=10)
        frm_longcp.grid(row=1, column=2, padx=10)
        frm_dl.grid(row=2, column=1, padx=10)
        frm_ul.grid(row=2, column=2, padx=10)
        frm_mimo.grid(row=3, column=0, padx=10)
        frm_nslices.grid(row=3, column=1, padx=10)
        # frm_nuegroups.grid(row=3, column=2, padx=10)
        frm_time_sim.grid(row=4, column=0, padx=10)
        frm_save.grid(row=4, column=2, padx=10)

        return     
                
 

