#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is a gui for the configuration of the user group - traffic generator association.

"""
import tkinter as tk
import tkinter.font as tkfont
import gui_traffic_generator as gtrgen


class AppUegrTrfgen():
    """ This class is the form for the configuration of the user group - traffic generator association. """

    def __init__(self,window,function_cbk,title,name,nuegr,inter_arrival,pkt_size,size_dist ,burst_size,ul,dl):
        """The constructor of the AppAUegrTrfgen Class.
        
        @type window: tkinter.Tk() window.
        @param window: The main window of this form.
        @type function_cbk: python function.
        @param function_cbk: The callback function to return when the user press OK button.
        @type title: string
        @param title: The title of the form.
        @type name: string
        @param name: The type of the traffic generator.
        @param nuegr: User group number.
        @param inter_arrival: The inter arrival time of the packets.
        @param pkt_size: The packet size.
        @param size_dist: The size of packets distribution.
        @param burst_size: The number of packets in a burst.
        @param ul: If the simulation is uplink.
        @param dl: If the simulation is downlink.

       """
        self.window  = window
        """ The main window of this form.""" 
        self.window.title(title)
        """ The title of this form. """ 
        self.function_cbk = function_cbk
        """ The callback function to return when the user press OK button. """ 
        print("--------name .......",name)
        """  The type of the traffic generator. """ 
        self.nuegr = nuegr
        self.trgen_type = name
        self.inter_arrival = inter_arrival
        self.pkt_size = pkt_size
        self.size_dist = size_dist
        self.burst_size = burst_size
        self.dl = dl
        self.ul = ul

        self.gui_configuration()
    
     
    def save(self):
        """ This method is called when the user press the save button
        """
        self.function_cbk(self.nuegr,self.trgen_type,self.inter_arrival,self.pkt_size,self.burst_size,self.size_dist)
            
    def trgen(self):
        """ This method is called when the user selects the traffic generator button.
        """
        self.__window_txa = tk.Tk()
        app = gtrgen.AppTrGen(self.__window_txa,self.function_trgen,"Traffic Generator Specification",self.trgen_type ,self.inter_arrival, self.pkt_size, self.size_dist, self.burst_size,self.dl,self.ul ) 
        self.__window_txa.mainloop()

         
    def gui_configuration(self):
        """This method builds the main form to enter the data
        
        """   
        ### General definitions of the form
        width, height = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        self.__FONT_SIZE = int(width /1.25/ 1920*25)
        font = tkfont.Font(family="Helvetica", size=self.__FONT_SIZE, weight = tkfont.BOLD)
        self.window.rowconfigure(10, minsize=50, weight=1)
        self.window.columnconfigure([0, 1, 2, 3], minsize=50, weight=1)       
        ###Title of the form
        frm_uegtg = tk.Frame(master=self.window)
        lbl_uegtg= tk.Label(master=frm_uegtg, text="User Group "+str(self.nuegr+1)+" Traffic Generator association",fg = "blue",bg = "white",font = "Verdana 14 bold")
        lbl_uegtg.grid(row=0, column=0, sticky="w")    
        ###Traffic generator button
        frm_trgen = tk.Frame(master=self.window)
        trgen = tk.Button(master=frm_trgen, text="Select Traffic \n Generator for UEG- "+str(self.nuegr+1),font=font, compound=tk.CENTER, command=self.trgen)
        trgen.grid(row=0, column=0, columnspan=1, sticky='EWNS') #padx=10)
        ###Save button
        frm_save = tk.Frame(master=self.window)
        save = tk.Button(master=frm_save, text="Save \nConfiguration",font=font, compound=tk.CENTER, command=self.save)
        save.grid(row=0, column=0, columnspan=1, sticky='EWNS') #padx=10)
        ### Place of the different objects in the form.
        frm_uegtg.grid(row=0, column=1, padx=10)
        frm_trgen.grid(row=1, column=0, padx=10)
        frm_save.grid(row=1, column=2, padx=10)


    def function_trgen(self,trfgen_type,pkt_dist,inter_arrival,pkt_size,burst_size):
        """ This is the callback function of the traffic generator.

        @param trgen_type: The type of the traffic generator.
        @param inter_arrival: The inter arrival time of the packets.
        @param pkt_size: The packet size.
        @param burst_size: The number of packets in a burst.
        @param pkt_dist: The size of packets distribution.
     
        """ 
      
        self.trgen_type = trfgen_type
        self.inter_arrival = inter_arrival
        self.pkt_size = pkt_size
        self.burst_size = burst_size
        self.size_dist = pkt_dist
        print(trfgen_type,pkt_dist,inter_arrival,pkt_size,burst_size)
        self.__window_txa.destroy()

                
 

