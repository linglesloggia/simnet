#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is a gui for the configuration of the scheduler.

"""
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont


class AppScheduler():
    """ This class is the form for the configuration of the scheduler. """

    def __init__(self,window,function_cbk,title,name):
        """The constructor of the AppScheduler Class.
        
        @type window: tkinter.Tk() window.
        @param window: The main window of this form.
        @type function_cbk: python function.
        @param function_cbk: The callback function to return when the user press OK button.
        @type title: string
        @param title: The title of the form.
        @type name: string
        @param name: The type of the scheduler.
       """
        self.window  = window
        """ The main window of this form.""" 
        self.window.title(title)
        """ The title of this form. """ 
        self.function_cbk = function_cbk
        """ The callback function to return when the user press OK button. """ 
        self.name = name
        self.schedulers_list = ["simple","round robin"]

        self.scheduler = self.schedulers_list.index(name)
        self.gui_configuration()
    
    
    def save(self):
        """ This method is called when the user press the save button.
        """
        self.function_cbk(self.combo_schtypes.get())
            
 
         
    def gui_configuration(self):
        """This method builds the main form to enter the data
        
        """    
        #### general definitions of the form
        width, height = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        self.__FONT_SIZE = int(width /1.25/ 1920*25)
        font = tkfont.Font(family="Helvetica", size=self.__FONT_SIZE, weight = tkfont.BOLD)
        self.window.rowconfigure(10, minsize=50, weight=1)
        self.window.columnconfigure([0, 1, 2, 3], minsize=50, weight=1)       
        ### Title of the form
        frm_sched = tk.Frame(master=self.window)
        lbl_sched = tk.Label(master=frm_sched, text=" Scheduler Configuration",fg = "blue",bg = "white",font = "Verdana 14 bold")
        lbl_sched.grid(row=0, column=0, sticky="w")    
        ## Combobox to select the scheduler type
        frm_schtypes = tk.Frame(master=self.window)
        self.combo_schtypes = ttk.Combobox(master=frm_schtypes, width=12,state="readonly",values=self.schedulers_list)
        self.combo_schtypes.current(self.scheduler)
        lbl_schtypes = tk.Label(master=frm_schtypes, text="Scheduler type")
        self.combo_schtypes.grid(row=1, column=0, sticky="w")
        lbl_schtypes.grid(row=0, column=0, sticky="w")
        
        ##Save button
        frm_save = tk.Frame(master=self.window)
        save = tk.Button(master=frm_save, text="Save \nConfiguration",font=font, compound=tk.CENTER, command=self.save)
        save.grid(row=0, column=0, columnspan=1, sticky='EWNS') #padx=10)
 
        ### Places of the objects in the form
        frm_schtypes.grid(row=1, column=0, padx=10)
        frm_sched.grid(row=0, column=1, padx=10)
        frm_save.grid(row=2, column=2, padx=10)

                
 

