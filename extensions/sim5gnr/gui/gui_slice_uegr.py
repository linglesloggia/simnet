#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is a gui for the configuration of the slice-user group association.


"""
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont


class AppSliceUegr():
    """ This class is the form for the configuration of the slice-user group association. """

    def __init__(self,window,function_cbk,title,nslice,nuegr,uegr_list):
        """The constructor of the AppSliceUegr Class.
        
        @type window: tkinter.Tk() window.
        @param window: The main window of this form.
        @type function_cbk: python function.
        @param function_cbk: The callback function to return when the user press OK button.
        @type title: string
        @param title: The title of the form.
        @type nslice: int.
        @param nslice: the slice number.
        @type nuegr: int
        @param nuegr: the user group number.
        @type uegr_list: list
        @param uegr_list: The list of user groups.
        
       """
        self.window  = window
        """ The main window of this form.""" 
        self.window.title(title)
        """ The title of this form. """ 
        self.function_cbk = function_cbk
        """ The callback function to return when the user press OK button. """ 
        self.nslice = nslice
        self.nuegr = nuegr
        self.uegr_list = uegr_list

        self.gui_configuration()
    

    
    def save(self):
        """ This method is called when the user press the save button
        """
        self.function_cbk(self.nslice,self.combo_uegr.current())
            

         
    def gui_configuration(self):
        """This method builds the main form to enter the data
        
        """    
        ### General definitions of the form
        width, height = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        self.__FONT_SIZE = int(width /1.25/ 1920*25)
        font = tkfont.Font(family="Helvetica", size=self.__FONT_SIZE, weight = tkfont.BOLD)
        self.window.rowconfigure(10, minsize=50, weight=1)
        self.window.columnconfigure([0, 1, 2, 3], minsize=50, weight=1)       
        ### The title of the form
        frm_sluegr = tk.Frame(master=self.window)
        lbl_sluegr = tk.Label(master=frm_sluegr, text="Slice " + str(self.nslice+1) +" -User Group Association",fg = "blue",bg = "white",font = "Verdana 14 bold")
        lbl_sluegr.grid(row=0, column=0, sticky="w")    
        ###Combo box of user groups
        frm_uegr = tk.Frame(master=self.window)
        self.combo_uegr = ttk.Combobox(master=frm_uegr, width=12,state="readonly",values=self.uegr_list)
        self.combo_uegr.current(self.nuegr)
        lbl_uegr = tk.Label(master=frm_uegr, text="User Groups List")
        self.combo_uegr.grid(row=1, column=0, sticky="w")
        lbl_uegr.grid(row=0, column=0, sticky="w")
        ###save button
        frm_save = tk.Frame(master=self.window)
        save = tk.Button(master=frm_save, text="Save \nConfiguration",font=font, compound=tk.CENTER, command=self.save)
        save.grid(row=0, column=0, columnspan=1, sticky='EWNS') #padx=10)
        ###Place of the user groups in the form
        frm_sluegr.grid(row=0, column=1, padx=10)
        frm_uegr.grid(row=1, column=0, padx=10)
        frm_save.grid(row=2, column=2, padx=10)


        
                
 

