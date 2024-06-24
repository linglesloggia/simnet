#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is a gui for asking the user to enter one data.

@author: pablobelzarena
"""

import tkinter as tk
from tkinter import ttk

import gui_user_message as gum


class AppAskOptions():
    """ This class is the form for enter the data"""
    
    def __init__(self,window,function_cbk,title,ok_title,data_description,list_val,val_defualt):
        """The constructor of the AppNameInput Class
        
        @type window: tkinter.Tk() window.
        @param window: The main window of this form.
        @type function_cbk: python function.
        @param function_cbk: The callback function to return when the user press OK button.
        @type title: string
        @param title: The title of the form.
        @type ok_title: string
        @param ok_title: The text to show in the OK button.
        @type data_description: string
        @param data_description: The label to explain the data.
        @type value: string,int,float depend on the data.
        @param value: The default value of the data.

        """
        self.ok_title = ok_title
        """ The text to show in the OK button.""" 
        self.window  = window
        """ The main window of this form.""" 
        self.window.title(title)
        self.data_description = data_description
        """ The label to explain the data.""" 
        self.function_cbk = function_cbk
        """ The callback function to return when the user press OK button.""" 
        self.list_val = list_val
        self.val_defualt = val_defualt
        """ the default value of the data."""
        self.gui_configuration()

    def gui_configuration(self):
        """This method builds the form to enter the data
        
        """ 
        self.window.rowconfigure(6, minsize=50, weight=1)
        self.window.columnconfigure([0, 1, 2, 3, 4], minsize=50, weight=1) 
        frm_dato = tk.Frame(master=self.window)
        """The tkinter Entry object to enter the data"""
        self.combo = ttk.Combobox(master=frm_dato, width=12,state="readonly",values=self.list_val)
        self.combo.current(self.list_val.index(self.val_defualt))
 
        lbl_dato = tk.Label(master=frm_dato, text=self.data_description)
        self.combo.grid(row=1, column=0, sticky="e")
        lbl_dato.grid(row=0, column=0, sticky="w")
        frm_dato.grid(row=0, column=0, padx=10)
        
    
        aux1 = tk.Button(self.window, text=self.ok_title, command=self.OK_button)
        aux1.grid(row=1, column=0, padx=10)

    def OK_button(self):
        """This method is called when user press the OK button
 
        This method calls the callback function.
        """ 
        self.function_cbk(self.combo.get())

