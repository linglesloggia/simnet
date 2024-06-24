#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is a gui for the configuration of the user group.

"""
import tkinter as tk
import tkinter.font as tkfont

class AppUserGroup():
    """ This class is the form for the configuration of the user group. """

    def __init__(self,window,function_cbk,title,name,number,par1,numue):
        """The constructor of the AppUserGroup Class.
        
        @type window: tkinter.Tk() window.
        @param window: The main window of this form.
        @type function_cbk: python function.
        @param function_cbk: The callback function to return when the user press OK button.
        @type title: string
        @param title: The title of the form.
        @type name: string.
        @param name: The name of the user group..
        @type number: int
        @param number: The number of the user group.
        @type par1: int
        @param par1: The maximum number of resources that can be assigned to a UE.
        @type numue: int
        @param numue: Number of users in this user group.      

       """
        self.window  = window
        """ The main window of this form.""" 
        self.window.title(title)
        """ The title of this form. """ 
        self.function_cbk = function_cbk
        """ The callback function to return when the user press OK button. """ 
        self.name = name
        self.number= number
        self.par1 = par1
        self.numue = numue
        self.gui_configuration()
    
    
    def save(self):
        """ This method is called whn the user press the save button
        """
        self.function_cbk(self.ent_name.get(),self.number,int(self.ent_par1.get()),int(self.ent_numue.get()))
            
 
         
    def gui_configuration(self):
        """This method builds the main form to enter the data
        
        """   
        ##General definitions of the form
        width, height = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        self.__FONT_SIZE = int(width /1.25/ 1920*25)
        font = tkfont.Font(family="Helvetica", size=self.__FONT_SIZE, weight = tkfont.BOLD)
        self.window.rowconfigure(10, minsize=50, weight=1)
        self.window.columnconfigure([0, 1, 2, 3], minsize=50, weight=1)       
        ###Title of the form
        frm_uegr = tk.Frame(master=self.window)
        lbl_uegr = tk.Label(master=frm_uegr, text=" User Group "+str(self.number+1)+" Configuration",fg = "blue",bg = "white",font = "Verdana 14 bold")
        lbl_uegr.grid(row=0, column=0, sticky="w")    
        ### entry for the name of the user group
        frm_name = tk.Frame(master=self.window)
        self.ent_name = tk.Entry(master=frm_name, width=12)
        lbl_name = tk.Label(master=frm_name, text="Name of the User Group")
        self.ent_name.grid(row=1, column=0, sticky="e")
        lbl_name.grid(row=0, column=0, sticky="w")
        self.ent_name.insert(tk.END,self.name)
        self.ent_name.configure(state= 'disable', disabledbackground='white', disabledforeground='red')
        ### entry to assign the max bound of assigned resources
        frm_par1 = tk.Frame(master=self.window)
        self.ent_par1 = tk.Entry(master=frm_par1, width=12)
        lbl_par1 = tk.Label(master=frm_par1, text="Max bound of assigned resources")
        self.ent_par1.grid(row=1, column=0, sticky="e")
        lbl_par1.grid(row=0, column=0, sticky="w")
        self.ent_par1.insert(tk.END,str(self.par1))
        ## entry for the number of user equipments in the group
        frm_numue = tk.Frame(master=self.window)
        self.ent_numue = tk.Entry(master=frm_numue, width=12)
        lbl_numue = tk.Label(master=frm_numue, text="Number of user equipments in this group")
        self.ent_numue.grid(row=1, column=0, sticky="e")
        lbl_numue.grid(row=0, column=0, sticky="w")
        self.ent_numue.insert(tk.END,str(self.numue))
        ###The save button
        frm_save = tk.Frame(master=self.window)
        save = tk.Button(master=frm_save, text="Save \nConfiguration",font=font, compound=tk.CENTER, command=self.save)
        save.grid(row=0, column=0, columnspan=1, sticky='EWNS') #padx=10)
 
        ###The place of the diferent objects in the form
        frm_uegr.grid(row=0, column=1, padx=10)
        frm_name.grid(row=1, column=0, padx=10)
        frm_par1.grid(row=1, column=1, padx=10)
        frm_numue.grid(row=1, column=2, padx=10)
        frm_save.grid(row=2, column=2, padx=10)


 
