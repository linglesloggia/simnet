#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is a gui for the configuration of the Resources.

"""
import tkinter as tk
import tkinter.font as tkfont
from extensions.sim5gnr.tables import load_bands
from dframe import dFrame

class AppResource():
    """ This class is the form for the configuration of the resources. """

    def __init__(self,window,function_cbk,title,namedl="PRB",nameul ="",n_resdl=2,n_resul=0,sym_slot=14,ul=False,dl=True,band="n258",sl_number=0):
        """The constructor of the AppResource Class.
        
        @type window: tkinter.Tk() window.
        @param window: The main window of this form.
        @type function_cbk: python function.
        @param function_cbk: The callback function to return when the user press OK button.
        @type title: string
        @param title: The title of the form.
         @param namedl: Name of the resources for downlink.
        @param nameul: Name of the resources for uplink.
        @param n_resdl: Number of rsources for downlink.
        @param n_resul: Number of resources for uplink.
        @param sym_slot: In TDD the resources are shared for downlik and uplink. This is the number of symbols used for downlik. Uplink uses 14 - sym_slot.
        @param ul: If the simulation is for uplink.
        @param dl: If the simulation is for downlink.
        @param band: The frequency band of the simulation.
        @param sl_number: The slice number where these resources are initially assigned.      
 
       """
        self.window  = window
        """ The main window of this form.""" 
        self.window.title(title)
        """ The title of this form. """ 
        self.function_cbk = function_cbk
        """ The callback function to return when the user press OK button. """ 
        self.namedl = namedl
        self.nameul = nameul
        """  The type of resource. """ 
        self.nresdl = n_resdl
        self.nresul = n_resul
        self.sym_slot = sym_slot
        self.ul = ul
        self.dl = dl
        self.bands = load_bands()
        self.tdd = self.bands[band][2] 
        self.sl_number = sl_number
        self.gui_configuration()
    
    
    def save(self):
        """ This method is called when the user press the save button.
        """
        self.function_cbk(self.ent_name.get(),self.ent_name1.get(),self.ent_nres.get(),self.ent_nres1.get(),self.ent_sym.get(),self.sl_number)
            
    def validate_nres(self,event:tk.Event=None):
        """ This method is called when the user change the number of resources.
        """
        nres = self.ent_nres.get()
        try:
            int(nres)
        except ValueError:
            tk.messagebox.showinfo(message="The number of resources must be an integer", title="Error")
            self.ent_nres.delete(0, 'end')
            self.ent_nres.insert(0,"1")  
            self.ent_nres.focus()
            
    def validate_nsym(self,event:tk.Event=None):
        """ This method is called when the user change the symbols per slot.
        """
        nsym = self.ent_sym.get()
        try:
            int(nsym)
            if nsym>14 or nsym < 0:
                tk.messagebox.showinfo(message="The number of symbols per slot must be between 0 and 14", title="Error")
                self.ent_sym.delete(0, 'end')
                self.ent_sym.insert(0,"14")  
                self.ent_sym.focus()  
        except ValueError:
            tk.messagebox.showinfo(message="The number of symbols per slot must be an integer", title="Error")
            self.ent_sym.delete(0, 'end')
            self.ent_sym.insert(0,"14")  
            self.ent_sym.focus()
        if self.ul and self.dl and not self.tdd:
            self.ent_sym1.delete(0, 'end')
            self.ent_sym1.insert(0,"14")  
        if self.ul and self.dl and  self.tdd:
           self.ent_sym1.delete(0, 'end')
           self.ent_sym1.insert(0,str(14-int(nsym)))  
         
        
 
         
    def gui_configuration(self):
        """This method builds the main form to enter the data
        
        """    
        ### general definitions of the form
        width, height = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        self.__FONT_SIZE = int(width /1.25/ 1920*25)
        font = tkfont.Font(family="Helvetica", size=self.__FONT_SIZE, weight = tkfont.BOLD)
        self.window.rowconfigure(10, minsize=50, weight=1)
        self.window.columnconfigure([0, 1, 2, 3], minsize=50, weight=1)       
        ### depending if the band is tdd and if the simulation is dl and ul thre resources are shared or not
        frm_res = tk.Frame(master=self.window)
        if self.tdd:
            if self.dl:
                if self.ul:
                    lbl_res = tk.Label(master=frm_res, text=" TDD shared DL/UL Resources Configuration for Slice "+str(self.sl_number+1),fg = "blue",bg = "white",font = "Verdana 14 bold")
                else:
                    lbl_res = tk.Label(master=frm_res, text=" TDD DL Resources Configuration for Slice "+str(self.sl_number+1),fg = "blue",bg = "white",font = "Verdana 14 bold")
            else:
                lbl_res = tk.Label(master=frm_res, text=" TDD UL Resources Configuration for Slice "+str(self.sl_number+1),fg = "blue",bg = "white",font = "Verdana 14 bold")

        else:
            if self.dl:
                if self.ul:
                    lbl_res = tk.Label(master=frm_res, text=" FDD DL Resources Configuration for Slice "+str(self.sl_number+1),fg = "blue",bg = "white",font = "Verdana 14 bold")
            else:
                lbl_res = tk.Label(master=frm_res, text=" FDD UL Resources Configuration for Slice "+str(self.sl_number+1),fg = "blue",bg = "white",font = "Verdana 14 bold")

        lbl_res.grid(row=0, column=0, sticky="w")    
        
        #### entry for the number of resources for this slice
        frm_nres = tk.Frame(master=self.window)
        self.ent_nres = tk.Entry(master=frm_nres, width=12)
        self.ent_nres.bind("<FocusOut>", self.validate_nres)
        lbl_nres = tk.Label(master=frm_nres, text="Number of resources \nof slice "+str(self.sl_number+1))
        self.ent_nres.grid(row=1, column=0, sticky="e")
        lbl_nres.grid(row=0, column=0, sticky="w")
        self.ent_nres.insert(tk.END,self.nresdl)

        #### type of resources for DL
        frm_name = tk.Frame(master=self.window)
        self.ent_name = tk.Entry(master=frm_name, width=12)
        lbl_name = tk.Label(master=frm_name, text="Type of resource")
        self.ent_name.grid(row=1, column=0, sticky="e")
        lbl_name.grid(row=0, column=0, sticky="w")
        self.ent_name.insert(tk.END,self.namedl)
        
        ###entry for symbols per slot used for dl
        frm_sym = dFrame(master=self.window)
        self.ent_sym = tk.Entry(master=frm_sym, width=12)
        self.ent_sym.bind("<FocusOut>", self.validate_nsym)
        lbl_sym = tk.Label(master=frm_sym, text="Symbols per \n slot (max 14).")
        self.ent_sym.grid(row=1, column=0, sticky="e")
        lbl_sym.grid(row=0, column=0, sticky="w")
        self.ent_sym.insert(tk.END,self.sym_slot)


        if self.ul and self.dl and self.tdd:
            frm_sym.enable()
        else:
            frm_sym.disable()

        ##################  UL ################################
        frm_res1 = dFrame(master=self.window)
        lbl_res1 = tk.Label(master=frm_res1, text=" FDD UL Resources Configuration",fg = "blue",bg = "white",font = "Verdana 14 bold")
        lbl_res1.grid(row=0, column=0, sticky="w")    

        frm_nres1 =dFrame(master=self.window)
        self.ent_nres1 = tk.Entry(master=frm_nres1, width=12)
        self.ent_nres.bind("<FocusOut>", self.validate_nres)
        lbl_nres1 = tk.Label(master=frm_nres1, text="Number of resources")
        self.ent_nres1.grid(row=1, column=0, sticky="e")
        lbl_nres1.grid(row=0, column=0, sticky="w")
        self.ent_nres1.insert(tk.END,self.nresul)

        
        frm_name1 = dFrame(master=self.window)
        self.ent_name1 = tk.Entry(master=frm_name1, width=12)
        lbl_name1 = tk.Label(master=frm_name1, text="Type of resource")
        self.ent_name1.grid(row=1, column=0, sticky="e")
        lbl_name1.grid(row=0, column=0, sticky="w")
        self.ent_name1.insert(tk.END,self.nameul)

        frm_sym1 = dFrame(master=self.window)
        self.ent_sym1 = tk.Entry(master=frm_sym1, width=12)
        lbl_sym1 = tk.Label(master=frm_sym1, text="Symbols per \n slot (max 14).")
        self.ent_sym1.grid(row=1, column=0, sticky="e")
        lbl_sym1.grid(row=0, column=0, sticky="w")
        self.ent_sym1.insert(tk.END,str(14-self.sym_slot))

        frm_res1.disable()
        frm_nres1.disable()
        frm_name1.disable()
        frm_sym1.disable()
        if self.ul and self.dl and not self.tdd:
            frm_res1.enable()
            frm_nres1.enable()
            frm_name1.enable()
            frm_sym1.enable()
            self.ent_sym1.delete(0, 'end')
            self.ent_sym1.insert(0,str(14))
            frm_sym1.disable()


        ###################################################
        frm_save = tk.Frame(master=self.window)
        save = tk.Button(master=frm_save, text="Save \nConfiguration",font=font, compound=tk.CENTER, command=self.save)
        save.grid(row=0, column=0, columnspan=1, sticky='EWNS') #padx=10)
 

        frm_res.grid(row=0, column=1, padx=10)
        frm_nres.grid(row=1, column=1, padx=10)
        frm_name.grid(row=1, column=0, padx=10)
        frm_sym.grid(row=2, column=0, padx=10)

        frm_res1.grid(row=3, column=1, padx=10)
        frm_nres1.grid(row=4, column=1, padx=10)
        frm_name1.grid(row=4, column=0, padx=10)
        frm_sym1.grid(row=5, column=0, padx=10)
        
        frm_save.grid(row=6, column=2, padx=10)
     
                
 

