#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is a gui for the configuration of the slice.

"""
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont
from extensions.sim5gnr.tables import load_numerology_table

class AppSlice():
    """ This class is the form for the configuration of the slice. """

    def __init__(self,window,function_cbk,title,name="SL-1",sl_number =1,numerology=[2],sl_ugr=[[0,1]],names_uegrs=["UG-1","UG-2"],num_slices=1,num_uegr=2,ugr_not_assigned=[]):
        """The constructor of the AppSlice Class.
        
        @type window: tkinter.Tk() window.
        @param window: The main window of this form.
        @type function_cbk: python function.
        @param function_cbk: The callback function to return when the user press OK button.
        @type title: string
        @param title: The title of the form.
        @type name: string
        @param name: The name of the slice.
        @type sl_number: string
        @param sl_number: The number of the slice.
        @type numerology: int
        @param numerology: The numerology of the slice.        
        
       """
        self.window  = window
        """ The main window of this form.""" 
        self.window.title(title)
        """ The title of this form. """ 
        self.function_cbk = function_cbk
        """ The callback function to return when the user press OK button. """ 
        self.name = name
        """  The name of the slice. """ 
        self.sl_number = sl_number
        self.numerology = numerology
        self.sl_ugr = sl_ugr
        self.names_uegrs = names_uegrs
        self.num_slices = num_slices
        self.l_slices = []
        for i in range(self.num_slices):
            self.l_slices.append("SL-"+str(i+1))
        self.num_uegr =num_uegr
        self.ugr_notassigned =ugr_not_assigned
        self.gui_configuration()
    
    def num_selection_changed(self,event):
        """ Function called when the numerology changes. It display the numerology information.
        """
        self.numerology[self.sl_number]= int(self.combo_num.get())
        txt = "Subcarrier spacing: " + str(self.num_table[int(self.combo_num.get())]["sub_carrier_spacing"])+ " kHz\nSlot duration: "+ str(self.num_table[int(self.combo_num.get())]["slot_duration"]) + " ms "
        self.lbl_num_selected.config(text = txt)

    def sls_selection_changed(self,event):
        """ Function called when the numerology changes. It display the numerology information.
        """
        self.sl_number = int(self.combo_sls.current())
        self.combo_num.current(self.numerology[self.sl_number])

        self.update_listbox()

    
    def save(self):
        """ Function called when the user press the save button.
        
        """
        self.function_cbk(self.numerology,self.num_uegr,self.sl_ugr,self.ugr_notassigned)
            
    def add_ugr(self):
        """ Function called when the user press the add user group button.
        
        """
        self.num_uegr +=1
        self.names_uegrs.append("UG-"+str(self.num_uegr))
        self.l_ugr.append(self.names_uegrs[self.num_uegr-1])
        self.ugr_notassigned.append(self.num_uegr-1)
        self.update_listbox()
        
         
    def gui_configuration(self):
        """This method builds the main form to enter the data
        
        """    
        ###general definitions of the form
        width, height = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        self.__FONT_SIZE = int(width /1.25/ 1920*25)
        font = tkfont.Font(family="Helvetica", size=self.__FONT_SIZE, weight = tkfont.BOLD)
        self.window.rowconfigure(10, minsize=50, weight=1)
        self.window.columnconfigure([0, 1, 2, 3], minsize=50, weight=1)       

        # label for enter the titel of the form
        frm_slice = tk.Frame(master=self.window)
        lbl_slice = tk.Label(master=frm_slice, text="Slice Configuration",fg = "blue",bg = "white",font = "Verdana 14 bold")
        lbl_slice.grid(row=0, column=0, sticky="w")    
        
        #load numerology table
        self.num_table = load_numerology_table()
        #the list of possibles numerologies
        numerology_list = [0,1,2,3,4,5,6]
        
        #Combo box with the list of possibles numerologies
        frm_num = tk.Frame(master=self.window)
        self.combo_num = ttk.Combobox(master=frm_num, width=12,state="readonly",values=numerology_list)
        self.combo_num.current(self.numerology[self.sl_number])
        self.combo_num.bind("<<ComboboxSelected>>", self.num_selection_changed)
        lbl_num = tk.Label(master=frm_num, text="Slice Numerology")
        self.combo_num.grid(row=1, column=0, sticky="w")
        lbl_num.grid(row=0, column=0, sticky="w")
        # when the user selects the numerology, its information is displayed
        txt = "Subcarrier spacing: " + str(self.num_table[int(self.combo_num.get())]["sub_carrier_spacing"])+ " kHz\nSlot duration: "+ str(self.num_table[int(self.combo_num.get())]["slot_duration"]) + " ms "
        self.lbl_num_selected =tk.Label(master=frm_num, text=txt)
        self.lbl_num_selected.grid(row=2, column=0, sticky="w")

        ##Entry for the name of the slice, it is disable the user cannot edit it.
        frm_name = tk.Frame(master=self.window)
        self.combo_sls = ttk.Combobox(master=frm_name, width=12,state="readonly",values=self.l_slices)
        self.combo_sls.current(self.sl_number)
        self.combo_sls.bind("<<ComboboxSelected>>", self.sls_selection_changed)
        self.combo_sls.grid(row=1, column=0, sticky="w")

        #self.ent_name = tk.Entry(master=frm_name, width=12)
        lbl_name = tk.Label(master=frm_name, text="Name of the Slice")
        #self.ent_name.grid(row=1, column=0, sticky="e")
        lbl_name.grid(row=0, column=0, sticky="w")
        #self.ent_name.insert(tk.END,self.name)
        #self.ent_name.configure(state= 'disable', disabledbackground='white', disabledforeground='red')

        frm_ugr = tk.Frame(master=self.window)
        self.listbox_ugr = tk.Listbox(frm_ugr,selectmode="multiple", exportselection=0,height=10,width=20)
        self.listbox_ugr.grid(row=1, column=0, padx=10)
        self.l_ugr = self.names_uegrs.copy()
        for sl in range(self.num_slices):
            for ugr in self.sl_ugr[sl]:
                self.l_ugr[ugr] = self.l_ugr[ugr]+"-- SL-"+str(sl+1)
        for values in self.l_ugr:
            self.listbox_ugr.insert(tk.END,values)
            
               
        for sl in range(self.num_slices):
            for ugr in self.sl_ugr[sl]:
                if sl == self.sl_number:
                    self.listbox_ugr.select_set(ugr)
                else:
                    self.disable_item(ugr)
        self.listbox_ugr.bind("<<ListboxSelect>>",self.no_selection)

        
        lbl_ugr = tk.Label(master=frm_ugr, text="User groups associated")
        lbl_ugr.grid(row=0, column=0, sticky="w")
        #itemconfig(index, fg="gray")
          ## save button        
        frm_add_ugr = tk.Frame(master=self.window)
        add_ugr = tk.Button(master=frm_add_ugr, text="Add User Group",font=font, compound=tk.CENTER, command=self.add_ugr)
        add_ugr.grid(row=0, column=0, columnspan=1, sticky='EWNS') #padx=10)
 
        
        ## save button        
        frm_save = tk.Frame(master=self.window)
        save = tk.Button(master=frm_save, text="Save \nConfiguration",font=font, compound=tk.CENTER, command=self.save)
        save.grid(row=0, column=0, columnspan=1, sticky='EWNS') #padx=10)
 
        ### place of the different objects in the form
        frm_slice.grid(row=0, column=1, padx=10)
        frm_num.grid(row=2, column=0, padx=10)
        frm_ugr.grid(row=2, column=1, padx=10)

        frm_name.grid(row=1, column=0, padx=10)
        frm_add_ugr.grid(row=3, column=1, padx=10)
        frm_save.grid(row=4, column=2, padx=10)


    def disable_item(self,index):
        self.listbox_ugr.itemconfig(index, fg="gray")

    def no_selection(self, event):
        l= self.listbox_ugr.curselection()
    
        for index in l:
            if index in self.ugr_notassigned:
                self.sl_ugr[self.sl_number].append(index)
                self.ugr_notassigned.remove(index)
                self.l_ugr[index]= self.names_uegrs[index]+"-- SL-"+str(self.sl_number+1)
            if index not in self.sl_ugr[self.sl_number]:
                self.listbox_ugr.selection_clear(index)
        
        for index in self.sl_ugr[self.sl_number]:
            if index  not in l:
                self.l_ugr[index]= self.names_uegrs[index]
                self.sl_ugr[self.sl_number].remove(index)
                self.ugr_notassigned.append(index)

        self.update_listbox()

    def update_listbox(self):
        self.listbox_ugr.delete('0', 'end')

        for values in self.l_ugr:
            self.listbox_ugr.insert(tk.END,values)
      
        for sl in range(self.num_slices):
            for ugr in self.sl_ugr[sl]:
                if sl == self.sl_number:
                    self.listbox_ugr.select_set(ugr)        
                else:
                    self.disable_item(ugr)
   
 

