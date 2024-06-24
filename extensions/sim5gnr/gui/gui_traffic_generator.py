#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is a gui for the configuration of the antenna.

@author: pablo belzarena
"""
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont
from dframe import dFrame

class AppTrGen():
    """ This class is the form for the configuration of the antenna. """

    def __init__(self,window,function_cbk,title,name =["fixed"],uegr=0,inter_arrival =[1], pkt_size=[300], size_dist = ["Exponential"], burst_size=[1],nugr = 1,dl = True,ul =False,keep_pkts = [False],max_len = [0],last_k = [100] ):
        """The constructor of the AppAntenna Class.
        
        @type window: tkinter.Tk() window.
        @param window: The main window of this form.
        @type function_cbk: python function.
        @param function_cbk: The callback function to return when the user press OK button.
        @type title: string
        @param title: The title of the form.
       """
        self.window  = window
        """ The main window of this form.""" 
        self.window.title(title)
        """ The title of this form. """ 
        self.function_cbk = function_cbk
        """ The callback function to return when the user press OK button. """ 
        self.ugr =uegr
        self.nugr = nugr
        self.l_ugr =[]
        for i in range(nugr):
            self.l_ugr.append("UG-"+str(i+1))
        self.tr_type = name
        self.trgen_list = ["fixed", "poisson"]
        self.inter_arrival = inter_arrival
        self.pkt_size = pkt_size
        self.size_dist = size_dist
        self.pkt_dist_list = ["Fixed","Exponential"]
        self.burst_size = burst_size
        self.dl = dl
        self.ul = ul
        self.keep_pkt = keep_pkts
        self.max_len = max_len
        self.last_k = last_k
        self.gui_configuration()
    
    
    def saveall(self):      
        if self.validate_inter_arrival() and self.validate_pkt_size()and self.validate_burst_size() and  self.validate_inter_arrival1() and self.validate_pkt_size1() and self.validate_burst_size1() and self.validate_max_len() and self.validate_last_k() :
                                    
            self.function_cbk(self.tr_type,self.inter_arrival,self.pkt_size,self.burst_size,self.size_dist, self.max_len, self.last_k,self.keep_pkt)
  
   
    def ugr_selection_changed(self,event):
        """ Function called when the numerology changes. It display the numerology information.
        """
        ugr = self.ugr
        self.ugr = int(self.combo_sls.current())
        self.update()
        # self.sl_number = int(self.combo_sls.current())
        # print(self.numerology[self.sl_number])
        # self.combo_num.current(self.numerology[self.sl_number])

        # self.update_listbox()
        return
         
    def validate_inter_arrival(self,event:tk.Event=None):
        val1 = self.ent_inter_arrival.get()
        try:
            self.inter_arrival[self.ugr]=float(val1)
        except ValueError:
            tk.messagebox.showinfo(message="The number must be an float", title="Error")
            self.ent_inter_arrival.delete(0, 'end')
            self.ent_inter_arrival.insert(0,str(self.inter_arrival[self.ugr]))  
            self.ent_inter_arrival.focus()
            return False
        return True        

    def validate_pkt_size(self,event:tk.Event=None):
        val2 = self.ent_pkt_size.get()
        try:
            self.pkt_size[self.ugr]=int(val2)
        except ValueError:
            tk.messagebox.showinfo(message="The number must be an integer", title="Error")
            self.ent_pkt_size.delete(0, 'end')
            self.ent_pkt_size.insert(0,str(self.pkt_size[self.ugr]))  
            self.ent_pkt_size.focus()
            return False
        return True        


    def validate_burst_size(self,event:tk.Event=None):
        val = self.ent_burst_size.get()
        try:            
            self.burst_size[self.ugr] = int(val)
        except ValueError:
            tk.messagebox.showinfo(message="The burst size must be an integer", title="Error")
            self.ent_burst_size.delete(0, 'end')
            self.ent_burst_size.insert(0,str(self.burst_size[self.ugr]))  
            self.ent_burst_size.focus()
            return False
        return True        

    def validate_inter_arrival1(self,event:tk.Event=None):
        val1 = self.ent_inter_arrival1.get()
        try:
            self.inter_arrival[self.ugr]=float(val1)
        except ValueError:
            tk.messagebox.showinfo(message="The number must be an float", title="Error")
            self.ent_inter_arrival1.delete(0, 'end')
            self.ent_inter_arrival1.insert(0,str(self.inter_arrival[self.ugr]))  
            self.ent_inter_arrival1.focus()
            return False
        return True        

    def validate_pkt_size1(self,event:tk.Event=None):
        val2 = self.ent_pkt_size1.get()
        try:
            self.pkt_size[self.ugr]=int(val2)
        except ValueError:
            tk.messagebox.showinfo(message="The number must be an integer", title="Error")
            self.ent_pkt_size1.delete(0, 'end')
            self.ent_pkt_size1.insert(0,str(self.pkt_size[self.ugr]))  
            self.ent_pkt_size1.focus()
            return False
        return True        


    def validate_burst_size1(self,event:tk.Event=None):
        val = self.ent_burst_size1.get()
        try:
            
            self.burst_size[self.ugr] = int(val)

        except ValueError:
            tk.messagebox.showinfo(message="The burst size must be an integer", title="Error")
            self.ent_burst_size1.delete(0, 'end')
            self.ent_burst_size1.insert(0,str(self.burst_size[self.ugr]))  
            self.ent_burst_size1.focus()
            return False
        return True   

    def validate_max_len(self,event:tk.Event=None):
        val = self.ent_max_len.get()
        try:
            
            self.max_len[self.ugr] = int(val)
            if self.max_len[self.ugr] < 0 :
                tk.messagebox.showinfo(message="The max length must be an integer greater than 0", title="Error")
                self.ent_max_len.delete(0, 'end')
                self.ent_max_len.insert(0,str(0))  
                self.ent_max_len.focus()
                return False
        except ValueError:
            tk.messagebox.showinfo(message="The max length must be an integer", title="Error")
            self.ent_max_len.delete(0, 'end')
            self.ent_max_len.insert(0,str(self.burst_size[self.ugr]))  
            self.ent_max_len.focus()
            return False
        return True        

    def validate_last_k(self,event:tk.Event=None):
        val = self.ent_last_k.get()
        try:
            
            self.last_k[self.ugr] = int(val)
            if self.last_k[self.ugr] < 0 :
                tk.messagebox.showinfo(message="The value k must be an integer greater than 0", title="Error")
                self.ent_last_k.delete(0, 'end')
                self.ent_last_k.insert(0,str(100))  
                self.ent_last_k.focus()
                return False
        except ValueError:
            tk.messagebox.showinfo(message="The value k must be an integer", title="Error")
            self.ent_last_k.delete(0, 'end')
            self.ent_last_k.insert(0,str(self.last_k[self.ugr]))  
            self.ent_last_k.focus()
            return False
        return True        

         
    def gui_configuration(self):
        """This method builds the main form to enter the data
        
        """    
        width, height = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        self.__FONT_SIZE = int(width /1.25/ 1920*25)
        font = tkfont.Font(family="Helvetica", size=self.__FONT_SIZE, weight = tkfont.BOLD)
        self.window.rowconfigure(10, minsize=50, weight=1)
        self.window.columnconfigure([0, 1, 2, 3], minsize=50, weight=1)       
        

        frm_trgen = tk.Frame(master=self.window,relief=tk.SUNKEN)
        lbl_trgen = tk.Label(master=frm_trgen, text=" Traffic Generators Configuration",fg = "blue",bg = "white",font = "Verdana 14 bold")
        lbl_trgen.grid(row=0, column=0, sticky="w")    
        
        frm_name = tk.Frame(master=self.window,relief=tk.SUNKEN)
        self.combo_sls = ttk.Combobox(master=frm_name, width=12,state="readonly",values=self.l_ugr)
        self.combo_sls.current(self.ugr)
        self.combo_sls.bind("<<ComboboxSelected>>", self.ugr_selection_changed)
        self.combo_sls.grid(row=1, column=0, sticky="w")
        lbl_name = tk.Label(master=frm_name, text="Select User Group \n to configure it traffic generator",fg = "blue",bg = "white",font = "Verdana 12 bold")
        lbl_name.grid(row=0, column=0, sticky="w")
    
    
  
    
        frm_trgen_types = tk.Frame(master=self.window,relief=tk.SUNKEN)
        self.combo_trgen_types = ttk.Combobox(master=frm_trgen_types, width=12,state="readonly",values=self.trgen_list)
        self.combo_trgen_types.current(self.trgen_list.index(self.tr_type[self.ugr]))
        self.combo_trgen_types.bind("<<ComboboxSelected>>", self.trgen_types_selection_changed)
        lbl_trgen_types = tk.Label(master=frm_trgen_types, text="Traffic Generator type",fg = "blue",bg = "white",font = "Verdana 12 bold")
        self.combo_trgen_types.grid(row=1, column=0, sticky="w")
        lbl_trgen_types.grid(row=0, column=0, sticky="w")
        
        self.frm_fixed = dFrame(master=self.window,relief=tk.SUNKEN)
        self.ent_inter_arrival = tk.Entry(master=self.frm_fixed, width=12)
        self.ent_inter_arrival.bind("<FocusOut>", self.validate_inter_arrival)
        lbl_inter_arrival = tk.Label(master=self.frm_fixed, text="Inter arrival time (ms)")
        lbl_inter_arrival.grid(row=0, column=0, sticky="e")
        self.ent_inter_arrival.grid(row=0, column=1, sticky="e")
        self.ent_inter_arrival.insert(tk.END,str(self.inter_arrival[self.ugr]))
        
        self.ent_pkt_size = tk.Entry(master=self.frm_fixed, width=12)
        self.ent_pkt_size.bind("<FocusOut>", self.validate_pkt_size)
        self.lbl_pkt_size = tk.Label(master=self.frm_fixed, text=" Packet size (bytes)")
        self.lbl_pkt_size.grid(row=0, column=2, sticky="e")
        self.ent_pkt_size.grid(row=0, column=3, sticky="e")
        self.ent_pkt_size.insert(tk.END,str(self.pkt_size[self.ugr]))

        self.ent_burst_size = tk.Entry(master=self.frm_fixed, width=12)
        self.ent_burst_size.bind("<FocusOut>", self.validate_burst_size)
        lbl_burst_size = tk.Label(master=self.frm_fixed, text=" Burst size (packets)")
        lbl_burst_size.grid(row=0, column=4, sticky="e")
        self.ent_burst_size.grid(row=0, column=5, sticky="e")
        self.ent_burst_size.insert(tk.END,str(self.burst_size[self.ugr]))

        self.frm_poisson = dFrame(master=self.window,relief=tk.SUNKEN)

        self.combo_pkt_dist = ttk.Combobox(master=self.frm_poisson, width=12,state="readonly",values=self.pkt_dist_list)
        self.combo_pkt_dist.current(self.pkt_dist_list.index(self.size_dist[self.ugr]))
        self.combo_pkt_dist.bind("<<ComboboxSelected>>", self.pkt_dist_selection_changed)
        self.lbl_pkt_dist = tk.Label(master=self.frm_poisson, text="Packet size distribution")
        self.combo_pkt_dist.grid(row=3, column=2, sticky="w")
        self.lbl_pkt_dist.grid(row=3, column=1, sticky="w")

        
        self.ent_inter_arrival1 = tk.Entry(master=self.frm_poisson, width=12)
        self.ent_inter_arrival1.bind("<FocusOut>", self.validate_inter_arrival1)
        lbl_inter_arrival1 = tk.Label(master=self.frm_poisson, text="Mean value of \ninter arrival time (ms)")
        lbl_inter_arrival1.grid(row=2, column=0, sticky="e")
        self.ent_inter_arrival1.grid(row=2, column=1, sticky="e")
        self.ent_inter_arrival1.insert(tk.END,str(self.inter_arrival[self.ugr]))



        self.ent_pkt_size1 = tk.Entry(master=self.frm_poisson, width=12)
        self.ent_pkt_size1.bind("<FocusOut>", self.validate_pkt_size1)
        self.lbl_pkt_size1 = tk.Label(master=self.frm_poisson, text=" Packet size (bytes)")
        self.lbl_pkt_size1.grid(row=2, column=2, sticky="e")
        self.ent_pkt_size1.grid(row=2, column=3, sticky="e")
        self.ent_pkt_size1.insert(tk.END,str(self.pkt_size[self.ugr]))

        self.ent_burst_size1 = tk.Entry(master=self.frm_poisson, width=12)
        self.ent_burst_size1.bind("<FocusOut>", self.validate_burst_size1)
        lbl_burst_size1 = tk.Label(master=self.frm_poisson, text=" Burst size (packets)")
        lbl_burst_size1.grid(row=2, column=4, sticky="e")
        self.ent_burst_size1.grid(row=2, column=5, sticky="e")
        self.ent_burst_size1.insert(tk.END,str(self.burst_size[self.ugr]))
      
        self.frm_queue = tk.Frame(master=self.window,relief=tk.SUNKEN)
        lbl_queue = tk.Label(master=self.frm_queue, text=" Packet queue parameters ",fg = "blue",bg = "white",font = "Verdana 12 bold")
        lbl_queue.grid(row=1, column=0, sticky="w")

        self.ent_max_len = tk.Entry(master=self.frm_queue, width=12)
        self.ent_max_len.bind("<FocusOut>", self.validate_max_len)
        lbl_max_len = tk.Label(master=self.frm_queue, text=" Maximum queue length \n 0 for no limit ")
        lbl_max_len.grid(row=2, column=0, sticky="e")
        self.ent_max_len.grid(row=2, column=1, sticky="e")
        self.ent_max_len.insert(tk.END,str(self.max_len[self.ugr]))

        self.ent_last_k = tk.Entry(master=self.frm_queue, width=12)
        self.ent_last_k.bind("<FocusOut>", self.validate_last_k)
        lbl_last_k = tk.Label(master=self.frm_queue, text=" Keep last k values of data ")
        lbl_last_k.grid(row=2, column=2, sticky="e")
        self.ent_last_k.grid(row=2, column=3, sticky="e")
        self.ent_last_k.insert(tk.END,str(self.last_k[self.ugr]))

        self.l_keep_pkt = [True,False]
        self.combo_keep_pkt = ttk.Combobox(master=self.frm_queue, width=12,state="readonly",values=self.l_keep_pkt)
        self.combo_keep_pkt.current(self.l_keep_pkt.index(self.keep_pkt[self.ugr]))
        self.combo_keep_pkt.bind("<<ComboboxSelected>>", self.keep_pkt_selection_changed)
        self.lbl_keep_pkt = tk.Label(master=self.frm_queue, text="Keep all information \n of all packets")
        self.combo_keep_pkt.grid(row=3, column=2, sticky="w")
        self.lbl_keep_pkt.grid(row=3, column=1, sticky="w")

                    
        ###################################################
        frm_save = tk.Frame(master=self.window,relief=tk.SUNKEN)
        save = tk.Button(master=frm_save, text="Save Configuration \n of all UGs \n and close",font=font, compound=tk.CENTER, command=self.saveall)
        save.grid(row=0, column=0, columnspan=1, sticky='EWNS') #padx=10)


    

        frm_name.grid(row=1, column=0, padx=10,pady=10)
        frm_trgen.grid(row=0, column=1, padx=10,pady=10)
        frm_trgen_types.grid(row=2, column=1, padx=10,pady=10)
        self.frm_poisson.grid(row=3, column=1, padx=10,pady=20)
        self.frm_fixed.grid(row=3, column=1, padx=10,pady=20)
        self.frm_queue.grid(row=4, column=1, padx=10,pady=20)

        frm_save.grid(row=5, column=3, padx=10,pady=20)
        
        if self.combo_trgen_types.get() == "fixed":
                self.frm_fixed.show(row=3, column=1, padx=10)
                self.frm_poisson.hide()
        if self.combo_trgen_types.get()  == "poisson":
            self.frm_poisson.show(row=3, column=1, padx=10)
            self.frm_fixed.hide()



        
        
    def pkt_dist_selection_changed(self,event):
        selection=self.combo_pkt_dist.get()
        # self.frm_random.enable()
        # self.frm_fixed.disable()
        self.size_dist[self.ugr] = selection
        if selection == "fixed":
            self.lbl_pkt_size1.config(text="Packet size (bytes)")
            
        if selection == "exponential":
            self.lbl_pkt_size1.config(text="Mean packet size (bytes)")

    def keep_pkt_selection_changed(self,event):
        selection=self.combo_keep_pkt.get()
        self.keep_pkt[self.ugr] = eval(selection)



    def trgen_types_selection_changed(self,event):
        selection=self.combo_trgen_types.get()
        self.tr_type[self.ugr] = selection
        # self.frm_random.enable()
        # self.frm_fixed.disable()
        if selection == "fixed":
            self.frm_fixed.show(row=3, column=1, padx=10)
            self.frm_poisson.hide()
        if selection == "poisson":
            self.frm_poisson.show(row=3, column=1, padx=10)
            self.frm_fixed.hide()

    def update(self):
        self.ent_inter_arrival.delete(0, 'end')
        self.ent_pkt_size.delete(0, 'end')
        self.ent_burst_size.delete(0, 'end')
        self.ent_inter_arrival1.delete(0, 'end')
        self.ent_pkt_size1.delete(0, 'end')
        self.ent_burst_size1.delete(0, 'end')
        self.ent_max_len.delete(0, 'end')
        self.ent_last_k.delete(0, 'end')


        self.combo_trgen_types.current(self.trgen_list.index(self.tr_type[self.ugr]))
        self.ent_inter_arrival.insert(tk.END,str(self.inter_arrival[self.ugr]))
        self.ent_pkt_size.insert(tk.END,str(self.pkt_size[self.ugr]))
        self.ent_burst_size.insert(tk.END,str(self.burst_size[self.ugr]))
        self.combo_pkt_dist.current(self.pkt_dist_list.index(self.size_dist[self.ugr]))
        self.ent_inter_arrival1.insert(tk.END,str(self.inter_arrival[self.ugr]))
        self.ent_pkt_size1.insert(tk.END,str(self.pkt_size[self.ugr]))
        self.ent_burst_size1.insert(tk.END,str(self.burst_size[self.ugr]))
        self.ent_max_len.insert(tk.END,str(self.max_len[self.ugr]))
        self.ent_last_k.insert(tk.END,str(self.last_k[self.ugr]))
        self.combo_keep_pkt.current(self.l_keep_pkt.index(self.keep_pkt[self.ugr]))

        if self.combo_trgen_types.get() == "fixed":
                self.frm_fixed.show(row=3, column=1, padx=10)
                self.frm_poisson.hide()
        if self.combo_trgen_types.get()  == "poisson":
            self.frm_poisson.show(row=3, column=1, padx=10)
            self.frm_fixed.hide()




     
        
