#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is a gui for the configuration of the channel.

"""
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont
import gui_user_message as gum
from tkinter import filedialog
from dframe import dFrame

class AppChannel():
    """ This class is the form for the configuration of the channel. """

    def __init__(self,window,function_cbk,title,name="random or fixed",file =None, chan_mode="random", val_1=-100, val_2=100,loss_prob = 0):
        """The constructor of the AppChannel Class.
        
        @type window: tkinter.Tk() window.
        @param window: The main window of this form.
        @type function_cbk: python function.
        @param function_cbk: The callback function to return when the user press OK button.
        @type title: string
        @param title: The title of the form.
        @type name: string
        @param name: The type of channel.
        @type file: string
        @param file: The file in case of file channel.
        @type chan_mode: string
        @param chan_mode: the channel mode: fixed or random.
        @type val1_: float
        @param val_1: The min value of the snr for fixed or random channel.      
        @type val_2: float
        @param val_2: The max value of the snr for fixed or random channel.      

        
       """
        self.window  = window
        """ The main window of this form.""" 
        self.window.title(title)
        """ The title of this form. """ 
        self.function_cbk = function_cbk
        """ The callback function to return when the user press OK button. """ 
        self.name = name
        self.channels_list = ["random or fixed", "file"]
        self.channel = self.channels_list.index(name)
        self.file =file
        self.mode = chan_mode
        self.val1 = val_1
        self.val2 = val_2
        self.loss_prob = loss_prob
        self.gui_configuration()
    
    
    def save(self):      
        """ This method is called when the user press the save button.
        
        """
        if self.validate_val1() and self.validate_val2() and self.validate_val() and self.validate_loss():
            val1 = 0
            if self.combo_chantypes.get() == "random or fixed":
                if self.mode == "Random":
                    val1 = int(self.ent_val1.get())
                else:
                    val1 = int(self.ent_val.get())
            self.function_cbk(self.combo_chantypes.get(),self.mode,self.file,val1 , int(self.ent_val2.get()),float(self.ent_loss.get()))

    def validate_loss(self,event:tk.Event=None):
        """ This method is called when the val1 entry changes.
        """
        val1 = self.ent_loss.get()
        try:
            aux = float(val1)
            if aux < 0 or aux > 1:
                tk.messagebox.showinfo(message="The number must be an float and greater than 0 and lower than 1", title="Error")
                self.ent_loss.delete(0, 'end')
                self.ent_loss.insert(0,"0.0")  
                self.ent_loss.focus()
                return False
     
        except ValueError:
            tk.messagebox.showinfo(message="The number must be an float", title="Error")
            self.ent_loss.delete(0, 'end')
            self.ent_loss.insert(0,"0.0")  
            self.ent_loss.focus()
            return False
        return True        

            
    def validate_val1(self,event:tk.Event=None):
        """ This method is called when the val1 entry changes.
        """
        val1 = self.ent_val1.get()
        try:
            int(val1)
        except ValueError:
            tk.messagebox.showinfo(message="The number must be an integer", title="Error")
            self.ent_val1.delete(0, 'end')
            self.ent_val1.insert(0,"-100")  
            self.ent_val1.focus()
            return False
        return True        

    def validate_val2(self,event:tk.Event=None):
        """ This method is called when the val2 entry changes.
        """
        val2 = self.ent_val2.get()
        try:
            int(val2)
        except ValueError:
            tk.messagebox.showinfo(message="The number must be an integer", title="Error")
            self.ent_val2.delete(0, 'end')
            self.ent_val2.insert(0,"100")  
            self.ent_val2.focus()
            return False
        return True        


    def validate_val(self,event:tk.Event=None):
        """ This method is called when the val entry changes.
        """
        val = self.ent_val.get()
        try:
            int(val)
        except ValueError:
            tk.messagebox.showinfo(message="The number must be an integer", title="Error")
            self.ent_val.delete(0, 'end')
            self.ent_val.insert(0,"100")  
            self.ent_val.focus()
            return False
        return True        

         
    def gui_configuration(self):
        """This method builds the main form to enter the data
        
        """    
        ### general definitions of the form
        width, height = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        self.__FONT_SIZE = int(width /1.25/ 1920*25)
        font = tkfont.Font(family="Helvetica", size=self.__FONT_SIZE, weight = tkfont.BOLD)
        self.window.rowconfigure(10, minsize=50, weight=1)
        self.window.columnconfigure([0, 1, 2, 3], minsize=50, weight=1)       
        #### Title
        frm_chan = tk.Frame(master=self.window)
        lbl_chan = tk.Label(master=frm_chan, text=" Channel Configuration",fg = "blue",bg = "white",font = "Verdana 14 bold")
        lbl_chan.grid(row=0, column=0, sticky="w")    
        ### Combobox for selecting the channel type
        frm_chantypes = tk.Frame(master=self.window)
        self.combo_chantypes = ttk.Combobox(master=frm_chantypes, width=12,state="readonly",values=self.channels_list)
        self.combo_chantypes.current(self.channel)
        self.combo_chantypes.bind("<<ComboboxSelected>>", self.chan_selection_changed)
        lbl_chantypes = tk.Label(master=frm_chantypes, text="Cahnnel type")
        self.combo_chantypes.grid(row=1, column=0, sticky="w")
        lbl_chantypes.grid(row=0, column=0, sticky="w")
  
        ### buttons for channel mode
        self.frm_rb_chtypes = dFrame(master=self.window)
        self.v = tk.IntVar(self.window)
        if self.mode == "Random":
            self.v.set(1) 
        else:
            self.v.set(2)
        self.rb_random = tk.Radiobutton(master=self.frm_rb_chtypes, text='Random',variable = self.v,value =1,command=self.cmd_random)
        self.rb_fixed = tk.Radiobutton(master=self.frm_rb_chtypes, text='Fixed',variable = self.v,value =2, command=self.cmd_fixed)
        self.rb_random.grid(row=1, column=1, sticky="w")
        self.rb_fixed.grid(row=2, column=1, sticky="w")
        lbl_rf = tk.Label(master=self.frm_rb_chtypes, text="Fixed or Random \nChannel model")
        lbl_rf.grid(row=0, column=1, sticky="e")

        ### Entry for val1. Minimum in random channel
        self.frm_random = dFrame(master=self.window)
        self.ent_val1 = tk.Entry(master=self.frm_random, width=12)
        self.ent_val1.bind("<FocusOut>", self.validate_val1)
        lbl_val1 = tk.Label(master=self.frm_random, text="Min value of \nthe channel state")
        lbl_val1.grid(row=0, column=0, sticky="e")
        self.ent_val1.grid(row=0, column=1, sticky="e")
        self.ent_val1.insert(tk.END,str(self.val1))
        ## Entry for val2. maximum in random channel
        self.ent_val2 = tk.Entry(master=self.frm_random, width=12)
        self.ent_val2.bind("<FocusOut>", self.validate_val2)
        lbl_val2 = tk.Label(master=self.frm_random, text="Max value of \nthe channel state")
        lbl_val2.grid(row=0, column=2, sticky="e")
        self.ent_val2.grid(row=0, column=3, sticky="e")
        self.ent_val2.insert(tk.END,str(self.val2))
        
        ### Entry for val. Value in case of a fixed value channel
        self.frm_fixed = dFrame(master=self.window)
        self.ent_val = tk.Entry(master=self.frm_fixed, width=12)
        self.ent_val.bind("<FocusOut>", self.validate_val)
        lbl_val = tk.Label(master=self.frm_fixed, text="value of the \nchannel state")
        lbl_val.grid(row=0, column=0, sticky="e")
        self.ent_val.grid(row=0, column=1, sticky="e")
        self.ent_val.insert(tk.END,str(self.val1))
        
        self.frm_loss = dFrame(master=self.window)
        self.ent_loss = tk.Entry(master=self.frm_loss, width=12)
        self.ent_loss.bind("<FocusOut>", self.validate_loss)
        lbl_loss = tk.Label(master=self.frm_loss, text="Value of the \nchannel loss probability")
        lbl_loss.grid(row=0, column=0, sticky="e")
        self.ent_loss.grid(row=0, column=1, sticky="e")
        self.ent_loss.insert(tk.END,str(self.loss_prob))
        
        
        #### Enable or disable parts of the form
        if self.mode=="Random":
            self.frm_random.show(row=2, column=1, padx=10)
            self.frm_fixed.hide()
        else:
            self.frm_fixed.show(row=2, column=1, padx=10)
            self.frm_random.hide()

        ##To select the file of the channel file
        self.frm_directory = dFrame(master=self.window)
        lbl_filech = tk.Label(master=self.frm_directory, text=" \nFile Channel ")
        lbl_filech.grid(row=0, column=1, sticky="e")
        aux0 = tk.Button(master=self.frm_directory, text="Select \n File", font=font, compound=tk.CENTER,command=self.cmd_directory)
        aux0.grid(row=1, column=1,columnspan=1, sticky='EWNS')
        self.lbl_file = tk.Label(master=self.frm_directory,wraplength=200, text=self.file)
        self.lbl_file.grid(row=2, column=0, columnspan=6,sticky="w")

            
        ###save button
        frm_save = tk.Frame(master=self.window)
        save = tk.Button(master=frm_save, text="Save \nConfiguration",font=font, compound=tk.CENTER, command=self.save)
        save.grid(row=0, column=0, columnspan=1, sticky='EWNS') #padx=10)
        
        ###Places of the objects in the form
        frm_chantypes.grid(row=1, column=1, padx=10)
        self.frm_rb_chtypes.grid(row=2, column=0, padx=10)
        frm_chan.grid(row=0, column=1, padx=10)
        self.frm_random.grid(row=2, column=1, padx=10)
        self.frm_fixed.grid(row=3, column=1, padx=10)
        self.frm_directory.grid(row=4, column=0, padx=10)         
        frm_save.grid(row=4, column=2, padx=10)
        self.frm_loss.grid(row=4, column=1, padx=10)
        
        ###Hide or show parts of the form.
        if self.combo_chantypes.get() == "random or fixed":
            if self.mode=="Random":
                self.frm_random.show(row=2, column=1, padx=10)
                self.frm_fixed.hide()
            else:
                self.frm_fixed.show(row=2, column=1, padx=10)
                self.frm_random.hide()
            self.frm_directory.hide()
        if self.combo_chantypes.get()  == "file":
            self.frm_directory.show(row=4, column=0, padx=10)   
            self.frm_rb_chtypes.hide()
            self.frm_random.hide()
            self.frm_fixed.hide()



    def cmd_random(self):
        "This method is called when the user selects channel mode: random"

        self.frm_random.show(row=2, column=1, padx=10)
        self.frm_fixed.hide()
        self.mode = "Random"
        self.v.set(1)
        
    def chan_selection_changed(self,event):
        """ This method is called when the type of channel changes.
        
        """
        selection=self.combo_chantypes.get()
        if selection == "random or fixed":
            self.frm_rb_chtypes.show(row=2, column=0, padx=10)
            if self.v.get() == 1:
                self.frm_fixed.hide()
                self.frm_random.show(row=2, column=1, padx=10)
            else:
                self.frm_fixed.show(row=2, column=1, padx=10)
                self.frm_random.hide()
            self.frm_directory.hide()  
            
        if selection == "file":
            self.frm_directory.show(row=4, column=0, padx=10)   
            self.frm_rb_chtypes.hide()
            self.frm_random.hide()
            self.frm_fixed.hide()
            self.lbl_file.config(text = self.file)




    def cmd_fixed(self):
        "This method is called when the user selects channel mode: fixed"
        self.frm_fixed.show(row=2, column=1, padx=10)
        self.frm_random.hide()
        self.mode = "Fixed"
        self.v.set(2)

    def cmd_directory(self):
        """ This method is called when the user select the file of the file channel. 
        
        """ 
        try:
            file_name = filedialog.askopenfilename(initialdir="./",title='Please select the source directory')

            if file_name is not None and file_name != "" :
                self.file = file_name
                self.lbl_file.config(text = file_name)
            else:
                gum.AppUserMsg('Error!','You must select a File. Please try again')
        except BaseException as error:
            gum.AppUserMsg('Exception occurred!','{}'.format(error)+ " try again please!" )



       
                        
         
        
