# -*- coding: utf-8 -*-

import tkinter as tk

class dFrame(tk.Frame):        
        """ Auxiliary class to enable and disable a tkinter frame. 
        
        """
        def enable(self, st='normal'):
            """ This method enable all components of a frame. 
            
            @type st: string
            @param st: the state to set the frame: 'normal' or 'disable'. Default: 'normal'.
            """
            for w in self.winfo_children():
                # change its state
                w.config(state = st)
                # and then recurse to process ITS children
                        #cstate(w)
        def disable(self):
            """ This method disable all components of a frame. 
            
            """
            self.enable('disabled')
            
        def show(self,row=2, column=1, padx=10):
            self.grid(row=row, column=column, padx=padx)
         
        # Hide the window
        def hide(self):
            self.grid_forget() 
         
      