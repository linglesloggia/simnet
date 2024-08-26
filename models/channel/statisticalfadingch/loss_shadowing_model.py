# -*- coding: utf-8 -*-
import numpy as np
class SimpleLossModel:
      
      """This class implements an extended Friis loss model scenario\
      where the loss is not cuadratic with the distance but depends as distance**order  
      """  
      
      def __init__(self,env,order): 
            """ The constructor of the extended Friis loss model scenario. Calls the parent class constructor.  \
            
            @type env: ChannelEnvironment .
            @param env: ChannelEnvironment object
            """ 
            self.env =env
            self._order = order
            """ The order of the exponent of the distance in the loss model. """ 
            self.loss = 0
            """ The path loss """
            
      def get_loss (self,distance,h_MS=1): 
            """ This method computes the path loss of the scenario using the Friis equation but with\
            the distance**order 
            @type distance: float.
            @param distance: The distance between the BS and MS positions.
            @type h_MS: float.
            @param h_MS: The MS antenna height. Default value 1.
            @return: -20*np.log10(3e8/4/np.pi/self.fcGHz/1e9) +10*np.log10( (distance)**self._brder)
            """ 
            self.loss = 0
            if distance > 0:
                self.loss = -20*np.log10(3e8/4/np.pi/self.env.fcGHz/1e9) +10*np.log10( (distance)**self._order)
            return self.loss
            
      def generate_correlated_shadow_vector(self,MS_pos,type_approx):
            """ This method given the shadowing grid and a MS position, estimates the shadowing for this point.\
        
            The method can use two approximation methods. The first one use the parameter of the closest point in the grid. \
            The second on interpolates between the two closests point in the grid.\
            @type MS_pos: 3D array or list.\
            @param MS_pos: the position of the mobile device in the scenario.\
            @type type_approx: int.\
            @param type_approx: The type of approximation. 0 for the closest point. 1 for interpolation.\
            @return: the LSP parameters for the MS_pos point in the scenario.\
            """ 
        
            grid_shadow = np.copy(self.env.grid_shadow)
            grid_shadow = grid_shadow* self.env.sigma_shadow
            return self.shadow_vector_position(grid_shadow,MS_pos,type_approx)
        
      def shadow_vector_position(self,gridLSP,MS_pos,type_approx):
            LSP_xy = np.zeros((self.env._paranum))
            absolute_val_array = np.abs(self.env.XY[0][0]- MS_pos[0])
            smallest_difference_index_x = absolute_val_array.argmin()
            closest_element_x = self.env.XY[0][0][smallest_difference_index_x]
            
            absolute_val_array = np.abs(self.env.XY[1][:,0]- MS_pos[1])
            smallest_difference_index_y = absolute_val_array.argmin()
            closest_element_y = self.env.XY[1][smallest_difference_index_y,0]
            if type_approx == 0:
                LSP_xy = gridLSP[smallest_difference_index_y,smallest_difference_index_x] 
            else:
                if smallest_difference_index_x < self.env.XY[0][0].size-1 and smallest_difference_index_y < self.env.XY[1][:,0].size-1:    
                    distance = np.sqrt((closest_element_x - MS_pos[0])**2 + (closest_element_y - MS_pos[1])**2 )
                    d_step = np.sqrt((closest_element_x -self.env.XY[0][0][smallest_difference_index_x]+1)**2 + (closest_element_y - self.env.XY[1][smallest_difference_index_y+1,0])**2) 
                    LSP_xy = (d_step -distance)/d_step*gridLSP[smallest_difference_index_y,smallest_difference_index_x] +  distance/d_step*gridLSP[smallest_difference_index_y+1,smallest_difference_index_x+1]
                else:
                    LSP_xy= gridLSP[smallest_difference_index_y,smallest_difference_index_x] 
            return LSP_xy

      def get_shadowing_db(self,MS_pos,type_approx):
            """ This method computes the shadowing value for the MS position, sets its values and return it.\
            
            @type MS_pos: 3D array or list.
            @param MS_pos: the position of the movil device in the scenario.
            @type type_approx: int.
            @param type_approx: The type of approximation used. 0 for the closest point in the grid. 1 for interpolating between closets points in the grid.
            @return: The shadowing valu for the MS position.
            """ 
            
            self.shadow  = self.generate_correlated_shadow_vector(MS_pos,type_approx)
            
            return self.shadow
      def get_shadowing_db1(self,MS_pos,type_approx):
     
            d= np.sqrt((self.pos[0] - MS_pos[0] )**2+(self.pos[1] - MS_pos[1])**2)
            self.pos = [MS_pos[0],MS_pos[1],MS_pos[2]]
            a = np.exp(-d/self.shadow_corr_distance )
            b = self.sigma_shadow*np.sqrt(1-a**2)
            sample = np.random.normal(0,1)
            shadow = sample + self.shadow_pre * a
            self.shadow_pre = shadow
            if (b != 0):
                shadow = shadow * b
            return shadow
     
      def get_loss_los (self,distance,h_MS):   
            """ The default method the get the path loss in the LOS condition of the scenario. By default 0. 
            
            @type distance: float.
            @param distance: The distance between the BS and MS positions.
            @type h_MS: float.
            @param h_MS: The MS antenna height.
            @return: 0
            """  
            return self.get_loss(distance,h_MS)
        
      def get_loss_nlos (self,distance,h_MS):
            """ The default method the get the path loss in the Non LOS condition of the scenario. By default 0. 
            
            @type distance: float.
            @param distance: The distance between the BS and MS positions.
            @type h_MS: float.
            @param h_MS: The MS antenna height.
            @return: 0
            """ 
            return self.get_loss(distance,h_MS)
        
      def is_los_cond(self, MS_pos): 
            """The default method to calculate if the scenario is in LOS condition or not. Default True. 
            
            @type MS_pos: 3D array or list.
            @param MS_pos: the position of the movil device in the scenario.
            @return: True
            """   
            return True
                
      def _prob_los(self, distance,h_MS):
            """The default method to calculate the probability function that defines if the scenario is in LOS condition or not. Default 1. 
            
            @type distance: float.
            @param distance: The distance between the BS and MS positions.
            @type h_MS: float.
            @param h_MS: The MS antenna height.
            @return: 1
            """   
            return 1
        
    



