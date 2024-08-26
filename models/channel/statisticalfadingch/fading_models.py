#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module computes the fading model of differents scenarios.

@author: pablo belzarena
"""

import numpy as np
from time import perf_counter


class Fading:
  """This class implments the base fading model.
  
  This class defins differents methods common to all fading models.
  """  
  def __init__(self,scenario,loss_model):
    """
    The constructor method of the Fading Class.
    This method sets the scenario of the fading model and generates an empty short scale parameters grid.
    
    @type scenario: Class Scenario.
    @param scenario: The scenario for the fast fading model.
    """
    self.scenario = scenario
    self.loss_model = loss_model
    """ The scenario for the fading model.""" 
    self.__ssp_grid = np.empty(shape=(self.scenario.X[0].size,self.scenario.Y[:,0].size), dtype=object)
    """ An array of objects with one element for each point of the grid. Each element is an SSPs object. """

    
  def compute_ssps(self,pos):
        """
        This method computes the ssps for this position. It is an abstract method. It must be implemented
        in the child classes.
        
        @type pos: array.
        @param pos: The position of the MS.
        """ 
   
  def set_correlated_ssps(self,pos,ret_ssp):
    """ This method using the ssps grid, computes the ssps for this position in the scenario. It
    also sets the ssps for this fading object.

    TODO: uncouple LSP and ssp grids.
    @type pos: Array.
    @param pos: the psoition in the scenario to interpolate.
    @type ret_ssp: SSP object.
    @param ret_ssp: the SSP object to return the SSP parameters of the position pos.
    """
    i_x1,i_x2,i_y1,i_y2 = self.find_point_meshgrid(pos,self.scenario.X,self.scenario.Y)
    ########
    los = self.loss_model.is_los_cond(pos)
    ######
    p00 = [self.scenario.X[0][i_x1],self.scenario.Y[i_y1][0],pos[2]]
    p01 = [self.scenario.X[0][i_x1],self.scenario.Y[i_y2][0],pos[2]]
    p10 = [self.scenario.X[0][i_x2],self.scenario.Y[i_y1][0],pos[2]]
    p11 = [self.scenario.X[0][i_x2],self.scenario.Y[i_y2][0],pos[2]]
    los_p00 = self.loss_model.is_los_cond(p00)
    los_p01 = self.loss_model.is_los_cond(p01)
    los_p10 = self.loss_model.is_los_cond(p10)
    los_p11 = self.loss_model.is_los_cond(p11)
    points = []
    ssps_grid = []

    if los_p00 == los: 
        if self.__ssp_grid[i_x1][i_y1]==None:
            self.__ssp_grid[i_x1][i_y1] = self.compute_ssps(p00)
        points.append(p00)
        ssps_grid.append((self.__ssp_grid[i_x1][i_y1]))
        
    if los_p01 == los:
        if self.__ssp_grid[i_x1][i_y2]==None:
            self.__ssp_grid[i_x1][i_y2] = self.compute_ssps(p01)
        points.append(p01)
        ssps_grid.append((self.__ssp_grid[i_x1][i_y2]))
    if los_p10 == los:
        if self.__ssp_grid[i_x2][i_y1]==None:
            self.__ssp_grid[i_x2][i_y1] = self.compute_ssps(p10)            
        points.append(p10)
        ssps_grid.append((self.__ssp_grid[i_x2][i_y1]))
    if los_p11 == los:
        if self.__ssp_grid[i_x2][i_y2]==None:
            self.__ssp_grid[i_x2][i_y2] = self.compute_ssps(p11)
        points.append(p11)
        ssps_grid.append((self.__ssp_grid[i_x2][i_y2]))
    if len(ssps_grid) > 0:
        ####################################################
        #ret_ssp = SSPs3GPP(15,ssps_grid[0].n_scatters)
        ####################################################
        for i in range(ssps_grid[0].number_sps):
            values = []
            for ssp in ssps_grid:
                values.append(ssp.ssp_array[i])    
            ret_ssp.ssp_array[i] = self.inverse_distance_interpol(pos, np.array(points),np.array(values))
    else:
            ret_ssp = self.compute_ssps(pos)
    return ret_ssp  

  def inverse_distance_interpol(self,point, XY,values, p = 2):
    """
    This method interpolates one point with the values in the XY points using inverse distance interpolation with module p.
    
    @type point: array.
    @param point: The point for the interpolation.
    @type XY: array
    @param XY: a matrix where each row is a point of the grid.
    @type values: array
    @param values: an array with the values in each XY point.
    @return: the value interpolated.
    """
    d = np.sqrt ((point[0] - XY[:,0]) ** 2 +(point[1] - XY[:,1]) ** 2) ** p
    if d.min () == 0:
      ret = values[np.unravel_index(d.argmin(), d.shape)]
    else:
      w = 1.0 / d
      shape = values.shape
      aux = (values.T * w).T #[:,None]
      aux = aux.reshape(shape)
      sumcols = np.sum (aux,axis=0)
      ret = sumcols / np.sum (w)
    return ret
 
  def find_point_meshgrid(self,pos,X,Y):
    """ This method given a position in the scenario, finds the  square (the four vertices)
    of the grid where the point is inside.

    @type pos: array.
    @param pos: The position of the MS in the scenario.
    @type X: array
    @param X: a matrix of the x coordinates of the grid. results of X,Y = np.meshgrid(x,y)
    @type Y: array
    @param Y: a matrix of the y coordinates of the grid. results of X,Y = np.meshgrid(x,y)
    @return: x1,x2,y1,y2 - the x and y coordinates of the four vertices.
    """       
    index_x1 = np.argmin(np.abs(X[0]-pos[0]))
    if pos[0] < X[0][index_x1]:
        if index_x1 == 0:
            index_x2 = 0
        else:
            index_x2 = index_x1
            index_x1 = index_x1 -1
    elif pos[0] > X[0][index_x1]:
       if index_x1 == X[0].size-1:
           index_x2 = index_x1
       else:
           index_x2 = index_x1+1
    else:
        index_x2 = index_x1
   
    index_y1 = np.argmin(np.abs(Y[:,0]-pos[1]))
    if pos[1] < Y[:,0][index_y1]:
        if index_y1 == 0:
            index_y2 = 0
        else:
            index_y2 = index_y1
            index_y1 = index_y1 -1
    elif pos[1] > Y[:,0][index_y1]:
       if index_y1 == Y[:,0].size-1:
           index_y2 = index_y1
       else:
           index_y2 = index_y1+1
    else:
        index_y2 = index_y1
    return(index_x1,index_x2,index_y1,index_y2)



class FadingSiSoRayleigh(Fading):
      """This class implments a simple SISO Rayleigh fading model using sum of sinusoids.\
      
      """  
      def __init__(self,scenario,loss_model,number_sin):
        """
        The constructor method of the  Fading Rayleigh Class.
        This method sets the scenario and generates the ssp object.
        
        @type scenario: Class Scenario.
        @param scenario: The scenario for the fast fading model.
        """
        super().__init__(scenario,loss_model)
        
        self.number_sin = number_sin # number of sinusoids to sum\
        self.ssp = self.compute_ssps([0,0,0])
        
      def compute_ssps(self,pos):
        """
        This method computes the ssps for this position.\
        
        It calls the methods to compute the Small Scale Parameters. The ssps parameters are three in this model.\
        The number of sinusoids, the alpha and phi angles of each sinusoid. The position is not used in this model.\
        
        @type pos: array.
        @param pos: The position of the MS.
        """ 
        ssp = SSPsRayleigh(3,self.number_sin) 
        return ssp
    
      def compute_ch_matrix(self,MS_pos,MS_vel,aMS=None,aBS=None,t=0,mode=0,random_weigh = 0.0):
        """ 
        This method computes  the channel matrix according to the calculation mode.\
        
        This method sets the MS_pos, the MS_vel, and the simulation time.
        The amplitud of each sinusoid is an independent random variable for each point and each sinusoid. \
    
        @type MS_pos: 3D array or list.
        @param MS_pos: The position of the mobile device in the scenario.
        @type MS_vel: 3D array or list.
        @param MS_vel: the velocity vector of the mobile device in the scenario.
        @type aMS: Class Antenna.
        @param aMS: The MS antenna.
        @type aBS: Class Antenna.
        @param aBS: The BS antenna.  
        @type t: float.
        @param t: The current time of the simulation in seconds.
        @type mode: int.
        @param mode: If mode =0 the ssps are generated for the MS_pos interpolating in the correlated ssp grid. If mode = 1
        uses the same alpha and phi for all points. If mode = 2 the alpha and phi are generated with independet random variables
        for each point.
    
        """ 
        self.MS_pos = MS_pos
        self.MS_vel = MS_vel
        self.MS_t = t
        self.H_usn = np.zeros((1,1,1), dtype=complex)  
        self.tau = np.zeros(1)
        self.tau[0] =  np.sqrt(MS_pos[0]**2+MS_pos[1]**2+MS_pos[2]**2)/3e8
        
        v = np.sqrt(MS_vel[0]**2+MS_vel[1]**2+MS_vel[2]**2)
        fd = v*self.scenario.fcGHz/3e8 # max Doppler shift\
        #t = np.arange(0, 1, 1/Fs) # time vector. (start, stop, step)\
        x = 0
        y = 0 
        if mode == 0:
            ssps = self.compute_ssps([0,0,0])
            self.ssp = self.set_correlated_ssps(MS_pos,ssps)
            alpha = self.ssp.alpha
            phi = self.ssp.phi
        elif mode == 1:    
            alpha = self.ssp.alpha
            phi = self.ssp.phi
        else:
            self.ssp = self.compute_ssps(MS_pos)
            alpha = self.ssp.alpha
            phi = self.ssp.phi
        for i in range(self.number_sin):
            #x = x + np.random.randn() * np.cos(2 * np.pi * fd * t * np.cos(self.alpha) + self.phi)\
            #y = y + np.random.randn() * np.sin(2 * np.pi * fd * t * np.cos(self.alpha) + self.phi)\
            x = x + (random_weigh*(2*np.random.randn()-1)+1-random_weigh)* np.cos(2 * np.pi * fd * t * np.cos(alpha[i])) * np.cos( phi[i])
            y = y + (random_weigh*(2*np.random.randn()-1)+1-random_weigh)* np.sin(2 * np.pi * fd * t * np.cos(alpha[i])) * np.sin(phi[i])
        self.H_usn[0] = (np.sqrt(2/self.number_sin)) * (x + 1j*y)
        return self.H_usn   

class FadingSiSoRician(Fading):
  """This class implments a simple SISO Rician fading model using sum of sinusoids\
  """  
  def __init__(self,scenario,loss_model,number_sin,K_LOS):
    """
    The constructor method of the  Fading Rician Class.
    This method sets the scenario and generates the ssp object.
    
    @type scenario: Class Scenario.
    @param scenario: The scenario for the fast fading model.
    @type K_LOS: Cfloat
    @param K_LOS: The K cosntant of the Rice model.
     
    """
    super().__init__(scenario,loss_model)

    
    self.number_sin = number_sin # number of sinusoids to sum
    self.ssp = self.compute_ssps([0,0,0])
    self.K = K_LOS
    
  def compute_ssps(self,pos):
    """
    This method computes the ssps for this position.
    
    It calls the methods to compute the Small Scale Parameters. The ssps parameters are three in this model.\
    The number of sinusoids, the alpha and phi angles of each sinusoid. The position is not used in this model.\
    
    @type pos: array.
    @param pos: The position of the MS.
    """ 
    ssp = SSPsRayleigh(3,self.number_sin) 
    return ssp


  def compute_ch_matrix(self,MS_pos,MS_vel,aMS=None,aBS=None,t=0,mode=0,phase_LOS=0,phase_ini=0,random_weigh=0.0):
    """ 
    This method computes  the channel matrix according to the calculation mode.
    
    This method sets the MS_pos, the MS_vel, and the simulation time.
    The amplitud of each sinusoid is an independent random variable for each point and each sinusoid. 

    @type MS_pos: 3D array or list.
    @param MS_pos: The position of the mobile device in the scenario.
    @type MS_vel: 3D array or list.
    @param MS_vel: the velocity vector of the mobile device in the scenario.
    @type aMS: Class Antenna.
    @param aMS: The MS antenna.
    @type aBS: Class Antenna.
    @param aBS: The BS antenna.  
    @type t: float.
    @param t: The cuurent time of the simulation in seconds.
    @type mode: int.
    @param mode: If mode =0 the ssps are generated for the MS_pos interpolating in the correlated ssp grid. If mode = 1
    uses the same alpha and phi for all points. If mode = 2 the alpha and phi are generated with independet random variables
    for each point.
    @type phase_LOS: float in [- pi,pi).
    @param phase_LOS: The arrival phase of the LOS ray.
    @type phase_ini:: float in [- pi,pi).
    @param phase_ini: the initial phase of the LOS ray. Tipically is a random variable uniformly distributed over [-pi,pi). 
    """ 

    self.MS_pos = MS_pos
    self.MS_vel = MS_vel
    self.MS_t = t
    self.H_usn = np.zeros((1,1,1), dtype=complex)  
    self.tau = np.zeros(1)
    self.tau[0] =  np.sqrt(MS_pos[0]**2+MS_pos[1]**2+MS_pos[2]**2)/3e8
    v = np.sqrt(MS_vel[0]**2+MS_vel[1]**2+MS_vel[2]**2)
    fd = v*self.scenario.fcGHz/3e8 # max Doppler shift
    x = 0 
    y = 0 
    if mode == 0:
        ssps = self.compute_ssps([0,0,0])
        self.ssp = self.set_correlated_ssps(MS_pos,ssps)
        alpha = self.ssp.alpha
        phi = self.ssp.phi
    elif mode == 1:    
        alpha = self.ssp.alpha
        phi = self.ssp.phi
    else:
        self.ssp = self.compute_ssps(MS_pos)
        alpha = self.ssp.alpha
        phi = self.ssp.phi
    for i in range(self.number_sin):
        #x = x + np.random.randn() * np.cos(2 * np.pi * fd * t * np.cos(self.alpha) + self.phi)\
        #y = y + np.random.randn() * np.sin(2 * np.pi * fd * t * np.cos(self.alpha) + self.phi)\
        x = x + (random_weigh*(2*np.random.randn()-1)+1-random_weigh)* np.cos(2 * np.pi * fd * t * np.cos(alpha[i])) * np.cos( phi[i])
        y = y + (random_weigh*(2*np.random.randn()-1)+1-random_weigh)* np.sin(2 * np.pi * fd * t * np.cos(alpha[i])) * np.sin(phi[i])
    z = (np.sqrt(2/self.number_sin)) * (x + 1j*y) # this is what you would actually use when simulating the channel\
    w = np.exp(1j*(2 * np.pi * fd * t * np.cos(phase_LOS)+phase_ini))
    self.H_usn[0] = z/np.sqrt(self.K+1)+w*np.sqrt(self.K/(self.K+1))
    return self.H_usn  


class SSPs:
  """This class implments the Short Scale Parameterss object.
  
  This class enables the access to all ssps in a single object. The ssps are
  stored in an array, and can be acceded in this way but also they can be access by setter and getter methos 
  like properties.
  """  
  def __init__(self,number_sps):
    """
    The constructor method of the  SSPs Class.
    
    @type n_scatters: int.
    @param n_scatters: The number of the scatters in this channel model.
    """ 
    self.number_sps = number_sps
    """ The number of the short scale parameters. """ 
    self.ssp_array = np.empty(shape=(self.number_sps),dtype=object)
    """ An array of ssps. Each element of the array is an array of each ssp.""" 

class SSPsRayleigh(SSPs):
  """This class implments the Short Scale Parameterss object for 3Rayleigh fading model.
  
  This class enables the access to all ssps in a single object. The ssps are
  stored in an array, and can be acceded in this way but also they can be access by setter and getter methos 
  like properties.
  """  
  def __init__(self,number_sps,number_sin):
    """
    The constructor method of the  SSPs Class.
    
    @type n_scatters: int.
    @param n_scatters: The number of the scatters in this channel model.
    """ 
    super().__init__(number_sps)
    self.ssp_array[0]= number_sin
    self.set_angles()
   
  def set_angles(self):
    alpha = np.zeros(self.ssp_array[0])
    phi = np.zeros(self.ssp_array[0])
    for i in range(self.ssp_array[0]):
        alpha[i] = (np.random.rand() - 0.5) * 2 * np.pi
        phi[i] = (np.random.rand() - 0.5) * 2 * np.pi
    self.ssp_array[1] = alpha
    self.ssp_array[2] = phi
   
  @property  
  def n_sin(self):
    """ Gets the number of simusoids """ 
    return int(self.ssp_array[0])

  @n_sin.setter
  def n_sin(self,value):
    """ Sets the number of simusoids""" 
    self.ssp_array[0] = int(value)
  
  @property  
  def alpha(self):
    """ Gets the alpha angles """ 
    return self.ssp_array[1]

  @alpha.setter
  def alpha(self,value):
    """ Sets the alpha angles""" 
    self.ssp_array[1] = value
    
  @property  
  def phi(self):
    """ Gets the alpha angles """ 
    return self.ssp_array[2]

  @phi.setter
  def phi(self,value):
    """ Sets the alpha angles""" 
    self.ssp_array[2] = value
 
      

