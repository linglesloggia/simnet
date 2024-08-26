import numpy as np 
from scipy.linalg import cholesky
import scipy.spatial
import scipy.stats
import scipy.signal as signal
from libsimnet.usernode import  ChannelEnvironment
import loss_shadowing_model as loss_model
import fading_models as fad
import snr as sr

class ChannelEnvironment(ChannelEnvironment):
  """ This class is the Environment of the statistical fading models.   \
  
  """
  def __init__(self,fcGHz=30,number_prbs=100,bw_prb=180000,posx_min=-300,posx_max=300,posy_min=-300,posy_max=300,grid_number=30,bspos=[0,0,20],Ptx_db=30,sigma_shadow=5,shadow_corr_distance=5,order =3.5,ray_ric = "Rayleigh",number_sin=10,K_LOS=10):  
    """\
    The constructor of the Channel environment.\
    \
    @type fcGHz: float .\
    @param fcGHz: Frequency in GHz of the carrier frequency of the scenario.\
    @type posx_min: float .\
    @param posx_min: The minimum limit of the x coordinate in the scenario. \
    @type posx_max: float .\
    @param posx_max: The maximum limit of the x coordinate in the scenario. \
    @type posy_min: float .\
    @param posy_min: The minimum limit of the y coordinate in the scenario. \
    @type posy_max: float .\
    @param posy_max: The maximum limit of the y coordinate in the scenario. \
    @type grid_number: int .\
    @param grid_number: For calculating the spacial distribution of the parameters of the scenario, \
    the scenario is divided by a grid in x and y cordinates. This value is the number of divisions in each coordinate. \
    @type bspos: 3d array or list .\
    @param bspos: The position of the Base Satation in the scenario in the coordinates system [x,y,z].\
    @type Ptx_db: float.\
    @param Ptx_db: The power transmited by the base station in dbm. \
    @type sigma_shadow: float.\
    @param sigma_shadow: The variance of the shadow gaussian model.\
    @type shadow_corr_distance: float.\
    @param shadow_corr_distance: The shadow correlation distance.\
    
    """ 
    self.fcGHz = fcGHz
    """ Frequency in GHz of the carrier frequency of the scenario. """
    self._LOS = True
    """ Line of sight in the scenario. Boolean. """    
    self.posx_max = posx_max
    """ The maximum limit of the x coordinate in the scenario. """ 
    self.posx_min = posx_min
    """ The minimum limit of the x coordinate in the scenario. """ 
    self.posy_min = posy_min
    """ The minimum limit of the y coordinate in the scenario. """ 
    self.posy_max = posy_max
    """ The maximum limit of the y coordinate in the scenario. """ 
    self.grid_number = int(grid_number)
    """ For calculating the spacial distribution of the parameters of the scenario, 
    the scenario is divided by a grid in x and y cordinates. This value is the number of divisions in each coordinate.""" 
    self.BS_pos =bspos
    """ The position of the Base Satation in the scenario in the coordinates system [x,y,z]. """ 
    self.Ptx_db = Ptx_db
    """ The power transmited by the base station in dbm. """ 
    self.sigma_shadow = sigma_shadow
    """ The variance of the shadow gaussian model """ 
    self.shadow_pre = 0
    """ The previous value of the shadow. Used to filter and impose correlation """ 
    self.shadow_corr_distance = shadow_corr_distance
    """ The shadow correlation distance """ 
    self.pos = [100,100,0]
    """ The previous position of the mobile""" 
    self.shadow_enabled = True
    """ If shadow is enabled or not """
    self.X = np.array([])
    """ The x grid "of the scenario """ 
    self.Y = np.array([])
    """ The y grid "of the scenario """ 
    self.XY = np.array([])
    """ np.array([self.X, self.Y]))  """ 
    stepx = (self.posx_max-self.posx_min)/self.grid_number
    stepy = (self.posy_max-self.posy_min)/self.grid_number

    x = np.linspace(self.posx_min,self.posx_max+stepx,self.grid_number+1) # 2*self.grid_number)*(self.posx_max-self.posx_min)/(2*self.grid_number-1)+self.posx_min\
    y = np.linspace(self.posy_min,self.posy_max+stepy,self.grid_number+1) #np.arange(0, 2*self.grid_number)*(self.posy_max-self.posy_min)/(2*self.grid_number-1)+self.posy_min\
    self.X, self.Y = np.meshgrid(x, y) 
    self._paranum = 1
    """ Number of LSP parameters """ 
    self.grid_shadow,self.XY = self.__generate_correlated_shadowing()
    self.loss = loss_model.SimpleLossModel(self, order)

    self.freq_band = sr.FrequencyBand(fcGHz,number_prbs,bw_prb,noise_figure_db=5.0,thermal_noise_dbm_Hz=-174.0) 
    if ray_ric == "Rayleigh":
        self.fading = fad.FadingSiSoRayleigh(self,self.loss,number_sin)
    else:
        self.fading = fad.FadingSiSoRician(self,self.loss,number_sin,K_LOS)
   

  def __generate_correlated_shadowing(self):
    """This method first generates for each LSP parameter an independent gaussian N(0,1) random variable for each point in the scenario grid. Later, 
    using cholesky method and the correlation matrix between the LSP parameters, generates a set of correlated LSP params.
    At last, the method applies to each parameter its expected value and its variance.

    """ 
    grid_shadow = np.zeros((1,self.grid_number+1,self.grid_number+1))   
    grid_shadow,XY = self.generate_shadow_grid(self.shadow_corr_distance)    
    return grid_shadow, XY


 
    
  def generate_shadow_grid(self,corr_distance):
    """This method generates a spatial correlated gaussian random variables using the correlation distance.\

    The covariance matrix is defined by cov = exp(-distance/correlation_distance)\
    @type corr_distance: float.\
    @param corr_distance: The correlation distance for the spatial correlated gaussian random variable.\
    @return: A 2D matrix where the values of the matrix are spatially correlated gaussian random variables.\
    """ 
    # Create a vector of cells\
    XY = np.column_stack((np.ndarray.flatten(self.X),np.ndarray.flatten(self.Y)))
    # Calculate a matrix of distances between the cells\
    dist = scipy.spatial.distance.pdist(XY)
    dist = scipy.spatial.distance.squareform(dist)    
    # Convert the distance matrix into a covariance matrix\
    cov = np.exp(-dist/(corr_distance)) 
    noise = scipy.stats.multivariate_normal.rvs(mean = np.zeros((self.grid_number+1)**2),cov = cov)
    return(noise.reshape((self.grid_number+1,self.grid_number+1)),np.array([self.X, self.Y]))
 

