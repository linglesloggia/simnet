# -*- coding: utf-8 -*-
import numpy as np
import copy
def compute_snr(fading,freq_band,wMS=[1],wBS=[1],MSAntenna=None,BSAntenna=None):
    """ This method computes the snr and other intermidate performance metrics for one point in the mobile path.
    
    The method computes the pathloss, the shadowing, the Fourier transform of the channel matrix,
    the beamforming gain and other variables. Using this information computes the average snr and the
    received power spectral density.

    @type fading: Fading Class.
    @param fading: The fading channel model object in one point of the path.
    @type freq_band: FrequencyBnad Class.
    @param freq_band: The frequency band containing the transmited power spectral density and the noise power spctral density for each prb. 
    @type wMS: numpy array.
    @param wMS: the beamforming vector at the MS.
    @type wBS: numpy array.
    @param wBS: the beamforming vector at the BS. 
    @return: snr (10 * np.log10(np.sum(rxPsd.psd) /np.sum(noisePsd.psd))), rxPsd (txPsd* 10**(-pathloss-shadowing) * (np.abs(beamforming gain)**2)), the channel matrix Fourier Transform, the beamforming gain, the linear path loss, the snr taking into account only the path loss,
    the spectral efficiency of the channel, the snr taking into account pathloss and shadow.
    """
    rxPsd = copy.deepcopy(freq_band.txpsd)  
    d3D= np.sqrt((fading.MS_pos[0] - fading.scenario.BS_pos[0] )**2+(fading.MS_pos[1] - fading.scenario.BS_pos[1] )**2+(fading.MS_pos[2] - fading.scenario.BS_pos[2] )**2)
    if fading.loss_model.is_los_cond(fading.MS_pos):
      ploss_db = fading.loss_model.get_loss_los(d3D,fading.MS_pos[2])
    else:
      ploss_db = fading.loss_model.get_loss_nlos(d3D,fading.MS_pos[2])
    ploss_linear = pow(10.0, -ploss_db / 10.0)
    ploss_linear_shadow = ploss_linear
    if (fading.scenario.shadow_enabled):
      shadow = fading.loss_model.get_shadowing_db(fading.MS_pos,1)  
      ploss_db_shadow = ploss_db+ shadow
      ploss_linear_shadow = pow(10.0, -ploss_db_shadow / 10.0)
    ploss_linear = pow(10.0, -ploss_db / 10.0)
    rxPsd *= ploss_linear
    snr_pl = 10 * np.log10(np.sum(rxPsd) /np.sum(freq_band.noisepsd))
    rxPsd = rxPsd/ploss_linear* ploss_linear_shadow
    snr_pl_shadow = 10 * np.log10(np.sum(rxPsd) /np.sum(freq_band.noisepsd))

    H = compute_Tfourier_H(fading,freq_band,MSAntenna,BSAntenna)
    G = compute_beamforming_gain(fading,freq_band.n_prbs,H,wMS,wBS,MSAntenna,BSAntenna)
    rxPsd = rxPsd*(np.abs(G)**2)

    snr = 10 * np.log10(np.sum(rxPsd) /np.sum(freq_band.noisepsd))
    spectral_eff=  np.log(1+np.sum(rxPsd) /np.sum(freq_band.noisepsd))
    return snr,rxPsd,H,G,ploss_linear,snr_pl,spectral_eff,snr_pl_shadow


def compute_Tfourier_H(fading,freq_band,MSAntenna=None,BSAntenna=None):
    """ This method computes the fourier transform of the channel matrix for each prb,
    each MS antenna element and each BS antenna element.

    @type fading: Fading3gpp Class.
    @param fading: The fast fading 3gpp channel model object in one point of the path.
    @type freq_band: FrequencyBnad Class.
    @param freq_band: The frequency band containing the transmited power spectral density and the noise power spctral density for each prb. 
    @return: The fourier transform of the channel matrix for each prb,
    each MS antenna element and each BS antenna element.
    """
    H_usn = fading.H_usn
    n_clusters = H_usn.shape[2]
    if BSAntenna is not None:
        n_BS = BSAntenna.get_number_of_elements()
    else:
        n_BS = 1
    if MSAntenna is not None:
        n_MS = MSAntenna.get_number_of_elements()
    else:
        n_MS = 1
    H_us_f = np.zeros((freq_band.n_prbs,n_MS,n_BS),dtype=complex)
    for prb in range(freq_band.n_prbs):
      f =freq_band.fc_prbs[prb]
      for i in range(n_MS):
        for j in range(n_BS):
          for k  in range(n_clusters):
            tau = -2 * np.pi * f * fading.tau[k]
            H_us_f[prb][i][j]= H_us_f[prb][i][j] + H_usn[i][j][k]* np.exp(complex(0, tau))
    return H_us_f

def compute_beamforming_gain(fading,n_prbs,H_f,wMS=[1],wBS=[1],MSAntenna=None,BSAntenna=None):
    """ This method given the Tx power spectral density, the Fourier Tranform of the 
    impulse response of the channel and the MS and BS beamforming vectors,
    computes the beamforming gain for each prb of the OFDM specrum.
 
    @type fading: Fading Class.
    @param fading: The fading channel model object in one point of the path.
    @type n_prbs: int.
    @param n_prbs: number of  prbs in the frequency band.
    @type H_f: 3d numpy array.
    @param H_f: the channel matrix fourier transform for each prb, each MS antenna element,
    and each BS antenna element.
    @type wMS: numpy array.
    @param wMS: the beamforming vector at the MS.
    @type wBS: numpy array.
    @param wBS: the beamforming vector at the BS. 
    @return: bemaforming Gain for each prb (wMS . H_f[prb].wBS)
    """

    if BSAntenna is not None:
        n_BS = BSAntenna.get_number_of_elements()
    else:
        n_BS = 1
    if MSAntenna is not None:
        n_MS = MSAntenna.get_number_of_elements()
    else:
        n_MS = 1
    G_f = np.zeros((n_prbs),dtype=complex)
    for prb in range(n_prbs):
      for i in range(n_MS):
        for j in range(n_BS):
         G_f[prb]= G_f[prb] + wMS[i]*H_f[prb][i][j]*wBS[j]
    return G_f

def compute_Tfourier_H(fading,freq_band,MSAntenna=None,BSAntenna=None):
    """ This method computes the fourier transform of the channel matrix for each prb,
    each MS antenna element and each BS antenna element.

    @type fading: Fading3gpp Class.
    @param fading: The fast fading 3gpp channel model object in one point of the path.
    @type freq_band: FrequencyBnad Class.
    @param freq_band: The frequency band containing the transmited power spectral density and the noise power spctral density for each prb. 
    @return: The fourier transform of the channel matrix for each prb,
    each MS antenna element and each BS antenna element.
    """
    H_usn = fading.H_usn
    n_clusters = H_usn.shape[2]
    if BSAntenna is not None:
        n_BS = BSAntenna.get_number_of_elements()
    else:
        n_BS = 1
    if MSAntenna is not None:
        n_MS = MSAntenna.get_number_of_elements()
    else:
        n_MS = 1
    H_us_f = np.zeros((freq_band.n_prbs,n_MS,n_BS),dtype=complex)
    for prb in range(freq_band.n_prbs):
      f =freq_band.fc_prbs[prb]
      for i in range(n_MS):
        for j in range(n_BS):
          for k  in range(n_clusters):
            tau = -2 * np.pi * f * fading.tau[k]
            H_us_f[prb][i][j]= H_us_f[prb][i][j] + H_usn[i][j][k]* np.exp(complex(0, tau))
    return H_us_f
    


class FrequencyBand:
  """This class implements the OFDM frequency spectrum. 
  
  """  
  def __init__(self,fcGHz,number_prbs=100,bw_prb=180000,noise_figure_db=5.0,thermal_noise_dbm_Hz=-174.0): 
    """ FrequencyBand constructor
    
    Note: This version assumes that all prbs transport data.
    @type fcGHz: float.
    @param fcGHz: The carrier frequency of OFDM spectrum in GHz.
    @type number_prbs: int.
    @param number_prbs: The number of physical reseource blocks (PRB) in OFDM. Default 100.
    @type bw_prb: float.
    @param bw_prb: The bandwidth of each physical reseource blocks in OFDM. In Hertz.
    Default 180000.
    @type noise_figure_db: float.
    @param noise_figure_db :The noise figure in db.
    @type thermal_noise_dbm_Hz: float.
    @param thermal_noise_dbm_Hz :The thermal noise in dbm per Hertz.

   """
    self.n_prbs = number_prbs
    """ The number of reseource blocks in OFDM.  """ 
    self.bw_prb = bw_prb
    """ The bandwidth of each reseource blocks in OFDM. In Hertz.""" 
    self.fcGHz = fcGHz
    """ carrier frequency of OFDM spectrum in GHz.""" 
    self.noise_figure_db = noise_figure_db
    """ The noise figure in db.""" 
    self.thermal_noise_dbm_Hz = thermal_noise_dbm_Hz
    """The thermal noise in dbm per Hertz.""" 
    self.fc_prbs = np.zeros(self.n_prbs)
    """The frequency of each prb """
    f = fcGHz*1e9 - (self.n_prbs * self.bw_prb / 2.0)
    for nrb in range(self.n_prbs):
      f = f + self.bw_prb/2
      self.fc_prbs[nrb] = f
      f = f + self.bw_prb/2
    self.txpsd = np.ones(self.n_prbs)
    """ The transmited power spectral density of each prb """ 
    self.noisepsd = np.ones(self.n_prbs)
    """ The noise  power spectral density of each prb """     
    self.compute_noise_psd()
    
  def compute_tx_psd (self,tx_power_dbm):
    """ This method given the Tx power computes the Tx power spectral density.

    TODO: In this version the Tx power is divided equally between prbs.
    @type tx_power_dbm: float.
    @param tx_power_dbm :The Tx power of the BS in dbm.
    """
    tx_power_W = pow(10,(tx_power_dbm - 30) / 10) # 30 is to convert from mW to W
    tx_power_density = tx_power_W / (self.n_prbs * self.bw_prb)
    for prb in range(self.n_prbs): 
      self.txpsd[prb] = tx_power_density
    
  def compute_noise_psd(self):
    """ This method given the noise figure and the thermal noise, computes the \
    noise power spectral density.

    """
    th_noise_W = pow(10, (self.thermal_noise_dbm_Hz - 30) / 10)
    noise_figure = pow (10, self.noise_figure_db / 10)
    noise_psd_value =  th_noise_W * noise_figure
    for prb in range(self.n_prbs): 
      self.noisepsd[prb] = noise_psd_value

################# Default Values ##############################\

# Stefania Sesia, Issam Toufik, Matthew Baker - LTE - The UMTS Long Term Evolution_ From Theory to Practice-Wiley (2011)\
# pag 478 : In the LTE specifications the thermal noise density, kT , is defined to be \uc0\u8722 174 dBm/Hz where k is Boltzmann\'92s \
#constant (1.380662 \'d7 10\uc0\u8722 23) and T is the temperature of the receiver (assumed to be 15\u9702 C)\
# pag 479:\
#LTE defines an NF requirement of 9 dB for the UE, the same as UMTS. This is somewhat higher than the NF of a state-of-the-art \
#receiver, which would be in the region of 5\'966 dB, with typically about 2.5 dB antenna filter insertion loss and an NF for the \
#receiver integrated circuit of 3 dB or less. Thus, a practical 3\'964 dB margin is allowed. The eNodeB requirement is for an NF of 5 dB.\


# 3GPP TR 36.942 version 13.0.0 Release 13\
#Table 4.6: E-UTRA FDD and E-UTRA TDD reference base station parameters\
#Maximum BS power: 43dBm for 1.25, 2.5 and 5MHz carrier, 46dBm for 10, 15 and 20MHz carrier\
#Maximum power per DL traffic channel : 32dBm\
#Noise Figure 5 db\

# In 5G NR see: https://www.etsi.org/deliver/etsi_ts/138100_138199/138104/15.04.00_60/ts_138104v150400p.pdf\
# In 5G NR there different base stations architectures (1-c,1-h 1-o,2-o ) and different bands fr1 and fr2.\
# Each case has differentt max output power all in the range 30-50 dbm   \
     
# kT_dBm_Hz  = -174.0 # dBm/Hz\
# noiseFigure = 5.0; # noise figure in dB\


# fb = FrequencyBand(1)\
# fb.compute_tx_psd(46.5)\
# print("tx psd",10*np.log10(fb.psd))\
# print(fb.psd*180000)\
# fb = FrequencyBand(1)\
# fb.compute_noise_psd(noiseFigure,kT_dBm_Hz)\
# print("noise psd",10*np.log10(fb.psd))\




