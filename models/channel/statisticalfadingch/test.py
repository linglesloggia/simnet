# -*- coding: utf-8 -*-

import statistical_fading_chenv as chenv
import loss_shadowing_model as loss_model
import numpy as np
import matplotlib.pyplot as plt
import fading_models as fad
import snr as sr

def test_SISORayleighFading(ray_ric):
    fcGHz = 30
    posx_min = -300
    posx_max = 300
    posy_min = -300
    posy_max = 300
    grid_number = 30
    BS_pos = [0,0,20]
    Ptx_db = 30
    sigma_shadow=5
    shadow_corr_distance=5
    pos=[0,0,0]
    order= 3.5
    number_sin = 10
    K_LOS =10
    prbs = 100
    bw_prb =180000
    scf=  chenv.ChannelEnvironment(fcGHz=fcGHz,number_prbs=prbs,bw_prb=bw_prb,posx_min=posx_min,posx_max=posx_max,posy_min=posy_min,posy_max=posy_max,grid_number=grid_number,bspos=BS_pos,Ptx_db=Ptx_db,sigma_shadow=sigma_shadow,shadow_corr_distance=shadow_corr_distance,order =order,ray_ric =ray_ric,number_sin=number_sin,K_LOS=K_LOS) 

    MS_pos = [300,300,2]    
    MS_vel = [30,30,0]
    # #loss = loss_model.SimpleLossModel(scf, order=4)
    # t=0
    iterations = 100
    # ls =  np.zeros(iterations)
    # sh =  np.zeros(iterations)
    dist =  np.zeros(iterations)
    # freq_band = sr.FrequencyBand(fcGHz=fcGHz,number_prbs=81,bw_prb=10000000,noise_figure_db=5.0,thermal_noise_dbm_Hz=-174.0) 
    # for i in range(iterations):
    #     t = t+0.001
    #     pos[0] = MS_pos[0]+t*MS_vel[0]
    #     pos[1] = MS_pos[1]+t*MS_vel[1]
    #     pos[2] = MS_pos[2]+t*MS_vel[2]

    #     d3D= np.sqrt((pos[0] - BS_pos[0] )**2+(pos[1] - BS_pos[1] )**2+(pos[2] - BS_pos[2] )**2)
    #     dist[i] = d3D
    #     ls[i] = loss.get_loss(d3D,pos[2])
    #     sh[i] = loss.get_shadowing_db(pos,1)
    # if ray_ric == "Rayleigh":
    #     fading = fad.FadingSiSoRayleigh(scf,loss,number_sin)
    # else:
    #     fading = fad.FadingSiSoRician(scf,loss,number_sin,K_LOS)

    z = np.zeros(iterations,dtype=complex)
    snr = np.zeros(iterations)
    snr_pl = np.zeros(iterations)
    snr_pl_shadow = np.zeros(iterations)
    t= 0
    for i in range(iterations):
         t = t+0.001
         pos[0] = MS_pos[0]+t*MS_vel[0]
         pos[1] = MS_pos[1]+t*MS_vel[1]
         pos[2] = MS_pos[2]+t*MS_vel[2]
         d3D= np.sqrt((pos[0] - BS_pos[0] )**2+(pos[1] - BS_pos[1] )**2+(pos[2] - BS_pos[2] )**2)
         dist[i] = d3D

         z[i] = scf.fading.compute_ch_matrix(pos,MS_vel,t=t,mode=2)
         snr[i],rxPsd,H,G,ploss_linear,snr_pl[i],spectral_eff,snr_pl_shadow[i] = sr.compute_snr(scf.fading,scf.freq_band)

    # z_mag = np.abs(z) # take magnitude for the sake of plotting
    # z_mag_dB = 10*np.log10(z_mag) # convert to dB
    # print(len(ls),len(sh),z_mag_dB.size,len(dist))
    # # Plot fading over time
    # aux = ls+sh+z_mag_dB
    # plt.figure()
    # plt.plot(dist,aux)
    # plt.show()
    
    plt.figure()
    plt.plot(dist,snr)
    plt.plot(dist,snr_pl)
    plt.plot(dist,snr_pl_shadow)
    plt.show()
  
    #print("results--------",snr,rxPsd,H,G,ploss_linear,snr_pl,spectral_eff,snr_pl_shadow)
    # count, bins_count = np.histogram(z_mag, bins=10)
  
    # # finding the PDF of the histogram using count values
    # pdf = count / sum(count)
  
    # # using numpy np.cumsum to calculate the CDF
    # # We can also find using the PDF values by looping and adding
    # cdf = np.cumsum(pdf)
  
    # # plotting PDF and CDF
    # #plt.plot(bins_count[1:], pdf, color="red", label="PDF")
    # plt.plot(bins_count[1:], cdf, label="CDF")
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    
   test_SISORayleighFading("Rayleigh")# "Riccian"
    