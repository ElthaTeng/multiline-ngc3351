import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

par = 'mom0'  # ratio, mom0, 1comp, 2comp, alpha
region = 'whole'  # center, ring, arms, whole
mask_whole = np.load('mask_whole_recovered.npy') #* np.load('mask_13co21_3sig.npy')

if par == 'ratio':
    maps = np.full((7,75,75), np.nan)
    line_up = np.array(('CO21', '13CO32', 'C18O32', 'CO21', 'CO21', '13CO21', '13CO32'))
    line_low = np.array(('CO10', '13CO21', 'C18O21', '13CO21', 'C18O21', 'C18O21', 'C18O32'))
    for i in range(7):    
        map_up = np.load('data_image/NGC3351_'+line_up[i]+'_mom0.npy')
        map_low = np.load('data_image/NGC3351_'+line_low[i]+'_mom0.npy')

        map = map_up/map_low
        map[map<=0] = np.nan
        maps[i] = map
        
elif par == 'mom0':
    maps = np.full((6,75,75), np.nan)
    maps[0] = np.load('data_image/NGC3351_CO10_mom0.npy')
    maps[1] = np.load('data_image/NGC3351_CO21_mom0.npy')
    maps[2] = np.load('data_image/NGC3351_13CO21_mom0.npy')
    maps[3] = np.load('data_image/NGC3351_13CO32_mom0.npy')
    maps[4] = np.load('data_image/NGC3351_C18O21_mom0.npy')
    maps[5] = np.load('data_image/NGC3351_C18O32_mom0.npy')

        
elif par == '1comp':
    maps = np.full((6,75,75), np.nan)
    maps[0] = np.load('radex_model/Nco_6d_coarse_rmcor_whole_los100_1dmax.npy')
    maps[1] = np.load('radex_model/Tk_6d_coarse_rmcor_whole_los100_1dmax.npy')
    maps[2] = np.load('radex_model/nH2_6d_coarse_rmcor_whole_los100_1dmax.npy')
    maps[3] = np.load('radex_model/X12to13_6d_coarse_rmcor_whole_los100_1dmax.npy')
    maps[4] = np.load('radex_model/X13to18_6d_coarse_rmcor_whole_los100_1dmax.npy')
    maps[5] = np.load('radex_model/phi_6d_coarse_rmcor_whole_los100_1dmax.npy')
    
elif par == '2comp':
    maps = np.full((8,75,75), np.nan)
    maps[0] = np.load('radex_model/Nco_largernH2_4d_2comp_1dmax.npy')[0]
    maps[1] = np.load('radex_model/Tk_largernH2_4d_2comp_1dmax.npy')[0]
    maps[2] = np.load('radex_model/nH2_largernH2_4d_2comp_1dmax.npy')[0]
    maps[3] = np.load('radex_model/phi_largernH2_4d_2comp_1dmax.npy')[0]
    maps[4] = np.load('radex_model/Nco_largernH2_4d_2comp_1dmax.npy')[1]
    maps[5] = np.load('radex_model/Tk_largernH2_4d_2comp_1dmax.npy')[1]
    maps[6] = np.load('radex_model/nH2_largernH2_4d_2comp_1dmax.npy')[1]
    maps[7] = np.load('radex_model/phi_largernH2_4d_2comp_1dmax.npy')[1]
    
else:
    maps = np.full((1,75,75), np.nan)
    maps[0] = np.load('radex_model/Xco_6d_coarse_alpha_median_los100.npy')
    
if region == 'center':
    mask = np.load('mask_cent3sig.npy')
elif region == 'ring':
    radius = np.load('ngc3351_radius_arcsec.npy') > 4.5
    mask = np.load('mask_cent3sig.npy') * radius
elif region == 'arms':
    mask = np.load('mask_arms.npy') * mask_whole #* np.load('mask_c18o21_1sig.npy')  #np.load('mask_rmcor_comb_lowchi2.npy')  #
else:
    mask = mask_whole #* np.load('mask_rmcor_comb_lowchi2.npy') #(np.load('radex_model/chi2_4d_2comp.npy') < 10)  #
  
maps = maps * mask
maps[:,mask==0] = np.nan 
maps[np.isinf(maps)] = np.nan 

if par == 'mom0':
    mom0 = np.nansum(maps, axis=(1,2))
    print(mom0)
    ratios = np.array((mom0[1]/mom0[0], mom0[3]/mom0[2], mom0[5]/mom0[4], mom0[1]/mom0[2], mom0[1]/mom0[4], mom0[2]/mom0[4], mom0[3]/mom0[5]))
    print(ratios)
else:
    print(np.nanmedian(maps, axis=(1,2)), '+/-', np.nanstd(maps, axis=(1,2)))
    print(np.nanmean(maps, axis=(1,2)))

'''
for i in range(7):
    plt.subplot(2,4,i+1)
    plt.imshow(maps[i], origin='lower', cmap='jet')
    plt.colorbar()
plt.show()
'''