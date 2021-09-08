import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.wcs import WCS
from astropy.io import fits

model = 'largernH2_4d_2comp'
output_map = '1dmax'
sou_model = 'radex_model/'
source = 'NGC3351'
mom0 = np.load('data_image/NGC3351_CO21_mom0.npy')
mask = np.load('mask_whole_recovered.npy') #* np.load('mask_13co21_2sig.npy')

fits_map = fits.open('data_image/NGC3351_CO21_mom0_broad_nyq.fits')
line = 'CO21'
wcs = WCS(fits_map[0].header)

N_co = np.load(sou_model+'Nco_'+model+'_'+output_map+'.npy')
T_k = np.load(sou_model+'Tk_'+model+'_'+output_map+'.npy')
n_h2 = np.load(sou_model+'nH2_'+model+'_'+output_map+'.npy')
phi = np.load(sou_model+'phi_'+model+'_'+output_map+'.npy')

par = np.array((N_co, T_k, n_h2, phi))  
title = np.array((r'$\log \left( N_{CO}\cdot\frac{15\ km\ s^{-1}}{\Delta v}\right)$  $(cm^{-2})$',
                  r'$\log\ T_k$  (K)',r'$\log\ n_{H_2}$  $(cm^{-3})$',r'$\log\ \Phi_{bf}$'))
fig = plt.figure(figsize=(20,9))
#cb_range = np.array(([16.,1.,2.,-1.3],[19.,2.,5.,-0.1]))

for i in range(8):
    ax = fig.add_subplot(2,4,i+1, projection=wcs)  #
    ra = ax.coords[0]
    ra.set_major_formatter('hh:mm:ss.s')
    map = par[i%4,i//4] * mask
    map[map == 0] = np.nan
    plt.imshow(map, origin='lower', cmap='inferno')  #, vmin=cb_range[0,i], vmax=cb_range[1,i]
    plt.colorbar()
    plt.contour(mom0, origin='lower', levels=(20,50,100,150,200,250,300), colors='dimgray', linewidths=1)
    if i//4 == 0:
        plt.title(title[i%4], fontsize=14)
        plt.xlabel(' ')
        plt.tick_params(axis="x", labelbottom=False)
    else:
        plt.xlabel('R.A. (J2000)')
    plt.xlim(15,60)
    plt.ylim(15,60)
    
    if i%4 == 0:
        plt.ylabel('Decl. (J2000)')
    else:
        plt.tick_params(axis="y", labelleft=False)
        plt.ylabel(' ')

plt.subplots_adjust(hspace=0.1)
plt.subplots_adjust(wspace=0.1)
plt.savefig('formal_plots/'+model+'_'+output_map+'_subplots.pdf', bbox_inches='tight', pad_inches=0)
plt.show()