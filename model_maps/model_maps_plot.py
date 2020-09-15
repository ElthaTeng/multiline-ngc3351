import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.wcs import WCS
from astropy.io import fits

model = '6d_coarse'
output_map = '1dmax'
sou_model = 'radex_model/'
source = 'NGC3351'
mom0 = np.load('data_image/NGC3351_CO21_mom0.npy')
mask = np.load('mask_139_pixels.npy')

fits_map = fits.open('data_image/NGC3351_CO21_mom0_broad_nyq.fits')
line = 'CO21'
wcs = WCS(fits_map[0].header)

N_co = np.load(sou_model+'Nco_'+model+'_'+output_map+'.npy')
T_k = np.load(sou_model+'Tk_'+model+'_'+output_map+'.npy')
n_h2 = np.load(sou_model+'nH2_'+model+'_'+output_map+'.npy')
X_co213co = np.load(sou_model+'X12to13_'+model+'_'+output_map+'.npy')
X_13co2c18o = np.load(sou_model+'X13to18_'+model+'_'+output_map+'.npy')
phi = np.load(sou_model+'phi_'+model+'_'+output_map+'.npy')

par = np.array((N_co, T_k, n_h2, X_co213co, X_13co2c18o, phi))  
title = np.array((r'(a) $\log \left( N_{CO}\cdot\frac{15\ km\ s^{-1}}{\Delta v}\right)$  $(cm^{-2})$',r'(b) $\log\ T_k$  (K)',r'(c) $\log\ n_{H_2}$  $(cm^{-3})$',r'(d) $X_{12/13}$',r'(e) $X_{13/18}$',r'(f) $\Phi_{bf}$'))
fig = plt.figure(figsize=(18,10))

for i in range(6):
    ax = fig.add_subplot(2,3,i+1, projection=wcs)  #
    ra = ax.coords[0]
    ra.set_major_formatter('hh:mm:ss.s')
    map = par[i] * mask
    map[map == 0] = np.nan
    #plt.imshow(map, origin='lower', cmap='inferno', norm=LogNorm(vmin=10,vmax=200))
    plt.imshow(map, origin='lower', cmap='inferno')
    plt.colorbar()
    plt.contour(mom0, origin='lower', levels=(20,50,100,150,200,250,300), colors='dimgray', linewidths=1)
    plt.title(title[i])
    plt.xlim(15,60)
    plt.ylim(15,60)
    plt.xlabel('R.A. (J2000)')
    plt.ylabel('Decl. (J2000)')

#plt.subplots_adjust(hspace=0.5)
plt.subplots_adjust(wspace=0.3)
plt.savefig('formal_plots/'+model+'_'+output_map+'_subplots.pdf', bbox_inches='tight', pad_inches=0)
plt.show()