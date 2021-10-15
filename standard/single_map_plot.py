import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits
from matplotlib.colors import LogNorm

mom0 = np.load('data_image/NGC3351_CO10_mom0.npy')
fits_map = fits.open('data_image/NGC3351_CO10_mom0_broad_nyq.fits')
wcs = WCS(fits_map[0].header)

map = fits.open('data_image/NGC3351_CO21_ew_broad_nyq.fits')[0].data  
mask = np.load('mask_whole_recovered.npy') 
map_masked = map * mask
map_masked[mask == 0] = np.nan

fig = plt.figure()

ax = fig.add_subplot(111, projection=wcs)
ra = ax.coords[0]
ra.set_major_formatter('hh:mm:ss.s')

plt.imshow(map_masked, cmap='viridis', origin='lower', vmin=0, vmax=30)   #, norm=LogNorm(vmin=0.05,vmax=50)
plt.tick_params(axis="y", labelleft=False)
#plt.tick_params(axis="x", labelbottom=False)
plt.colorbar()
#cb = plt.colorbar()
#cb.ax.plot(-0.25, 0.65, 'k.')
plt.contour(mom0,origin='lower',levels=(20,50,100,150,200,250), colors='dimgray', linewidths=1)
plt.title(r'(b) Effective Line Width of CO 2-1', fontsize=16) 
plt.xlim(15,60)   
plt.ylim(15,60)  
plt.xlabel('R.A. (J2000) ') # 
plt.ylabel(' ') # Decl. (J2000)

plt.savefig('data_image/ew_co21.pdf', bbox_inches='tight', pad_inches=0.1)  # tau_6d_coarse_rmcor_whole_los100_1dmax_co10
plt.show()
