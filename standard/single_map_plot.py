import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits
from matplotlib.colors import LogNorm

mom0 = np.load('data_image/NGC3351_CO10_mom0.npy')
fits_map = fits.open('data_image/NGC3351_CO10_mom0_broad_nyq.fits')
wcs = WCS(fits_map[0].header)

map = fits.open('data_image/NGC3351_CO21_mom1_broad_nyq.fits')[0].data  
mask = np.load('mask_whole_recovered.npy')
map_masked = map * mask
map_masked[mask == 0] = np.nan

fig = plt.figure()

ax = fig.add_subplot(111, projection=wcs)
ra = ax.coords[0]
ra.set_major_formatter('hh:mm:ss.s')

plt.imshow(map_masked, cmap='coolwarm', origin='lower')  
plt.tick_params(axis="y", labelsize=14, labelleft=True)
plt.tick_params(axis="x", labelsize=14, labelbottom=True)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
#cb.ax.plot(-0.25, 0.65, 'k.')
#plt.contour(mom0,origin='lower',levels=(20,50,100,150,200,250), colors='dimgray', linewidths=1)
plt.title(r'(c) Moment 1 of CO 2-1', fontsize=16) 
plt.xlim(15,60)   
plt.ylim(15,60)  
plt.xlabel('R.A. (J2000) ', fontsize=14) # 
plt.ylabel('Decl. (J2000)', fontsize=14) # 

plt.savefig('formal_plots/mom1_co21.pdf', bbox_inches='tight', pad_inches=0.1)  
plt.show()
