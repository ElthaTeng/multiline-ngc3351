import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits

source = 'NGC3351'
mom0 = np.load('data_image/'+source+'_CO21_mom0.npy')

line_up = 'CO21'
line_low = 'CO10'
map_up = np.load('data_image/'+source+'_'+line_up+'_mom0.npy')
map_low = np.load('data_image/'+source+'_'+line_low+'_mom0.npy')

fits_map = fits.open('data_image/'+source+'_CO21_mom0_broad_nyq.fits')
wcs = WCS(fits_map[0].header)

ratio = map_up/map_low
ratio[ratio<=0] = np.nan

np.save('data_image/ratio_'+line_up+'_'+line_low+'.npy', ratio)

fig = plt.figure()
ax = fig.add_subplot(111, projection=wcs)
ra = ax.coords[0]
ra.set_major_formatter('hh:mm:ss.s')
plt.imshow(ratio, vmin=0.5, vmax=1.5, cmap='Spectral_r', origin='lower')
plt.colorbar()
plt.contour(mom0,origin='lower',levels=(20,50,100,150,200,250), colors='k', linewidths=1)
plt.title('(a) CO 2-1/1-0', fontsize=14) 
plt.xlim(15,60)   
plt.ylim(15,60)  
plt.xlabel('R.A. (J2000)')
plt.ylabel('Decl. (J2000)')
plt.savefig('formal_plots/line_ratios/'+source+'_'+line_up+'_'+line_low+'_ratio.pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()
