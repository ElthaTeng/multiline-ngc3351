import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Patch
from astropy.io import fits
from astropy.wcs import WCS
import copy

mask = np.load('mask_whole_recovered.npy')
reg_ring = np.load('mask_ring.npy').astype('float')
reg_arms = (np.load('mask_arms.npy') * mask).astype('float')
reg_cent = np.load('mask_cent3sig.npy').astype('float')
radius = np.load('ngc3351_radius_arcsec_shiftlf.npy')
mom0 = np.load('data_image/NGC3351_CO21_mom0.npy')

fits_map = fits.open('data_image/NGC3351_CO21_mom0_broad_nyq.fits')
wcs = WCS(fits_map[0].header)

reg_cent[reg_cent==0] = np.nan
reg_ring[reg_ring==0] = np.nan
reg_arms[reg_arms==0] = np.nan

cmap_cent = copy.copy(plt.cm.get_cmap('cool')) 
cmap_cent.set_bad(alpha=0) 

cmap_ring = copy.copy(plt.cm.get_cmap('viridis')) 
cmap_ring.set_bad(alpha=0) 

cmap_arms = copy.copy(plt.cm.get_cmap('cool_r')) 
cmap_arms.set_bad(alpha=0) 


fig = plt.figure()
ax = fig.add_subplot(111, projection=wcs) #
ra = ax.coords[0]
ra.set_major_formatter('hh:mm:ss.s')

plt.imshow(reg_cent, origin='lower', cmap='cool', vmin=0, vmax=1)
plt.imshow(reg_ring, origin='lower', cmap=cmap_ring, vmin=0, vmax=1)
plt.imshow(reg_arms, origin='lower', cmap=cmap_arms, vmin=0, vmax=1)

plt.tick_params(axis="y", labelleft=False)

ring = Patch(facecolor='gold')
nucleus = Patch(facecolor='magenta')
arms = Patch(facecolor='cyan')
null = Patch(facecolor='w')

lprop = {'weight':'bold', 'size':'large'}
plt.legend(handles=[nucleus, null, null, ring, ring, arms],
          labels=['+', '', '', 'Center', 'Ring', 'Arms'],
          ncol=2, handletextpad=0.5, handlelength=1.0, columnspacing=0.5, prop=lprop,
          loc='lower right', fontsize=18)

plt.contour(mom0,origin='lower',levels=(20,50,100,150,200,250), colors='k', linewidths=1)
cont_radius = plt.contour(radius,origin='lower',levels=(10,20), colors='darkred', linewidths=2)
manual_locations = [(44, 41), (51, 42)]
plt.gca().clabel(cont_radius, inline=1, fontsize=10, fmt='%1.0f', manual=manual_locations) #
plt.xlim(15,60)
plt.ylim(15,60)
plt.title('(c) Definition of Regions', fontsize=16)
plt.xlabel('R.A. (J2000)') #
plt.ylabel(' ') #Decl. (J2000)
plt.savefig('formal_plots/region_def.pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()

