import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

source = 'NGC3351'
line = np.array(('CO10','CO21','13CO21','13CO32','C18O21','C18O32'))
title = np.array(('(a) CO 1-0','(b) CO 2-1',r'(c) $\rm {}^{13}$CO 2-1',r'(d) $\rm {}^{13}$CO 3-2',r'(e) $\rm {C}^{18}$O 2-1',r'(f) $\rm {C}^{18}$O 3-2'))

fits_map = fits.open('data_image/'+source+'_'+line[2]+'_mom0_broad_nyq.fits')
wcs = WCS(fits_map[0].header)

fig = plt.figure(figsize=(18,10))

for i in range(6):
    mom0 = np.load('data_image/'+source+'_'+line[i]+'_mom0.npy')
    mom0[mom0 < 0] = 0
    ax = fig.add_subplot(2,3,i+1, projection=wcs)
    ra = ax.coords[0]
    ra.set_major_formatter('hh:mm:ss.s')
    
    zeros = np.ma.masked_where(~(mom0 == 0), mom0)
    plt.imshow(mom0, cmap='hot', origin='lower')
    plt.colorbar()
    plt.imshow(zeros, origin='lower', cmap='Pastel2_r')
    
    #cbar = plt.colorbar()
    #cbar.ax.tick_params(labelsize=11) 
    if i == 0:
        ax.annotate("contact", color='b', fontsize=14,
            xy=(34,43.5), xycoords='data',
            xytext=(45, 38), textcoords='data',
            arrowprops=dict(arrowstyle="->", lw=2,
                            connectionstyle="arc3", color='b'),
            )
        ax.annotate("points", color='b', fontsize=14,
            xy=(38.5,29.5), xycoords='data',
            xytext=(45.5, 35), textcoords='data',
            arrowprops=dict(arrowstyle="->", lw=2,
                            connectionstyle="arc3", color='b'),
            )
    if i == 1:
        beam = plt.Circle((19, 19), 1, color='b')
        ax.add_patch(beam)
        plt.plot([47, 57], [20, 20], 'b-', lw=3)
        plt.annotate('500 pc', weight='bold', fontsize=14, xy=(47.5, 17), xycoords='data', color='b')
    
    plt.title(title[i], fontsize=14)
    plt.xlim(15,60)   
    plt.ylim(15,60)  
    
    if i//3 == 0:
        plt.xlabel(' ')
        plt.tick_params(axis="x", labelbottom=False)
    else:
        plt.xlabel('R.A. (J2000)')
    if i%3 == 0:
        plt.ylabel('Decl. (J2000)')
    else:
        plt.ylabel(' ')
        plt.tick_params(axis="y", labelleft=False)
    
plt.subplots_adjust(hspace=0.1)
plt.subplots_adjust(wspace=0.1)
plt.savefig('formal_plots/mom0_subplots.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
