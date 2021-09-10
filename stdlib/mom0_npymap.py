import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

source = 'NGC3351'
line = np.array(('CO10','CO21','13CO21','13CO32','C18O21','C18O32'))
num = line.shape[0]

for i in range(num):
    fits_map = fits.open('data_image/'+source+'_'+line[i]+'_mom0_broad_nyq.fits')[0].data
    fits_err = fits.open('data_image/errors/'+source+'_'+line[i]+'_emom0_broad_nyq.fits')[0].data
    
    fits_map[fits_map < 0] = 0
    if i > 3:   # 1 sigma cutoff for C18O lines
        fits_map[fits_map < fits_err] = 0
        fits_err[fits_map < fits_err] = 0
    else:   # 3 sigma cutoff
        fits_map[fits_map < 3 * fits_err] = 0
        fits_err[fits_map < 3 * fits_err] = 0
        
    np.save('data_image/'+source+'_'+line[i]+'_mom0.npy',fits_map)
    np.save('data_image/errors/'+source+'_'+line[i]+'_emom0_broad_nyq.npy',fits_err)
'''    
    plt.imshow(fits_map, origin='lower', cmap='hot')
    plt.colorbar()
    plt.show()
'''