from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

'''This script plots the regional averaged spectra (with stacking) and their best-fit Gaussian function.''' 

def Gauss(x,peak,vlsr,delv):
    sig = delv / (2*(2*np.log(2))**0.5)
    G = peak * np.exp(-(x-vlsr)**2/(2*sig**2))
    return G

def spec_extract(file, idx_x, idx_y):
    mom1 = fits.open('data_image/NGC3351_CO21_mom1_broad_nyq.fits')[0].data[idx_y,idx_x]
    cube = file[0].data
    header = file[0].header
    v_ref = header['CRVAL3']/1000
    v_del = header['CDELT3']/1000
    RefPix = header['CRPIX3']
    N_vel = header['NAXIS3']
    
    v_list = np.arange(v_ref-v_del*(RefPix-1), v_ref+v_del*(-RefPix+N_vel+0.5), v_del)
    mom1_idx = (np.abs(v_list - mom1)).argmin()
    spec = cube[mom1_idx-40:mom1_idx+40,idx_y,idx_x]
    return spec

line = np.array(('CO10','CO21','13CO21','13CO32','C18O21','C18O32'))
mask = np.load('mask_arms.npy')
indices = np.stack((np.nonzero(mask)[0],np.nonzero(mask)[1]),axis=1)
N_pix = indices.shape[0]

v_extract = np.arange(-40,40)
v_li = np.arange(-40,40,0.01)
fit_result = np.full((6,3), np.nan)
fit_error = np.full((6,3), np.nan)

for obs in range(6):
    spec_stacked = np.zeros((80,))
    file = fits.open('data_cube/NGC3351_'+line[obs]+'_cube_nyq.fits')

    for i in range(N_pix):
        spec = spec_extract(file, indices[i,1],indices[i,0])
        spec_stacked += spec
        #print(spec_stacked)

    spec_avg = spec_stacked/N_pix
    if obs == 5:
        ic = [spec_avg.max(), -15., 10.]
    else:
        ic = [spec_avg.max(), 0., 30.]
    popt, pcov = curve_fit(Gauss, v_extract, spec_avg, p0 = ic)  
    errors = np.sqrt(np.diagonal(pcov))
    # print('Peak =', np.max(spec_avg), popt[0])
    # print('Vpeak =', popt[1], 'km/s')
    # print('Line width =', popt[2], 'km/s')
    # print(errors)
    fit_result[obs] = popt
    fit_error[obs] = errors

    plt.subplot(2,3,obs+1)
    plt.step(v_extract,spec_avg,c='#8E8E8E',lw=2,where='mid')
    plt.plot(v_li, Gauss(v_li,popt[0],popt[1],popt[2]), c='r', lw=1)
    plt.title('Averaged spectrum of '+line[obs])
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Intensiity (K)')

print(fit_result,fit_error)
np.save('fitting_arms.npy',fit_result)
np.save('fitting_arms_errors.npy',fit_error)
plt.subplots_adjust(wspace=0.3)
plt.subplots_adjust(hspace=0.5)
plt.show()

