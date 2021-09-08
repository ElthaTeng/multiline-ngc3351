from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

region = 'center'  # center, ring, arms
line = np.array(('CO10','CO21','13CO21','13CO32','C18O21','C18O32'))  #,'13CO10','C18O10'
title = np.array(('(a) CO 1-0','(b) CO 2-1',r'(c) $\rm {}^{13}$CO 2-1',r'(d) $\rm {}^{13}$CO 3-2',r'(e) $\rm {C}^{18}$O 2-1',r'(f) $\rm {C}^{18}$O 3-2'))  #,r'(g) $\rm {}^{13}$CO 1-0',r'(h) $\rm {C}^{18}$O 1-0'
num = line.shape[0]

if region == 'center':
    mask = np.load('mask_cent3sig.npy')
elif region == 'ring':
    mask = np.load('mask_ring.npy')
elif region == 'arms':
    mask = np.load('mask_arms.npy') * np.load('old_masks/mask_whole.npy')

indices = np.stack((np.nonzero(mask)[0],np.nonzero(mask)[1]),axis=1)
N_pix = indices.shape[0]

v_extract = np.arange(-40,40)
v_li = np.arange(-40,40,0.01)
fit_result = np.full((num,3), np.nan)
fit_error = np.full((num,3), np.nan)

fig = plt.figure(figsize=(15,6))
for obs in range(num):
    spec_stacked = np.zeros((80,))
    file = fits.open('data_cube/NGC3351_'+line[obs]+'_cube_nyq.fits')

    for i in range(N_pix):
        spec = spec_extract(file, indices[i,1],indices[i,0])
        spec_stacked += spec
    # print(spec_stacked)

    spec_avg = spec_stacked/N_pix

    if region == 'arms' and (obs == 5 or obs == 7):
        ic = [spec_avg.max(), -15., 10.]
    else:
        ic = [np.nanmax(spec_avg), 0., 30.]
    popt, pcov = curve_fit(Gauss, v_extract, spec_avg, p0 = ic)  
    errors = np.sqrt(np.diagonal(pcov))
    print('Peak =', np.max(spec_avg), popt[0])
    print('Vpeak =', popt[1], 'km/s')
    print('Line width =', popt[2], 'km/s')
    print(errors)
    fit_result[obs] = popt
    fit_error[obs] = errors
    
    plt.subplot(2,num/2,obs+1)
    plt.step(v_extract,spec_avg,c='#8E8E8E',lw=2,where='mid')
    if region != 'arms' or obs != 5:
        plt.plot(v_li, Gauss(v_li,popt[0],popt[1],popt[2]), c='r', lw=1)
    lw = np.round(popt[2],1)
    lw_err = np.round(errors[2],1)
    plt.annotate(r'$\Delta v$ = '+str(lw)+r'$\pm$'+str(lw_err)+r' km/s', fontsize=10, xy=(0.02, 0.9), xycoords='axes fraction', color='k')
    
    plt.title(title[obs], fontsize=14)
    if obs//(num/2) == 1:
        plt.xlabel('Velocity (km/s)', fontsize=12)
    else:
        plt.tick_params(axis="x", labelbottom=False)
    if obs%(num/2) == 0:
        plt.ylabel('Intensity (K)', fontsize=12)

#print(fit_result,fit_error)
#np.save('data_cube/fitting_'+region+'.npy',fit_result)
#np.save('data_cube/fitting_'+region+'_errors.npy',fit_error)
plt.subplots_adjust(wspace=0.2)
plt.subplots_adjust(hspace=0.2)
plt.savefig('data_cube/specfit_'+region+'_lw.pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()

