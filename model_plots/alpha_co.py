import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.colors import LogNorm

# Load data
fits_map = fits.open('data_image/NGC3351_CO10_ew_broad_nyq.fits')
map_ew = fits_map[0].data
map_fwhm = map_ew * 2.35
wcs = WCS(fits_map[0].header)

model = '1dmax'
src = 'radex_model/Nco_6d_coarse_rmcor_whole_los50'
model_Nco = np.floor(np.load(src+'_'+model+'.npy') * 10) / 10
model_Nco_cent = np.floor(np.load(src+'_'+model+'.npy') * 10) / 10
model_Nco_med = np.floor(np.load(src+'_median.npy') * 10) / 10 
model_Nco_neg = np.floor(np.load(src+'_neg1sig.npy') * 10) / 10 
model_Nco_pos = np.floor(np.load(src+'_pos1sig.npy') * 10) / 10 

flux_co10 = fits.open('data_image/NGC3351_CO10_mom0_broad_nyq.fits')[0].data   
err_co10 = np.load('data_image/errors/NGC3351_CO10_emom0_broad_nyq.npy') 
radius = np.load('ngc3351_radius_arcsec.npy')

# Set mask for grouping
mask = np.load('mask_139_pixels.npy') * np.load('mask_rmcor_comb_lowchi2.npy') 
mask_cent = np.load('mask_139_pixels.npy') 

# Pre-processing / Masking
Nco = 10**model_Nco / 15. * map_fwhm  #cm^-2  
Nco = Nco * mask
Nco[Nco == 0] = np.nan
Nco[flux_co10 < 0] = np.nan

Ico = flux_co10 * mask
Ico[np.isnan(Nco)] = np.nan  

Nco_med = 10**model_Nco_med / 15. * map_fwhm
Nco_med[np.isnan(Nco)] = np.nan
Nco_neg = 10**model_Nco_neg / 15. * map_fwhm
Nco_neg[np.isnan(Nco)] = np.nan
Nco_pos = 10**model_Nco_pos / 15. * map_fwhm
Nco_pos[np.isnan(Nco)] = np.nan
err_Ico = np.copy(err_co10)
err_Ico[np.isnan(Nco)] = np.nan

# Useless if group == False
Nco_cent = 10**model_Nco_cent / 15. * map_fwhm  #cm^-2  
Nco_cent = Nco_cent * mask_cent
Nco_cent[Nco_cent == 0] = np.nan
Nco_cent[flux_co10 < 0] = np.nan

Ico_cent = flux_co10 * mask_cent 
Ico_cent[np.isnan(Nco_cent)] = np.nan  #masked_co10 == 0
Ico_cent[Ico_cent < 0] = np.nan

Nco_med_cent = 10**model_Nco_med / 15. * map_fwhm
Nco_med_cent[np.isnan(Nco_cent)] = np.nan
Nco_neg_cent = 10**model_Nco_neg / 15. * map_fwhm
Nco_neg_cent[np.isnan(Nco_cent)] = np.nan
Nco_pos_cent = 10**model_Nco_pos / 15. * map_fwhm
Nco_pos_cent[np.isnan(Nco_cent)] = np.nan
err_Ico_cent = np.copy(err_co10)
err_Ico_cent[np.isnan(Nco_cent)] = np.nan

def alpha_map(Nco, Ico):
    Xco = 3. * 10**(-4)
    X_co = Nco / (Xco*Ico)  
    alpha_co = X_co / (5.3*10**(19))

    X_avg = np.nansum(Nco) / np.nansum(Ico) / Xco
    alpha_avg = X_avg / (5.3*10**(19))
    
    print('Effective pixels:',np.sum(~np.isnan(Nco)), np.sum(~np.isnan(Ico))) 
    print('Average alpha_co is', alpha_avg)
    return alpha_co

def alpha_plot(map, cont_type, cont):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=wcs)  #
    ra = ax.coords[0]
    ra.set_major_formatter('hh:mm:ss.s')
    plt.imshow(map, origin='lower', cmap='inferno', norm=LogNorm(vmin=0.1,vmax=10))
    plt.colorbar()
    if cont_type == 'flux': 
        plt.contour(cont,origin='lower',levels=(20,50,100,150,200,250,300), colors='dimgray', linewidths=1)
    if cont_type == 'radius': 
        plt.contour(cont,origin='lower',levels=(5,10,15,20,25), colors='dimgray', linewidths=1)
    plt.xlim(15,60)
    plt.ylim(15,60)
    plt.xlabel('R.A. (J2000)')
    plt.ylabel('Decl. (J2000)')
    #plt.savefig('radex_model/factor_'+model+'_6d_coarse_rmcor_comb_los50.pdf')
    plt.show()

def NtoI_plot(group, errbar):
    log_Nco = np.log10(Nco).reshape(-1)
    log_Ico = np.log10(Ico).reshape(-1)
    if errbar:
        log_Nco_med = np.log10(Nco_med).reshape(-1)
        log_Nco_low = np.log10(Nco_neg).reshape(-1)
        log_Nco_up = np.log10(Nco_pos).reshape(-1)
        
        err_y_low = log_Nco_med - log_Nco_low
        err_y_up = log_Nco_up - log_Nco_med
        err_x_low = np.log10(Ico/(Ico - err_Ico)).reshape(-1)
        err_x_up = np.log10(1 + err_Ico/Ico).reshape(-1)
    if group:
        log_Nco_cent = np.log10(Nco_cent).reshape(-1)
        log_Ico_cent = np.log10(Ico_cent).reshape(-1)
        if errbar:
            log_Nco_cent_med = np.log10(Nco_med_cent).reshape(-1)
            log_Nco_cent_low = np.log10(Nco_neg_cent).reshape(-1)
            log_Nco_cent_up = np.log10(Nco_pos_cent).reshape(-1)
            
            err_y_cent_low = log_Nco_cent_med - log_Nco_cent_low
            err_y_cent_up = log_Nco_cent_up - log_Nco_cent_med
            err_x_cent_low = np.log10(Ico_cent/(Ico_cent - err_Ico_cent)).reshape(-1)
            err_x_cent_up = np.log10(1 + err_Ico_cent/Ico_cent).reshape(-1)
    
    # Plot
    x = np.arange(-0.5,2.6,0.1)
    y = x + 16. + np.log10(6)
    plt.plot(x, y, color='k', linestyle='-')
    if errbar:
        plt.errorbar(log_Ico, log_Nco_med, yerr=[err_y_low,err_y_up], xerr=[err_x_low,err_x_up], fmt='cx', ecolor='gray', capsize=3)  #xerr=[err_x_low,err_x_up], 
    plt.plot(log_Ico, log_Nco, c='b', marker='.', linestyle='', markersize=5, label='spiral arms')
    if group:
        if errbar:
            plt.errorbar(log_Ico_cent, log_Nco_cent_med, yerr=[err_y_cent_low,err_y_cent_up], xerr=[err_x_low,err_x_up], fmt='mx', ecolor='gray', capsize=3)  #xerr=[err_x_cent_low,err_x_cent_up], 
        plt.plot(log_Ico_cent, log_Nco_cent, c='r', marker='.', linestyle='', label='center')

    plt.xlabel(r'$\log\ I_{CO}\ (K\ km\ s^{-1})$')
    plt.ylabel(r'$\log\ N_{CO}\ (cm^{-2})$')
    if group:
        plt.legend()
    plt.show()

def alpha2rad_plot(alpha_co, rad):
    plt.semilogy(rad, alpha_co, c='k', marker='.', linestyle='')
    plt.xlabel('Radius (arcsec)')
    plt.ylabel(r'$\alpha_{CO}$ ($M_\odot\ (K\ km\ s^{-1}\ pc^2)^{-1}$)')
    plt.show()

# Start here
map = alpha_map(Nco, Ico)
alpha_plot(map, 'flux', flux_co10)  # 'flux', flux_co10 or 'radius', radius
alpha2rad_plot(map, radius)
NtoI_plot(group=False, errbar=True)

