import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.colors import LogNorm

model = 'bestfit'
fits_map = fits.open('data_image/NGC3351_CO10_ew_broad_nyq.fits')
map_ew = fits_map[0].data
map_fwhm = map_ew * 2.35
wcs = WCS(fits_map[0].header)
flux_co10 = np.load('data_image/NGC3351_CO10_mom0.npy')
model_Nco = np.floor(np.load('radex_model/Nco_6d_coarse_cov_los100_'+model+'.npy') * 10) / 10

mask = np.load('mask_139_pixels.npy')
# model_Nco_neg = np.load('radex_model/Nco_5d_fine_neg1sig.npy') * mask
# model_Nco_pos = np.load('radex_model/Nco_5d_fine_pos1sig.npy') * mask
# model_Nco_neg[model_Nco_neg == 0] = np.nan
# model_Nco_pos[model_Nco_pos == 0] = np.nan
# err_co10 = np.load('data_image/errors/NGC3351_CO10_emom0_broad_nyq.npy') * mask
# err_co10[err_co10 == 0] = np.nan 

Nco = 10**model_Nco / 15. * map_fwhm  #cm^-2  
Nco = Nco * mask
Nco[Nco == 0] = np.nan
Xco = 3. * 10**(-4)

masked_co10 = flux_co10 * mask
masked_co10[masked_co10 == 0] = np.nan  

# conversion factors
X_co = Nco / (Xco*flux_co10)  
alpha_co = X_co / (5.3*10**(19))

X_avg = np.nansum(Nco) / np.nansum(masked_co10) / Xco
alpha_avg = X_avg / (5.3*10**(19))
print(np.sum(~np.isnan(Nco)), np.sum(~np.isnan(masked_co10)))
print(np.nansum(Nco), np.nansum(masked_co10))
print('Average alpha_co is', alpha_avg)

# Conversion factor plot
fig = plt.figure()
ax = fig.add_subplot(111, projection=wcs)  #
ra = ax.coords[0]
ra.set_major_formatter('hh:mm:ss.s')
#plt.imshow(X_co, origin='lower', cmap='inferno', norm=LogNorm(vmin=3e18,vmax=1e22))
plt.imshow(alpha_co, origin='lower', cmap='inferno', norm=LogNorm(vmin=0.1,vmax=20))
plt.colorbar()
plt.contour(flux_co10,origin='lower',levels=(20,50,100,150,200,250,300), colors='grey', linewidths=1)
plt.xlim(15,60)
plt.ylim(15,60)
plt.xlabel('R.A. (J2000)')
plt.ylabel('Decl. (J2000)')
plt.show()

log_Nco = np.log10(Nco).reshape(-1)
# Nco_neg = 10**model_Nco_neg / 15. * map_fwhm
# Nco_pos = 10**model_Nco_pos / 15. * map_fwhm
# log_Nco_low = np.log10(Nco_neg).reshape(-1)
# log_Nco_up = np.log10(Nco_pos).reshape(-1)
# err_y_low = log_Nco - log_Nco_low
# err_y_up = log_Nco_up - log_Nco
log_Ico = np.log10(masked_co10).reshape(-1)
# err_x = np.log10(1 + err_co10/masked_co10).reshape(-1)

# Nco-to-Ico scatter plot
#plt.errorbar(log_Ico, log_Nco, xerr=err_x, yerr=[err_y_low,err_y_up], fmt='ko', ecolor='gray')
plt.plot(log_Ico, log_Nco, 'k.')
plt.xlabel(r'$\log\ I_{CO}\ (K\ km\ s^{-1})$')
plt.ylabel(r'$\log\ N_{CO}\ (cm^{-2})$')
plt.show()