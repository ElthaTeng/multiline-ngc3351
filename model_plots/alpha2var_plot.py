import numpy as np
import matplotlib.pyplot as plt

'''This script outputs the scatter plot of the median/1DMax alpha_CO versus another specified parameter.'''

def as2kpc(x):
    return x * 0.0485
    
def kpc2as(x):
    return x / 0.0485

input = 'radius'
two_category = False
model = '6d_coarse'
mask = np.load('mask_whole_recovered.npy')*np.load('mask_rmcor_comb_lowchi2.npy')  #np.load('mask_cent3sig.npy') #* np.load('old_masks/mask_armscent.npy') + np.load('mask_poop.npy')  
#(np.load('radex_model/chi2_4d_2comp.npy') < 10)  # #* np.load('mask_13co21_1sig.npy')
mask[35,43] = 0 
mask = mask.reshape(-1)

if input == 'radius':
    var = np.load('ngc3351_radius_arcsec.npy').reshape(-1)
elif input == 'tau':
    var = np.log10(np.load('radex_model/tau_6d_coarse_rmcor_whole_los100_1dmax_co10.npy').reshape(-1))
elif input == 'Tk': 
    var = np.load('radex_model/Tk_6d_coarse_rmcor_whole_los100_median_interp.npy').reshape(-1)
elif input == 'ratio': 
    var = np.load('data_image/ratio_CO21_CO10.npy').reshape(-1)
elif input == 'Nco': 
    var = np.load('radex_model/Nco_6d_coarse_rmcor_whole_los100_median_interp.npy').reshape(-1)
elif input == 'Ico': 
    var = np.log10(np.load('data_image/NGC3351_CO21_mom0.npy').reshape(-1))
    #var[var == 0] = np.nan

alpha = np.load('radex_model/Xco_'+model+'_alpha_1dmax_los100.npy').reshape(-1) * mask 
alpha[mask == 0] = np.nan
alpha_med = np.load('radex_model/Xco_'+model+'_alpha_median_los100.npy').reshape(-1) * mask 
alpha_med[mask == 0] = np.nan
alpha_pos = np.load('radex_model/Xco_'+model+'_alpha_pos1sig_los100.npy').reshape(-1) * mask 
alpha_pos[mask == 0] = np.nan
alpha_neg = np.load('radex_model/Xco_'+model+'_alpha_neg1sig_los100.npy').reshape(-1) * mask 
alpha_neg[mask == 0] = np.nan

err_y_up = alpha_pos - alpha_med
err_y_low = alpha_med - alpha_neg

if two_category:
    mask_2 = np.load('mask_arms.npy').reshape(-1)
    alpha_2 = np.load('radex_model/Xco_'+model+'_alpha_1dmax_los100.npy').reshape(-1) * mask_2 
    alpha_2[mask_2 == 0] = np.nan
    alpha_med_2 = np.load('radex_model/Xco_'+model+'_alpha_median_los100.npy').reshape(-1) * mask_2 
    alpha_med_2[mask_2 == 0] = np.nan
    alpha_pos_2 = np.load('radex_model/Xco_'+model+'_alpha_pos1sig_los100.npy').reshape(-1) * mask_2 
    alpha_pos_2[mask_2 == 0] = np.nan
    alpha_neg_2 = np.load('radex_model/Xco_'+model+'_alpha_neg1sig_los100.npy').reshape(-1) * mask_2 
    alpha_neg_2[mask_2 == 0] = np.nan
    err_y_up_2 = alpha_pos_2 - alpha_med_2
    err_y_low_2 = alpha_med_2 - alpha_neg_2


fig, ax = plt.subplots()
#ax2 = ax1.twiny()
if two_category:
    plt.errorbar(var, alpha_med, yerr=[err_y_low,err_y_up], linestyle='', marker='.', c='darkred', ecolor='gray', elinewidth=0.5, label='Center')
    ax.plot(var, alpha, c='gray', marker='_', linestyle='')
    plt.errorbar(var, alpha_med_2, yerr=[err_y_low_2,err_y_up_2], linestyle='', marker='.', c='darkblue', ecolor='gray', elinewidth=0.5, label='Arms')
    ax.plot(var, alpha_2, c='gray', marker='_', linestyle='')
    ax.axhline(0.65, c='k', linestyle='--')
    plt.legend(fontsize=12)
else:
    plt.errorbar(var, alpha_med, yerr=[err_y_low,err_y_up], linestyle='', marker='.', c='k', ecolor='gray', elinewidth=0.5)
    ax.plot(var, alpha, c='gray', marker='_', linestyle='')
    print(var.max())
    ax.axhline(0.65, c='darkred', linestyle='--')
ax.set_ylim(-2.5, 1.)
if input == 'tau':
    ax.set_xlim(-2., 2.)
    ax.set_xlabel(r'$\log\ \tau_{\rm CO(1-0)}$')  
elif input == 'radius':
    ax.set_xlabel('Galactocentric Radius (arcsec)')  
    secax = ax.secondary_xaxis('top', functions=(as2kpc, kpc2as))
    secax.set_xlabel('Galactocentric Radius (kpc)')
elif input == 'Tk': 
    ax.set_xlim(0.9, 2.)
    ax.set_xlabel(r'$\log\ T_k$ (K)')  
elif input == 'Nco': 
    ax.set_xlabel(r'$\log \left( N_{CO}\cdot\frac{15\ km\ s^{-1}}{\Delta v}\right)$  $(cm^{-2})$')
else:
    ax.set_xlabel(r'$\log\ I_{CO(2-1)}$  $(K\ km\ s^{-1})$')
ax.set_ylabel(r'$\log\ \alpha_{CO}$ ($M_\odot\ (K\ km\ s^{-1}\ pc^2)^{-1}$)')
plt.tight_layout()
plt.show()

