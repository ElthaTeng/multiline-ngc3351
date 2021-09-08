import numpy as np
import matplotlib.pyplot as plt
import time

region = 'arms'
log_bins = False

start_time = time.time()
model = '6d_coarse'
sou_model = 'radex_model/'
sou_data = 'data_cube/'

# Set parameter ranges
N_co = np.arange(16.,21.1,0.2)
T_k = np.arange(1.,2.4,0.1)
n_h2 = np.arange(2.,5.1,0.2)
X_13co = np.arange(10,205,10)
X_c18o = np.arange(2,21,1.5)
phi = np.arange(0.05, 1.01, 0.05)

size_N = N_co.shape[0]
size_T = T_k.shape[0]
size_n = n_h2.shape[0]
size_X1 = X_13co.shape[0]
size_X2 = X_c18o.shape[0]
size_phi = phi.shape[0]

size_6d = size_N*size_T*size_n*size_X1*size_X2*size_phi

fit_result = np.load(sou_data+'fitting_'+region+'_2.npy')[6:,:]
fit_err = np.load(sou_data+'fitting_'+region+'_errors_2.npy')[6:,:]
flux_obs = 1.0645 * fit_result[:,0] * fit_result[:,2]
flux_err = np.full((2,),np.nan)
if region == 'arms':
    flux_err[0] = flux_obs[0] * (0.001/fit_result[0,0] + fit_err[0,2]/fit_result[0,2])
    flux_err[1] = 1.0645 * 1 * 0.002 * fit_result[0,2] # 1 sigma upper bound + assume CO 2-1 line width
else:
    flux_err = flux_obs * (np.array((0.002,0.001))/fit_result[:,0] + fit_err[:,2]/fit_result[:,2])  
noise_10 = np.sqrt((0.1*flux_obs)**2 + flux_err**2)

# Compute chi2 and prob
chi2 = np.load(sou_model+'chi2_'+model+'_rmcor_'+region+'.npy').reshape(-1)
mask = np.load(sou_model+'mask_'+model+'_rmcor_'+region+'_los100.npy')
prob = np.exp(-0.5*chi2).reshape(-1) * mask

# Load line intensity grids
if log_bins:
    flux_mod_13co10 = np.log10(np.load(sou_model+'flux_'+model+'_10_13co10.npy').reshape(-1))
    flux_mod_c18o10 = np.log10(np.load(sou_model+'flux_'+model+'_10_c18o10.npy').reshape(-1))
else:
    flux_mod_13co10 = np.load(sou_model+'flux_'+model+'_10_13co10.npy').reshape(-1)
    flux_mod_c18o10 = np.load(sou_model+'flux_'+model+'_10_c18o10.npy').reshape(-1)

flux_mod = np.array((flux_mod_13co10, flux_mod_c18o10))

model_time = time.time()
print('Models loaded.', round(model_time - start_time, 1), 'sec elapsed.')
print('Start marginalizing intensity grids...')

# 1D likelihoods of line intensities 
if log_bins:
    if region == 'arms':
        num_bins = np.array((15, 15))
        ranges = np.array(([-1, -2],[0.5, -0.5]))
    else:
        num_bins = np.array((15, 15))
        #ranges = np.array(([0.3, -0.5],[1.3, 0.5]))
        ranges = np.array(([0., -0.8],[1.5, 0.7]))
else:
    if region == 'arms':
        num_bins = np.array((20, 20))
        ranges = np.array(([0, 0],[2, 0.2]))
    else:
        num_bins = np.array((15, 12))
        ranges = np.array(([0, 0],[15, 3]))

line = np.array(('13CO10', 'C18O10'))
title = np.array((r'$^{13}$CO 1-0', r'C$^{18}$O 1-0'))

plt.figure(figsize=(7,3))
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.title(title[i])
    counts_noweight, bins = np.histogram(flux_mod[i], bins=num_bins[i], range=ranges[:,i], weights=None, density=True)
    counts_weighted, bins = np.histogram(flux_mod[i], bins=num_bins[i], range=ranges[:,i], weights=prob, density=True)
    counts_norm = np.nan_to_num(counts_weighted / counts_noweight)

    quantile_values = np.array((0.16,0.5,0.84),dtype='float32')
    cdf = np.cumsum(counts_norm)
    cdf /= cdf[-1]
    pos1sig = np.interp(quantile_values[2], cdf, bins[:-1])
    neg1sig = np.interp(quantile_values[0], cdf, bins[:-1])

    plt.hist(bins[:-1], bins, weights=counts_norm, log=False, histtype='step', color='k')
    ax = plt.gca()
    ax.axes.yaxis.set_visible(False)
    if log_bins:
        plt.axvline(x=np.log10(flux_obs[i]+noise_10[i]), linewidth=1, color='b', linestyle='--')
        plt.axvline(x=np.log10(flux_obs[i]-noise_10[i]), linewidth=1, color='b', linestyle='--')
        plt.xlabel(r'$\log$ Intensity (K km $\rm s^{-1}$)')
    else:
        if region == 'arms' and i == 1:
            plt.axvspan(0, flux_obs[i]+noise_10[i], alpha=0.2, color='gray')
            plt.axvline(x=flux_obs[i]+noise_10[i], linewidth=1, color='k', linestyle='dotted')
        else:
            plt.axvspan(flux_obs[i]-noise_10[i], flux_obs[i]+noise_10[i], alpha=0.2, color='gray')
            plt.axvline(x=flux_obs[i], linewidth=1, color='k', linestyle='dotted')
        plt.xlabel(r'Intensity (K km $\rm s^{-1}$)')
    #plt.axvline(x=pos1sig, linewidth=1, color='r', linestyle='--')
    #plt.axvline(x=neg1sig, linewidth=1, color='r', linestyle='--')
    
plt.subplots_adjust(wspace=0.1)
#plt.subplots_adjust(hspace=0.5)
#plt.savefig('radex_model/prob1d_flux10_marg_'+region+'_cal10_lin_2.pdf', bbox_inches='tight', pad_inches=0.1)

end_time = time.time()
print('1D likelihoods generated.', round(end_time - model_time, 1), 'sec elapsed.')
print(flux_err, noise_10)
plt.show()

