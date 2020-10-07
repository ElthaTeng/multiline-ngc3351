import numpy as np
import matplotlib.pyplot as plt
import corner
import time
from astropy.io import fits

'''
This script generates a corner plot showing 1D and 2D likelihood distributions for
all six parameters (dimensions) by loading in the chi^2/probability arrays returned 
from 'bestfit_onepix.py' or 'bestfit_onepix_cov.py'. Note that 'bestfit_onepix.py'
only saves the chi^2 array, so the probability should be computed in this script. An
option for setting up priors/constraints is enabled by default and can be disenabled.

'''

start_time = time.time()
print('Constructing inputs for corner plot...')

idx_x = 36
idx_y = 43
model = '6d_coarse_cov'
sou_model = 'radex_model/'
chi2 = np.load(sou_model+'chi2_'+model+'_'+str(idx_x)+'_'+str(idx_y)+'.npy')

'''If use the output from 'bestfit_onepix_cov.py', just load in. Otherwise, compute probability here. '''
prob = np.load(sou_model+'prob_'+model+'_'+str(idx_x)+'_'+str(idx_y)+'.npy')
#prob = np.exp(-0.5*chi2).reshape(-1)

N_co = np.arange(16., 21.1, 0.2)
T_k = np.arange(1.,2.4,0.1)
n_h2 = np.arange(2., 5.1, 0.2)
X_12to13 = np.arange(10, 201, 10)
X_13to18 = np.arange(2, 21, 1.5)
phi = np.arange(0.05, 1.01, 0.05)

size_N = N_co.shape[0]
size_T = T_k.shape[0]
size_n = n_h2.shape[0]
size_x12to13 = X_12to13.shape[0]
size_x13to18 = X_13to18.shape[0]
size_phi = phi.shape[0]

N_co = N_co.reshape(size_N,1,1,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
T_k = T_k.reshape(1,size_T,1,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
n_h2 = n_h2.reshape(1,1,size_n,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
X_12to13 = X_12to13.reshape(1,1,1,size_x12to13,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
X_13to18 = X_13to18.reshape(1,1,1,1,size_x13to18,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
phi = phi.reshape(1,1,1,1,1,size_phi)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))

N_co = N_co.reshape(-1)
T_k = T_k.reshape(-1)
n_h2 = n_h2.reshape(-1)
X_12to13 = X_12to13.reshape(-1)
X_13to18 = X_13to18.reshape(-1)
phi = phi.reshape(-1)

'''Set priors (e.g. line-of-sight length) to exclude unreasonable parameter sets (optional) '''
los_max = 100.
x_co = 3 * 10**(-4)
map_ew = fits.open('data_image/NGC3351_CO10_ew_broad_nyq.fits')[0].data
map_fwhm = map_ew * 2.35  # Conversion of 1 sigma to FWHM assuming Gaussian
los_length = (10**N_co / 15. * map_fwhm[idx_y,idx_x]) / (np.sqrt(phi) * 10**n_h2 * x_co)  
mask = los_length < los_max * (3.086 * 10**18) 
print(np.sum(mask),'parameter sets have line-of-sight length smaller than',los_max,'pc')

chi2_masked = chi2 * mask
chi2_masked[mask == 0] = np.nan

idx_min = np.unravel_index(np.nanargmin(chi2_masked), (size_N,size_T,size_n,size_x12to13,size_x12to13,size_phi))
print('Minumum chi^2 =', np.nanmin(chi2_masked), 'at', idx_min)

par_min = np.asarray(idx_min)
Nco = np.round(0.2*par_min[0] + 16., 1)
Tk = np.round(0.1*par_min[1] + 1., 1)
nH2 = np.round(0.2*par_min[2] + 2., 1)
X12to13 = np.round(10*par_min[3] + 10., 1)
X13to18 = np.round(1.5*par_min[4] + 2., 1)
Phi = np.round(0.05*par_min[5] + 0.05, 2)
print('i.e. (Nco, Tk, nH2, X(12/13), X(13/18), Phi) =', Nco, Tk, nH2, X12to13, X13to18, Phi)

'''Apply mask if using priors; otherwise, comment the following line.'''
prob = prob * mask

# Corner plot
input = np.stack((N_co, T_k, n_h2, X_12to13, X_13to18, phi), axis=-1)

corner_time = time.time()
print(corner_time-start_time, 'sec elapsed. Generating corner plot')
figure = corner.corner(input, bins=[size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi], plot_datapoints=False, 
                       labels=[r"$N_{CO}$", r"$T_{kin}$", r"$n_{H_2}$", r"$X_{12/13}$", r"$X_{13/18}$", r"$\Phi_{bf}$"], 
                       show_titles=True, title_kwargs={"fontsize": 12}, weights=prob)

plt.savefig(sou_model+'corner_'+model+'_'+str(idx_x)+'_'+str(idx_y)+'.pdf')
end_time = time.time()
print(end_time-corner_time, 'sec elapsed. Corner plot saved.')
print('Total time elapsed =', end_time-start_time, 'sec')

plt.show(figure)

