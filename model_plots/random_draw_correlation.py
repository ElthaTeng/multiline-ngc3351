import time
import numpy as np
from joblib import Parallel, delayed
from astropy.io import fits

start_time = time.time()

two_category = True
input = 'Tk'
model = '6d_coarse'
sou_model = 'radex_model/'
sou_data = 'data_image/'

# Set parameter ranges
samples_Nco = np.arange(16.,21.1,0.2)
samples_Tk = np.arange(1.,2.4,0.1)
samples_nH2 = np.arange(2.,5.1,0.2)
samples_X12to13 = np.arange(10,205,10)
samples_X13to18 = np.arange(2,21,1.5)
samples_phi = np.arange(0.05, 1.01, 0.05)

size_N = samples_Nco.shape[0]
size_T = samples_Tk.shape[0]
size_n = samples_nH2.shape[0]
size_phi = samples_phi.shape[0]
size_x12to13 = samples_X12to13.shape[0]
size_x13to18 = samples_X13to18.shape[0]

# Set up constraints if using priors (optional)
los_max = 100.
x_co = 3 * 10**(-4)
Xco2alpha = 1 / (4.5 * 10**19) / x_co
map_ew = fits.open('data_image/NGC3351_CO10_ew_broad_nyq.fits')[0].data.astype('float32')
map_fwhm = map_ew * 2.35  # Conversion of 1 sigma to FWHM assuming Gaussian

if input == 'tau10':
    par_grid = np.load(sou_model+'tau_6d_coarse_co10.npy').reshape(-1)
elif input == 'Tk':
    par_grid = samples_Tk.reshape(1,size_T,1,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
    par_grid = par_grid.reshape(-1)

num_samples = 1000
map_mask = np.load('mask_whole_recovered.npy') * np.load('mask_cent3sig.npy') 
idx_y = np.where(map_mask==1)[0]
idx_x = np.where(map_mask==1)[1]
sampling_alpha = np.full((idx_x.shape[0], num_samples), np.nan)
sampling_par = np.full((idx_x.shape[0], num_samples), np.nan)

if two_category:
    map_mask_2 = np.load('mask_whole_recovered.npy') * np.load('mask_arms.npy') 
    idx_y_2 = np.where(map_mask_2==1)[0]
    idx_x_2 = np.where(map_mask_2==1)[1]
    sampling_alpha_2 = np.full((idx_x_2.shape[0], num_samples), np.nan)
    sampling_par_2 = np.full((idx_x_2.shape[0], num_samples), np.nan)

Nco = samples_Nco.reshape(size_N,1,1,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
nH2 = samples_nH2.reshape(1,1,size_n,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
phi = samples_phi.reshape(1,1,1,1,1,size_phi)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
Nco = Nco.reshape(-1)
nH2 = nH2.reshape(-1)
phi = phi.reshape(-1)

model_co10 = np.load(sou_model+'flux_'+model+'_co10.npy')
model_co21 = np.load(sou_model+'flux_'+model+'_co21.npy')
model_13co21 = np.load(sou_model+'flux_'+model+'_13co21.npy')
model_13co32 = np.load(sou_model+'flux_'+model+'_13co32.npy')
model_c18o21 = np.load(sou_model+'flux_'+model+'_c18o21.npy')
model_c18o32 = np.load(sou_model+'flux_'+model+'_c18o32.npy') 

size = size_N*size_T*size_n*size_x12to13*size_x13to18*size_phi

flux_co10 = fits.open(sou_data+'NGC3351_CO10_mom0_broad_nyq.fits')[0].data.astype('float32')
flux_co21 = fits.open(sou_data+'NGC3351_CO21_mom0_broad_nyq.fits')[0].data.astype('float32')
flux_13co21 = fits.open(sou_data+'NGC3351_13CO21_mom0_broad_nyq.fits')[0].data.astype('float32')
flux_13co32 = fits.open(sou_data+'NGC3351_13CO32_mom0_broad_nyq.fits')[0].data.astype('float32')
flux_c18o21 = fits.open(sou_data+'NGC3351_C18O21_mom0_broad_nyq.fits')[0].data.astype('float32')
flux_c18o32 = fits.open(sou_data+'NGC3351_C18O32_mom0_broad_nyq.fits')[0].data.astype('float32')

noise_co10 = fits.open(sou_data+'errors/NGC3351_CO10_emom0_broad_nyq.fits')[0].data.astype('float32')
noise_co21 = fits.open(sou_data+'errors/NGC3351_CO21_emom0_broad_nyq.fits')[0].data.astype('float32')
noise_13co21 = fits.open(sou_data+'errors/NGC3351_13CO21_emom0_broad_nyq.fits')[0].data.astype('float32')
noise_13co32 = fits.open(sou_data+'errors/NGC3351_13CO32_emom0_broad_nyq.fits')[0].data.astype('float32')
noise_c18o21 = fits.open(sou_data+'errors/NGC3351_C18O21_emom0_broad_nyq.fits')[0].data.astype('float32')
noise_c18o32 = fits.open(sou_data+'errors/NGC3351_C18O32_emom0_broad_nyq.fits')[0].data.astype('float32')

err_co10 = np.sqrt(noise_co10**2 + (0.1 * flux_co10)**2)
err_co21 = np.sqrt(noise_co21**2 + (0.1 * flux_co21)**2)
err_13co21 = np.sqrt(noise_13co21**2 + (0.1 * flux_13co21)**2)
err_13co32 = np.sqrt(noise_13co32**2 + (0.1 * flux_13co32)**2)
err_c18o21 = np.sqrt(noise_c18o21**2 + (0.1 * flux_c18o21)**2)
err_c18o32 = np.sqrt(noise_c18o32**2 + (0.1 * flux_c18o32)**2)

data_time = time.time()
print('Models & data loaded;', data_time-start_time, 'sec elapsed.')

def chi2_prob(x,y):
    chi_sum = ((model_co10 - flux_co10[y,x]) / err_co10[y,x])**2
    chi_sum += ((model_co21 - flux_co21[y,x]) / err_co21[y,x])**2
    chi_sum += ((model_13co21 - flux_13co21[y,x]) / err_13co21[y,x])**2
    chi_sum += ((model_13co32 - flux_13co32[y,x]) / err_13co32[y,x])**2
    chi_sum += ((model_c18o21 - flux_c18o21[y,x]) / err_c18o21[y,x])**2
    chi_sum += ((model_c18o32 - flux_c18o32[y,x]) / err_c18o32[y,x])**2
    
    prob = np.nan_to_num(np.exp(-0.5 * chi_sum)).reshape(-1)   
    return chi_sum.reshape(-1), prob

def random_draw(idx):
    if second_category:
        x = idx_x_2[idx]
        y = idx_y_2[idx]  
    else:
        x = idx_x[idx]
        y = idx_y[idx]
    chi2, prob = chi2_prob(x,y)

    los_length = (10**Nco / 15. * map_fwhm[y,x]) / (np.sqrt(phi) * 10**nH2 * x_co)
    mask = los_length < los_max * (3.086 * 10**18) 
    chi2 = np.array(chi2) * mask
    chi2[mask==0] = np.nan
    prob = np.array(prob) * mask
    
    if (np.isnan(chi2)).all():
        return 0, np.nan
    else:        
        alpha = 10**Nco / 15. * map_fwhm[y,x] * phi / model_co10.reshape(-1) * mask * Xco2alpha
        alpha[mask == 0] = np.nan
        prob = prob / np.nansum(prob)
        draw_idx = np.random.choice(size, num_samples, p=prob)
        draw_alpha = alpha[draw_idx]
        return draw_idx, draw_alpha

print('Start multi-processing on random draws over pixels')

second_category = False

results = Parallel(n_jobs=5, verbose=10)(delayed(random_draw)(idx) for idx in range(idx_x.shape[0]))
for count, result in enumerate(results):
    draw_idx, draw_alpha = result 
    sampling_alpha[count,:] = draw_alpha
    sampling_par[count,:] = par_grid[draw_idx]

np.save(sou_model+'random_draw_correlation_'+model+'_'+input+'_alpha_center.npy', sampling_alpha)
np.save(sou_model+'random_draw_correlation_'+model+'_'+input+'_center.npy', sampling_par)

if two_category:
    second_category = True
    results_2 = Parallel(n_jobs=5, verbose=10)(delayed(random_draw)(idx) for idx in range(idx_x_2.shape[0]))
    for count, result in enumerate(results_2):
        draw_idx_2, draw_alpha_2 = result
        sampling_alpha_2[count,:] = draw_alpha_2
        sampling_par_2[count,:] = par_grid[draw_idx_2]

np.save(sou_model+'random_draw_correlation_'+model+'_'+input+'_alpha_arms.npy', sampling_alpha_2)
np.save(sou_model+'random_draw_correlation_'+model+'_'+input+'_arms.npy', sampling_par_2)
    
end_time = time.time()
print('Total elapsed time: %s sec' % ((end_time - start_time)))
