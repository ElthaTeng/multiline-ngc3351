import numpy as np
import time
import corner
from joblib import Parallel, delayed
from astropy.io import fits

start_time = time.time()

model = '6d_coarse'
output_map = 'bestfit' 
sou_model = 'radex_model/'
sou_data = 'data_image/'
nobs = 6

# Apply mask or set up constraints if using priors (optional)
# mask = np.load('mask_los50.npy')
los_max = 50.
x_co = 3 * 10**(-4)
map_ew = fits.open('data_image/NGC3351_CO10_ew_broad_nyq.fits')[0].data
map_fwhm = map_ew * 2.35  # Conversion of 1 sigma to FWHM assuming Gaussian

# Set parameter ranges
samples_Nco = np.arange(16.,21.1,0.2).astype('float16')
samples_Tk = np.arange(1.,2.4,0.1).astype('float16')
samples_nH2 = np.arange(2.,5.1,0.2).astype('float16')
samples_X12to13 = np.arange(10,205,10).astype('float16')
samples_X13to18 = np.arange(2,21,1.5).astype('float16')
samples_phi = np.arange(0.05, 1.01, 0.05).astype('float16')

size_N = samples_Nco.shape[0]
size_T = samples_Tk.shape[0]
size_n = samples_nH2.shape[0]
size_x12to13 = samples_X12to13.shape[0]
size_x13to18 = samples_X13to18.shape[0]
size_phi = samples_phi.shape[0]

Nco = samples_Nco.reshape(size_N,1,1,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
Nco = Nco.reshape(-1)
# Tk = samples_Tk.reshape(1,size_T,1,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
# Tk = Tk.reshape(-1)
nH2 = samples_nH2.reshape(1,1,size_n,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
nH2 = nH2.reshape(-1)
# X12to13 = samples_X12to13.reshape(1,1,1,size_x12to13,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
# X12to13 = X12to13.reshape(-1)
# X13to18 = samples_X13to18.reshape(1,1,1,1,size_x13to18,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
# X13to18 = X13to18.reshape(-1)
phi = samples_phi.reshape(1,1,1,1,1,size_phi)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
phi = phi.reshape(-1)

# Set up correlation matrix between bands
A_cor = np.zeros((nobs,nobs))
A_cor[0,0] = 0.1**2
A_cor[1,1] = 0.1**2
for i in [2,4]:
    for j in [2,4]:
        A_cor[i,j] = 0.1**2
for i in [3,5]:
    for j in [3,5]:
        A_cor[i,j] = 0.1**2

two_pi_n = (2 * np.pi)**nobs
chi2_map = np.full((75,75), np.nan)
N_co = np.full((75,75), np.nan)
T_k = np.full((75,75), np.nan)
n_h2 = np.full((75,75), np.nan)
X_co213co = np.full((75,75), np.nan)
X_13co2c18o = np.full((75,75), np.nan)
beam_fill = np.full((75,75), np.nan)

print('Import model & data...') 

flux_mod = np.array((np.load(sou_model+'flux_'+model+'_co10.npy').reshape(-1),   
                     np.load(sou_model+'flux_'+model+'_co21.npy').reshape(-1),
                     np.load(sou_model+'flux_'+model+'_13co21.npy').reshape(-1),
                     np.load(sou_model+'flux_'+model+'_13co32.npy').reshape(-1),
                     np.load(sou_model+'flux_'+model+'_c18o21.npy').reshape(-1),
                     np.load(sou_model+'flux_'+model+'_c18o32.npy').reshape(-1)))
shape = flux_mod.shape[1]

flux_co10 = np.load(sou_data+'NGC3351_CO10_mom0.npy')
flux_co21 = np.load(sou_data+'NGC3351_CO21_mom0.npy')
flux_13co21 = np.load(sou_data+'NGC3351_13CO21_mom0.npy')
flux_13co32 = np.load(sou_data+'NGC3351_13CO32_mom0.npy')
flux_c18o21 = np.load(sou_data+'NGC3351_C18O21_mom0.npy')
flux_c18o32 = np.load(sou_data+'NGC3351_C18O32_mom0.npy')

err_co10 = np.load(sou_data+'errors/NGC3351_CO10_emom0_broad_nyq.npy')
err_co21 = np.load(sou_data+'errors/NGC3351_CO21_emom0_broad_nyq.npy')
err_13co21 = np.load(sou_data+'errors/NGC3351_13CO21_emom0_broad_nyq.npy')
err_13co32 = np.load(sou_data+'errors/NGC3351_13CO32_emom0_broad_nyq.npy')
err_c18o21 = np.load(sou_data+'errors/NGC3351_C18O21_emom0_broad_nyq.npy')
err_c18o32 = np.load(sou_data+'errors/NGC3351_C18O32_emom0_broad_nyq.npy')

data_time = time.time()
print('Models & data loaded;', data_time-start_time, 'sec elapsed.')

def chi2_prob(x,y):
    err = np.array((err_co10[y,x], err_co21[y,x], err_13co21[y,x],
          err_13co32[y,x], err_c18o21[y,x], err_c18o32[y,x]))
    flux_obs = np.array((flux_co10[y,x], flux_co21[y,x], flux_13co21[y,x],
               flux_13co32[y,x], flux_c18o21[y,x], flux_c18o32[y,x])).reshape(nobs,1)

    C_bkg = np.zeros((nobs,nobs))
    for i in range(nobs):
        C_bkg[i,i] = err[i]**2

    C_cal = np.matmul(flux_mod.T.reshape(shape, nobs, 1), flux_mod.T.reshape(shape, 1, nobs)) * A_cor
    C_tot = C_cal + C_bkg
    C_inv = np.full([shape, nobs, nobs], np.nan)
    #Q = np.full(shape, np.nan)
    
    for i, mat in enumerate(C_tot):
        try:
            C_inv[i] = np.linalg.inv(mat)
            #Q[i] = np.sqrt(np.linalg.det(mat) * two_pi_n)
        except np.linalg.LinAlgError:
            pass

    diff = (flux_mod - flux_obs).T
    chi2_grid = np.matmul(np.matmul(diff.reshape(shape, 1, nobs), C_inv), diff.reshape(shape, nobs, 1)).reshape(-1)
    #prob_grid = np.nan_to_num(np.exp(-0.5 * chi2_grid) / Q)   
    return chi2_grid#, prob_grid

def bestfit_maps(x,y):   
    los_length = (10**Nco / 15. * map_fwhm[y,x]) / (np.sqrt(phi) * 10**nH2 * x_co)
    mask = los_length < los_max * (3.086 * 10**18) 
    #chi2, _ = chi2_prob(x,y)
    chi2 = np.array(chi2_prob(x,y)) * mask
    chi2[mask==0] = np.nan
    
    if (np.isnan(chi2)).all():
        return x, y, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        chi2_min = np.nanmin(chi2)
        idx_min = np.unravel_index(np.nanargmin(chi2), (size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
        N_value = idx_min[0] * (samples_Nco[1] - samples_Nco[0]) + samples_Nco[0]
        T_value = idx_min[1] * (samples_Tk[1] - samples_Tk[0]) + samples_Tk[0]
        n_value = idx_min[2] * (samples_nH2[1] - samples_nH2[0]) + samples_nH2[0]
        X1_value = idx_min[3] * (samples_X12to13[1] - samples_X12to13[0]) + samples_X12to13[0]
        X2_value = idx_min[4] * (samples_X13to18[1] - samples_X13to18[0]) + samples_X13to18[0]
        phi_value = idx_min[5] * (samples_phi[1] - samples_phi[0]) + samples_phi[0]
        return x, y, chi2_min, N_value, T_value, n_value, X1_value, X2_value, phi_value 

def prob1d_maps(x,y):   
    chi2, prob = chi2_prob(x,y)
    chi2 = np.array(chi2)
    prob = np.array(prob)
    
    if (np.isnan(chi2)).all():
        return x, y, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
    elif output_map == '1dmax':
        N_1d, _ = np.histogram(Nco, bins=size_N, range=(15.9,21.1), weights=prob) 
        T_1d, _ = np.histogram(Tk, bins=size_T, range=(0.95,2.35), weights=prob) 
        n_1d, _ = np.histogram(nH2, bins=size_n, range=(1.9,5.1), weights=prob) 
        X1_1d, _ = np.histogram(X12to13, bins=size_x12to13, range=(5,205), weights=prob) 
        X2_1d, _ = np.histogram(X13to18, bins=size_x13to18, range=(1.25,20.75), weights=prob) 
        phi_1d, _ = np.histogram(phi, bins=size_phi, range=(0.025,1.025), weights=prob)
        N_value = samples_Nco[np.nanargmax(N_1d)]
        T_value = samples_Tk[np.nanargmax(T_1d)]
        n_value = samples_nH2[np.nanargmax(n_1d)]
        X1_value = samples_X12to13[np.nanargmax(X1_1d)]
        X2_value = samples_X13to18[np.nanargmax(X2_1d)]
        phi_value = samples_phi[np.nanargmax(phi_1d)]
        return x, y, N_value, T_value, n_value, X1_value, X2_value, phi_value
        
    elif output_map == 'mean':
        N_value = np.sum(Nco * prob) / prob.sum()
        T_value = np.sum(Tk * prob) / prob.sum()
        n_value = np.sum(nH2 * prob) / prob.sum()
        X1_value = np.sum(X12to13 * prob) / prob.sum()
        X2_value = np.sum(X13to18 * prob) / prob.sum() 
        phi_value = np.sum(phi * prob) / prob.sum()
        return x, y, N_value, T_value, n_value, X1_value, X2_value, phi_value  

    elif output_map == 'median':
        N_value = corner.quantile(Nco, [0.16,0.5,0.84], weights=prob)[1]  #0:neg1sig, 1:median, 2:pos1sig
        T_value = corner.quantile(Tk, [0.16,0.5,0.84], weights=prob)[1]
        n_value = corner.quantile(nH2, [0.16,0.5,0.84], weights=prob)[1]
        X1_value = corner.quantile(X12to13, [0.16,0.5,0.84], weights=prob)[1]
        X2_value = corner.quantile(X13to18, [0.16,0.5,0.84], weights=prob)[1]
        phi_value = corner.quantile(phi, [0.16,0.5,0.84], weights=prob)[1]
        return x, y, N_value, T_value, n_value, X1_value, X2_value, phi_value 

print('Start multi-processing on output maps')
 
if output_map == 'bestfit':
    results = Parallel(n_jobs=5, verbose=10)(delayed(bestfit_maps)(x,y) for x in range(30,45) for y in range(25,50))           
    for result in results:
        x, y, chi2_min, N_value, T_value, n_value, X1_value, X2_value, phi_value = result
        chi2_map[y,x] = chi2_min
        N_co[y,x] = N_value
        T_k[y,x] = T_value
        n_h2[y,x] = n_value
        X_co213co[y,x] = X1_value
        X_13co2c18o[y,x] = X2_value
        beam_fill[y,x] = phi_value
        
    np.save(sou_model+'chi2_'+model+'_cov_los50.npy', chi2_map)
    np.save(sou_model+'Nco_'+model+'_cov_los50_'+output_map+'.npy', N_co)
    np.save(sou_model+'Tk_'+model+'_cov_los50_'+output_map+'.npy', T_k)
    np.save(sou_model+'nH2_'+model+'_cov_los50_'+output_map+'.npy', n_h2)
    np.save(sou_model+'X12to13_'+model+'_cov_los50_'+output_map+'.npy', X_co213co)
    np.save(sou_model+'X13to18_'+model+'_cov_los50_'+output_map+'.npy', X_13co2c18o)
    np.save(sou_model+'phi_'+model+'_cov_los50_'+output_map+'.npy', beam_fill)
    
else:
    results = Parallel(n_jobs=1, verbose=10)(delayed(prob1d_maps)(x,y) for x in range(20,55) for y in range(20,55))           
    for result in results:
        x, y, N_value, T_value, n_value, X1_value, X2_value, phi_value = result
        N_co[y,x] = N_value
        T_k[y,x] = T_value
        n_h2[y,x] = n_value
        X_co213co[y,x] = X1_value
        X_13co2c18o[y,x] = X2_value
        beam_fill[y,x] = phi_value        
    
    np.save(sou_model+'Nco_'+model+'_cov_'+output_map+'.npy', N_co)
    np.save(sou_model+'Tk_'+model+'_cov_'+output_map+'.npy', T_k)
    np.save(sou_model+'nH2_'+model+'_cov_'+output_map+'.npy', n_h2)
    np.save(sou_model+'X12to13_'+model+'_cov_'+output_map+'.npy', X_co213co)
    np.save(sou_model+'X13to18_'+model+'_cov_'+output_map+'.npy', X_13co2c18o)
    np.save(sou_model+'phi_'+model+'_'+output_map+'.npy', beam_fill)
    
print('Output maps constructed.')

end_time = time.time()
print('Elapsed time per map: %s sec' % ((end_time - start_time)))
