import time
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import corner
from astropy.io import fits

'''
Assuming zero covariances between data, this script computes the bestfit/ 
1dmax/mean/median parameter values for each pixel in the galaxy image. 
Maps are saved as numpy arrays. The output_map input should be set manually.
If output_map = 'bestfit', an additional map showing the minimum chi^2 value 
at each pixel will be saved.   

'''

start_time = time.time()

model = '4d_2comp'
output_map = '1dmax'
sou_model = 'radex_model/'
sou_data = 'data_image/'

chi2_map = np.full((75,75), np.nan)
N_co = np.full((2,75,75), np.nan)
T_k = np.full((2,75,75), np.nan)
n_h2 = np.full((2,75,75), np.nan)
beam_fill = np.full((2,75,75), np.nan)

# Set parameter ranges
samples_Nco = np.arange(16.,19.1,0.25).astype('float32')
samples_Tk = np.arange(1.,2.1,0.1).astype('float32')
samples_nH2 = np.arange(2.,5.1,0.25).astype('float32')
samples_phi = np.arange(-1.3, -0.05, 0.1).astype('float32')

# Set up constraints if using priors (optional)
los_max = 100.
x_co = 3 * 10**(-4)
map_ew = fits.open('data_image/NGC3351_CO10_ew_broad_nyq.fits')[0].data
map_fwhm = map_ew * 2.35  # Conversion of 1 sigma to FWHM assuming Gaussian

size_N = samples_Nco.shape[0]
size_T = samples_Tk.shape[0]
size_n = samples_nH2.shape[0]
size_phi = samples_phi.shape[0]

Nco_0 = samples_Nco.reshape(size_N,1,1,1,1,1,1,1)*np.ones((size_N,size_T,size_n,size_phi,size_N,size_T,size_n,size_phi),dtype='float32')
Tk_0 = samples_Tk.reshape(1,size_T,1,1,1,1,1,1)*np.ones((size_N,size_T,size_n,size_phi,size_N,size_T,size_n,size_phi),dtype='float32')
nH2_0 = samples_nH2.reshape(1,1,size_n,1,1,1,1,1)*np.ones((size_N,size_T,size_n,size_phi,size_N,size_T,size_n,size_phi),dtype='float32')
phi_0 = samples_phi.reshape(1,1,1,size_phi,1,1,1,1)*np.ones((size_N,size_T,size_n,size_phi,size_N,size_T,size_n,size_phi),dtype='float32')
Nco_1 = samples_Nco.reshape(1,1,1,1,size_N,1,1,1)*np.ones((size_N,size_T,size_n,size_phi,size_N,size_T,size_n,size_phi),dtype='float32')
Tk_1 = samples_Tk.reshape(1,1,1,1,1,size_T,1,1)*np.ones((size_N,size_T,size_n,size_phi,size_N,size_T,size_n,size_phi),dtype='float32')
nH2_1 = samples_nH2.reshape(1,1,1,1,1,1,size_n,1)*np.ones((size_N,size_T,size_n,size_phi,size_N,size_T,size_n,size_phi),dtype='float32')
phi_1 = samples_phi.reshape(1,1,1,1,1,1,1,size_phi)*np.ones((size_N,size_T,size_n,size_phi,size_N,size_T,size_n,size_phi),dtype='float32')

Nco_0 = Nco_0.reshape(-1)
Tk_0 = Tk_0.reshape(-1)
nH2_0 = nH2_0.reshape(-1)
phi_0 = phi_0.reshape(-1)
Nco_1 = Nco_1.reshape(-1)
Tk_1 = Tk_1.reshape(-1)
nH2_1 = nH2_1.reshape(-1)
phi_1 = phi_1.reshape(-1)

model_co10 = np.load(sou_model+'fluxsum_'+model+'_co10_8d.npy')
model_co21 = np.load(sou_model+'fluxsum_'+model+'_co21_8d.npy')
model_13co21 = np.load(sou_model+'fluxsum_'+model+'_13co21_8d.npy')
model_13co32 = np.load(sou_model+'fluxsum_'+model+'_13co32_8d.npy')
model_c18o21 = np.load(sou_model+'fluxsum_'+model+'_c18o21_8d.npy')
model_c18o32 = np.load(sou_model+'fluxsum_'+model+'_c18o32_8d.npy')

flux_co10 = fits.open(sou_data+'NGC3351_CO10_mom0_broad_nyq.fits')[0].data.astype('float32')
flux_co21 = fits.open(sou_data+'NGC3351_CO21_mom0_broad_nyq.fits')[0].data.astype('float32')
flux_13co21 = fits.open(sou_data+'NGC3351_13CO21_mom0_broad_nyq.fits')[0].data.astype('float32')
flux_13co32 = fits.open(sou_data+'NGC3351_13CO32_mom0_broad_nyq.fits')[0].data.astype('float32')
flux_c18o21 = fits.open(sou_data+'NGC3351_C18O21_mom0_broad_nyq.fits')[0].data.astype('float32')
flux_c18o32 = fits.open(sou_data+'NGC3351_C18O32_mom0_broad_nyq.fits')[0].data.astype('float32')

noise_co10 = fits.open(sou_data+'errors/NGC3351_CO10_emom0_broad_nyq.fits')[0].data
noise_co21 = fits.open(sou_data+'errors/NGC3351_CO21_emom0_broad_nyq.fits')[0].data
noise_13co21 = fits.open(sou_data+'errors/NGC3351_13CO21_emom0_broad_nyq.fits')[0].data
noise_13co32 = fits.open(sou_data+'errors/NGC3351_13CO32_emom0_broad_nyq.fits')[0].data
noise_c18o21 = fits.open(sou_data+'errors/NGC3351_C18O21_emom0_broad_nyq.fits')[0].data
noise_c18o32 = fits.open(sou_data+'errors/NGC3351_C18O32_emom0_broad_nyq.fits')[0].data

err_co10 = np.sqrt(noise_co10**2 + (0.1 * flux_co10)**2).astype('float32')
err_co21 = np.sqrt(noise_co21**2 + (0.1 * flux_co21)**2).astype('float32')
err_13co21 = np.sqrt(noise_13co21**2 + (0.1 * flux_13co21)**2).astype('float32')
err_13co32 = np.sqrt(noise_13co32**2 + (0.1 * flux_13co32)**2).astype('float32')
err_c18o21 = np.sqrt(noise_c18o21**2 + (0.1 * flux_c18o21)**2).astype('float32')
err_c18o32 = np.sqrt(noise_c18o32**2 + (0.1 * flux_c18o32)**2).astype('float32')

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
    return chi_sum, prob

def bestfit_maps(x,y):   
    #los_length = (10**Nco / 15. * map_fwhm[y,x]) / (np.sqrt(10**phi) * 10**nH2 * x_co)
    #mask = los_length < los_max * (3.086 * 10**18)
    chi2, _ = chi2_prob(x,y)
    chi2 = np.array(chi2) #* mask
    #chi2[mask==0] = np.nan
    
    if (np.isnan(chi2)).all():
        return x, y, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        chi2_min = np.nanmin(chi2)
        idx_min = np.unravel_index(np.nanargmin(chi2, axis=None), chi2.shape)
        #idx_min_comp0 = np.unravel_index(idx_min[0], (size_N, size_T, size_n, size_phi)) 
        #idx_min_comp1 = np.unravel_index(idx_min[1], (size_N, size_T, size_n, size_phi))
        
        N0_value = idx_min[0] * (samples_Nco[1] - samples_Nco[0]) + samples_Nco[0]
        T0_value = idx_min[1] * (samples_Tk[1] - samples_Tk[0]) + samples_Tk[0]
        n0_value = idx_min[2] * (samples_nH2[1] - samples_nH2[0]) + samples_nH2[0]
        phi0_value = idx_min[3] * (samples_phi[1] - samples_phi[0]) + samples_phi[0]
        N1_value = idx_min[4] * (samples_Nco[1] - samples_Nco[0]) + samples_Nco[0]
        T1_value = idx_min[5] * (samples_Tk[1] - samples_Tk[0]) + samples_Tk[0]
        n1_value = idx_min[6] * (samples_nH2[1] - samples_nH2[0]) + samples_nH2[0]
        phi1_value = idx_min[7] * (samples_phi[1] - samples_phi[0]) + samples_phi[0]
        
        return x, y, chi2_min, N0_value, T0_value, n0_value, phi0_value, N1_value, T1_value, n1_value, phi1_value

def prob1d_maps(x,y): 
    #los_length = (10**Nco / 15. * map_fwhm[y,x]) / (np.sqrt(phi) * 10**nH2 * x_co)
    #mask = los_length < los_max * (3.086 * 10**18)  
    chi2, prob = chi2_prob(x,y)
    chi2 = np.array(chi2) #* mask
    #chi2[mask==0] = np.nan
    prob = np.array(prob) #* mask
    
    if (np.isnan(chi2)).all():
        return x, y, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
    elif output_map == 'mean':
        N0_value = np.sum(Nco_0 * prob) / prob.sum()
        T0_value = np.sum(Tk_0 * prob) / prob.sum()
        n0_value = np.sum(nH2_0 * prob) / prob.sum()
        phi0_value = np.sum(phi_0 * prob) / prob.sum()
        N1_value = np.sum(Nco_1 * prob) / prob.sum()
        T1_value = np.sum(Tk_1 * prob) / prob.sum()
        n1_value = np.sum(nH2_1 * prob) / prob.sum()
        phi1_value = np.sum(phi_1 * prob) / prob.sum()
        return x, y, N0_value, T0_value, n0_value, phi0_value, N1_value, T1_value, n1_value, phi1_value  
       
    else:
        N0_1d, _ = np.histogram(Nco_0, bins=size_N, range=(15.75,19.25), weights=prob) 
        T0_1d, _ = np.histogram(Tk_0, bins=size_T, range=(0.9,2.1), weights=prob) 
        n0_1d, _ = np.histogram(nH2_0, bins=size_n, range=(1.75,5.25), weights=prob) 
        phi0_1d, _ = np.histogram(phi_0, bins=size_phi, range=(-1.2,0.), weights=prob)
        N1_1d, _ = np.histogram(Nco_1, bins=size_N, range=(15.75,19.25), weights=prob) 
        T1_1d, _ = np.histogram(Tk_1, bins=size_T, range=(0.9,2.1), weights=prob) 
        n1_1d, _ = np.histogram(nH2_1, bins=size_n, range=(1.75,5.25), weights=prob) 
        phi1_1d, _ = np.histogram(phi_1, bins=size_phi, range=(-1.2,0.), weights=prob)
        
        if output_map == '1dmax':      
            N0_value = samples_Nco[np.nanargmax(N0_1d)]
            T0_value = samples_Tk[np.nanargmax(T0_1d)]
            n0_value = samples_nH2[np.nanargmax(n0_1d)]
            phi0_value = samples_phi[np.nanargmax(phi0_1d)]
            N1_value = samples_Nco[np.nanargmax(N1_1d)]
            T1_value = samples_Tk[np.nanargmax(T1_1d)]
            n1_value = samples_nH2[np.nanargmax(n1_1d)]
            phi1_value = samples_phi[np.nanargmax(phi1_1d)]
            return x, y, N0_value, T0_value, n0_value, phi0_value, N1_value, T1_value, n1_value, phi1_value
        
        elif output_map == 'median':
            quantile_values = np.array((0.16,0.5,0.84),dtype='float32')
            value = 2  #0:neg1sig, 1:median, 2:pos1sig
            
            N0_cdf = np.cumsum(N0_1d)
            N0_cdf /= N0_cdf[-1]
            N0_value = np.interp(quantile_values[value], N0_cdf, samples_Nco)  
            T0_cdf = np.cumsum(T0_1d)
            T0_cdf /= T0_cdf[-1]
            T0_value = np.interp(quantile_values[value], T0_cdf, samples_Tk)
            n0_cdf = np.cumsum(n0_1d)
            n0_cdf /= n0_cdf[-1]
            n0_value = np.interp(quantile_values[value], n0_cdf, samples_nH2)
            phi0_cdf = np.cumsum(phi0_1d)
            phi0_cdf /= phi0_cdf[-1]
            phi0_value = np.interp(quantile_values[value], phi0_cdf, samples_phi)
            
            N1_cdf = np.cumsum(N1_1d)
            N1_cdf /= N1_cdf[-1]
            N1_value = np.interp(quantile_values[value], N1_cdf, samples_Nco)  
            T1_cdf = np.cumsum(T1_1d)
            T1_cdf /= T1_cdf[-1]
            T1_value = np.interp(quantile_values[value], T1_cdf, samples_Tk)
            n1_cdf = np.cumsum(n1_1d)
            n1_cdf /= n1_cdf[-1]
            n1_value = np.interp(quantile_values[value], n1_cdf, samples_nH2)
            phi1_cdf = np.cumsum(phi1_1d)
            phi1_cdf /= phi1_cdf[-1]
            phi1_value = np.interp(quantile_values[value], phi1_cdf, samples_phi)
            
            return x, y, N0_value, T0_value, n0_value, phi0_value, N1_value, T1_value, n1_value, phi1_value

print('Start multi-processing on output maps')
 
if output_map == 'bestfit':
    results = Parallel(n_jobs=10, verbose=10)(delayed(bestfit_maps)(x,y) for x in range(20,55) for y in range(20,55))           
    for result in results:
        x, y, chi2_min, N0_value, T0_value, n0_value, phi0_value, N1_value, T1_value, n1_value, phi1_value = result
        chi2_map[y,x] = chi2_min
        N_co[0,y,x] = N0_value
        T_k[0,y,x] = T0_value
        n_h2[0,y,x] = n0_value
        beam_fill[0,y,x] = phi0_value
        N_co[1,y,x] = N1_value
        T_k[1,y,x] = T1_value
        n_h2[1,y,x] = n1_value
        beam_fill[1,y,x] = phi1_value
        
    np.save(sou_model+'chi2_'+model+'.npy', chi2_map)
    np.save(sou_model+'Nco_'+model+'_'+output_map+'.npy', N_co)
    np.save(sou_model+'Tk_'+model+'_'+output_map+'.npy', T_k)
    np.save(sou_model+'nH2_'+model+'_'+output_map+'.npy', n_h2)
    np.save(sou_model+'phi_'+model+'_'+output_map+'.npy', beam_fill)
    
else:
    results = Parallel(n_jobs=5, verbose=10)(delayed(prob1d_maps)(x,y) for x in range(20,55) for y in range(20,55))           
    for result in results:
        x, y, N0_value, T0_value, n0_value, phi0_value, N1_value, T1_value, n1_value, phi1_value = result
        N_co[0,y,x] = N0_value
        T_k[0,y,x] = T0_value
        n_h2[0,y,x] = n0_value
        beam_fill[0,y,x] = phi0_value
        N_co[1,y,x] = N1_value
        T_k[1,y,x] = T1_value
        n_h2[1,y,x] = n1_value
        beam_fill[1,y,x] = phi1_value        
    
    np.save(sou_model+'Nco_'+model+'_'+output_map+'.npy', N_co)
    np.save(sou_model+'Tk_'+model+'_'+output_map+'.npy', T_k)
    np.save(sou_model+'nH2_'+model+'_'+output_map+'.npy', n_h2)
    np.save(sou_model+'phi_'+model+'_'+output_map+'.npy', beam_fill)
    
print('Output maps constructed.')

end_time = time.time()
print('Total elapsed time: %s sec' % ((end_time - start_time)))


