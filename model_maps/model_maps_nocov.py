import time
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import corner

'''
Assuming zero covariances between data, this script computes the bestfit/ 
1dmax/mean/median parameter values for each pixel in the galaxy image. 
Maps are saved as numpy arrays. The output_map input should be set manually.
If output_map = 'bestfit', an additional map showing the minimum chi^2 value 
at each pixel will be saved.   

'''

start_time = time.time()

model = '6d_coarse'
output_map = 'bestfit'
sou_model = 'radex_model/'
sou_data = 'data_image/'

chi2_map = np.full((75,75), np.nan)
N_co = np.full((75,75), np.nan)
T_k = np.full((75,75), np.nan)
n_h2 = np.full((75,75), np.nan)
X_co213co = np.full((75,75), np.nan)
X_13co2c18o = np.full((75,75), np.nan)
beam_fill = np.full((75,75), np.nan)

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

Nco = samples_Nco.reshape(size_N,1,1,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi),dtype='float16')
Nco = Nco.reshape(-1)
Tk = samples_Tk.reshape(1,size_T,1,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi),dtype='float16')
Tk = Tk.reshape(-1)
nH2 = samples_nH2.reshape(1,1,size_n,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi),dtype='float16')
nH2 = nH2.reshape(-1)
X12to13 = samples_X12to13.reshape(1,1,1,size_x12to13,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi),dtype='float16')
X12to13 = X12to13.reshape(-1)
X13to18 = samples_X13to18.reshape(1,1,1,1,size_x13to18,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi),dtype='float16')
X13to18 = X13to18.reshape(-1)
phi = samples_phi.reshape(1,1,1,1,1,size_phi)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi),dtype='float16')
phi = phi.reshape(-1)

model_co10 = np.load(sou_model+'flux_'+model+'_co10.npy').astype('float16')     
model_co21 = np.load(sou_model+'flux_'+model+'_co21.npy').astype('float16')
model_13co21 = np.load(sou_model+'flux_'+model+'_13co21.npy').astype('float16')
model_13co32 = np.load(sou_model+'flux_'+model+'_13co32.npy').astype('float16')
model_c18o21 = np.load(sou_model+'flux_'+model+'_c18o21.npy').astype('float16')
model_c18o32 = np.load(sou_model+'flux_'+model+'_c18o32.npy').astype('float16') 

flux_co10 = np.load(sou_data+'NGC3351_CO10_mom0.npy')
flux_co21 = np.load(sou_data+'NGC3351_CO21_mom0.npy')
flux_13co21 = np.load(sou_data+'NGC3351_13CO21_mom0.npy')
flux_13co32 = np.load(sou_data+'NGC3351_13CO32_mom0.npy')
flux_c18o21 = np.load(sou_data+'NGC3351_C18O21_mom0.npy')
flux_c18o32 = np.load(sou_data+'NGC3351_C18O32_mom0.npy')

# err_co10 = np.load(sou_data+'errors/NGC3351_CO10_emom0_broad_nyq.npy')
# err_co21 = np.load(sou_data+'errors/NGC3351_CO21_emom0_broad_nyq.npy')
# err_13co21 = np.load(sou_data+'errors/NGC3351_13CO21_emom0_broad_nyq.npy')
# err_13co32 = np.load(sou_data+'errors/NGC3351_13CO32_emom0_broad_nyq.npy')
# err_c18o21 = np.load(sou_data+'errors/NGC3351_C18O21_emom0_broad_nyq.npy')
# err_c18o32 = np.load(sou_data+'errors/NGC3351_C18O32_emom0_broad_nyq.npy')

err_co10 = 0.1 * flux_co10
err_co21 = 0.1 * flux_co21
err_13co21 = 0.1 * flux_13co21
err_13co32 = 0.1 * flux_13co32
err_c18o21 = 0.1 * flux_c18o21
err_c18o32 = 0.1 * flux_c18o32

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
    chi2, _ = chi2_prob(x,y)
    chi2 = np.array(chi2)
    
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
        
    else:
        N_1d, _ = np.histogram(Nco, bins=size_N, range=(15.9,21.1), weights=prob) 
        T_1d, _ = np.histogram(Tk, bins=size_T, range=(0.95,2.35), weights=prob) 
        n_1d, _ = np.histogram(nH2, bins=size_n, range=(1.9,5.1), weights=prob) 
        X1_1d, _ = np.histogram(X12to13, bins=size_x12to13, range=(5,205), weights=prob) 
        X2_1d, _ = np.histogram(X13to18, bins=size_x13to18, range=(1.25,20.75), weights=prob) 
        phi_1d, _ = np.histogram(phi, bins=size_phi, range=(0.025,1.025), weights=prob)
        
        if output_map == '1dmax':      
            N_value = samples_Nco[np.nanargmax(N_1d)]
            T_value = samples_Tk[np.nanargmax(T_1d)]
            n_value = samples_nH2[np.nanargmax(n_1d)]
            X1_value = samples_X12to13[np.nanargmax(X1_1d)]
            X2_value = samples_X13to18[np.nanargmax(X2_1d)]
            phi_value = samples_phi[np.nanargmax(phi_1d)]
            return x, y, N_value, T_value, n_value, X1_value, X2_value, phi_value
            
        elif output_map == 'peaks':
            np.save('radex_model/6d_prob1d/Nco_'+model+'_prob1d_'+str(x)+'_'+str(y)+'.npy', N_1d)
            np.save('radex_model/6d_prob1d/Tk_'+model+'_prob1d_'+str(x)+'_'+str(y)+'.npy', T_1d)
            np.save('radex_model/6d_prob1d/nH2_'+model+'_prob1d_'+str(x)+'_'+str(y)+'.npy', n_1d)
            np.save('radex_model/6d_prob1d/X12to13_'+model+'_prob1d_'+str(x)+'_'+str(y)+'.npy', X1_1d)
            np.save('radex_model/6d_prob1d/X13to18_'+model+'_prob1d_'+str(x)+'_'+str(y)+'.npy', X2_1d)
            np.save('radex_model/6d_prob1d/phi_'+model+'_prob1d_'+str(x)+'_'+str(y)+'.npy', phi_1d)
            
            N_value = count_peaks(N_1d, 0.2*np.nanmax(N_1d))
            T_value = count_peaks(T_1d, 0.2*np.nanmax(T_1d))
            n_value = count_peaks(n_1d, 0.2*np.nanmax(n_1d))
            X1_value = count_peaks(X1_1d, 0.2*np.nanmax(X1_1d))
            X2_value = count_peaks(X2_1d, 0.2*np.nanmax(X2_1d))
            phi_value = count_peaks(phi_1d, 0.2*np.nanmax(phi_1d))
            return x, y, N_value, T_value, n_value, X1_value, X2_value, phi_value 

def count_peaks(list, threshold):
    count = 0
    for idx, item in enumerate(list[:-1]):
        if list[idx] < threshold and list[idx+1] > threshold:
            count += 1
        if list[idx] > threshold and list[idx+1] < threshold:
            count += 1
    return count/2

print('Start multi-processing on output maps')
 
if output_map == 'bestfit':
    results = Parallel(n_jobs=20, verbose=5)(delayed(bestfit_maps)(x,y) for x in range(20,55) for y in range(20,55))           
    for result in results:
        x, y, chi2_min, N_value, T_value, n_value, X1_value, X2_value, phi_value = result
        chi2_map[y,x] = chi2_min
        N_co[y,x] = N_value
        T_k[y,x] = T_value
        n_h2[y,x] = n_value
        X_co213co[y,x] = X1_value
        X_13co2c18o[y,x] = X2_value
        beam_fill[y,x] = phi_value
        
    np.save(sou_model+'chi2_'+model+'.npy', chi2_map)
    np.save(sou_model+'Nco_'+model+'_'+output_map+'.npy', N_co)
    np.save(sou_model+'Tk_'+model+'_'+output_map+'.npy', T_k)
    np.save(sou_model+'nH2_'+model+'_'+output_map+'.npy', n_h2)
    np.save(sou_model+'X12to13_'+model+'_'+output_map+'.npy', X_co213co)
    np.save(sou_model+'X13to18_'+model+'_'+output_map+'.npy', X_13co2c18o)
    np.save(sou_model+'phi_'+model+'_'+output_map+'.npy', beam_fill)
    
else:
    results = Parallel(n_jobs=20, verbose=5)(delayed(prob1d_maps)(x,y) for x in range(20,55) for y in range(20,55))           
    for result in results:
        x, y, N_value, T_value, n_value, X1_value, X2_value, phi_value = result
        N_co[y,x] = N_value
        T_k[y,x] = T_value
        n_h2[y,x] = n_value
        X_co213co[y,x] = X1_value
        X_13co2c18o[y,x] = X2_value
        beam_fill[y,x] = phi_value        
    
    np.save(sou_model+'Nco_'+model+'_'+output_map+'.npy', N_co)
    np.save(sou_model+'Tk_'+model+'_'+output_map+'.npy', T_k)
    np.save(sou_model+'nH2_'+model+'_'+output_map+'.npy', n_h2)
    np.save(sou_model+'X12to13_'+model+'_'+output_map+'.npy', X_co213co)
    np.save(sou_model+'X13to18_'+model+'_'+output_map+'.npy', X_13co2c18o)
    np.save(sou_model+'phi_'+model+'_'+output_map+'.npy', beam_fill)
    
print('Output maps constructed.')

end_time = time.time()
print('Total elapsed time: %s sec' % ((end_time - start_time)))
