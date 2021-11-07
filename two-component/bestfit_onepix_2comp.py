import numpy as np
import time

'''
Assuming zero covariances between data, this script returns the minimum chi^2 
value and the corresponding best-fit parameter set from the two-component model.
The chi^2 values of all parameter sets are saved as a 1D numpy array.  
'''

start_time = time.time()
print('Loading models and data...')

idx_x = 34
idx_y = 43
extent_nT = (2.,5.5,1,2.1)
num_Nco = 13
num_Tk = 11
num_nH2 = 13
num_phi = 13

sou_model = 'radex_model/'
sou_data = 'data_image/'
model = '4d_2comp'

model_co10 = np.load(sou_model+'fluxsum_'+model+'_co10_8d.npy')   
model_co21 = np.load(sou_model+'fluxsum_'+model+'_co21_8d.npy')
model_13co21 = np.load(sou_model+'fluxsum_'+model+'_13co21_8d.npy')
model_13co32 = np.load(sou_model+'fluxsum_'+model+'_13co32_8d.npy')
model_c18o21 = np.load(sou_model+'fluxsum_'+model+'_c18o21_8d.npy')
model_c18o32 = np.load(sou_model+'fluxsum_'+model+'_c18o32_8d.npy')

flux_co10 = np.load(sou_data+'NGC3351_CO10_mom0.npy')[idx_y,idx_x]
flux_co21 = np.load(sou_data+'NGC3351_CO21_mom0.npy')[idx_y,idx_x]
flux_13co21 = np.load(sou_data+'NGC3351_13CO21_mom0.npy')[idx_y,idx_x]
flux_13co32 = np.load(sou_data+'NGC3351_13CO32_mom0.npy')[idx_y,idx_x]
flux_c18o21 = np.load(sou_data+'NGC3351_C18O21_mom0.npy')[idx_y,idx_x]
flux_c18o32 = np.load(sou_data+'NGC3351_C18O32_mom0.npy')[idx_y,idx_x]

noise_co10 = np.load(sou_data+'errors/NGC3351_CO10_emom0_broad_nyq.npy')[idx_y,idx_x]
noise_co21 = np.load(sou_data+'errors/NGC3351_CO21_emom0_broad_nyq.npy')[idx_y,idx_x]
noise_13co21 = np.load(sou_data+'errors/NGC3351_13CO21_emom0_broad_nyq.npy')[idx_y,idx_x]
noise_13co32 = np.load(sou_data+'errors/NGC3351_13CO32_emom0_broad_nyq.npy')[idx_y,idx_x]
noise_c18o21 = np.load(sou_data+'errors/NGC3351_C18O21_emom0_broad_nyq.npy')[idx_y,idx_x]
noise_c18o32 = np.load(sou_data+'errors/NGC3351_C18O32_emom0_broad_nyq.npy')[idx_y,idx_x]

err_co10 = np.sqrt(noise_co10**2 + (0.1 * flux_co10)**2)
err_co21 = np.sqrt(noise_co21**2 + (0.1 * flux_co21)**2)
err_13co21 = np.sqrt(noise_13co21**2 + (0.1 * flux_13co21)**2)
err_13co32 = np.sqrt(noise_13co32**2 + (0.1 * flux_13co32)**2)
err_c18o21 = np.sqrt(noise_c18o21**2 + (0.1 * flux_c18o21)**2)
err_c18o32 = np.sqrt(noise_c18o32**2 + (0.1 * flux_c18o32)**2)

print('CO (1-0):',flux_co10,'+/-',err_co10)
print('CO (2-1):',flux_co21,'+/-',err_co21)
print('13CO (2-1):',flux_13co21,'+/-',err_13co21)
print('13CO (3-2):',flux_13co32,'+/-',err_13co32)
print('C18O (2-1):',flux_c18o21,'+/-',err_c18o21)
print('C18O (3-2):',flux_c18o32,'+/-',err_c18o32)

loaded_time = time.time()
print(loaded_time - start_time, 'sec elapsed.')
print('Computing chi^2 and best-fit parameter set')

# Compute minimum chi^2 and its correspoding parameter set
chi_sum = ((model_co10 - flux_co10) / err_co10)**2
chi_sum = chi_sum + ((model_co21 - flux_co21) / err_co21)**2
chi_sum = chi_sum + ((model_13co21 - flux_13co21) / err_13co21)**2
chi_sum = chi_sum + ((model_13co32 - flux_13co32) / err_13co32)**2
chi_sum = chi_sum + ((model_c18o21 - flux_c18o21) / err_c18o21)**2
chi_sum = chi_sum + ((model_c18o32 - flux_c18o32) / err_c18o32)**2

idx_min = np.unravel_index(np.nanargmin(chi_sum, axis=None), chi_sum.shape)
#par_min_comp0 = np.unravel_index(idx_min[0], (num_Nco, num_Tk, num_nH2, num_phi)) 
#par_min_comp1 = np.unravel_index(idx_min[1], (num_Nco, num_Tk, num_nH2, num_phi)) 
N0 = np.round(0.25*idx_min[0] + 16., 2)
T0 = np.round(0.1*idx_min[1] + 1., 1)
n0 = np.round(0.25*idx_min[2] + 2., 2)
phi0 = np.round(10**(0.1*idx_min[3] - 1.3), 2)
N1 = np.round(0.25*idx_min[4] + 16., 2)
T1 = np.round(0.1*idx_min[5] + 1., 1)
n1 = np.round(0.25*idx_min[6] + 2., 2)
phi1 = np.round(10**(0.1*idx_min[7] - 1.3), 2)

end_time = time.time()
print(end_time - loaded_time, 'sec elapsed.')

print('Minumum chi^2 =', np.nanmin(chi_sum), 'at', idx_min)
print('(N0, T0, n0, phi0) =', N0, T0, n0, phi0)
print('(N1, T1, n1, phi1) =', N1, T1, n1, phi1)
np.save(sou_model+'chi2_'+model+'_'+str(idx_x)+'_'+str(idx_y)+'_8d.npy', chi_sum)
