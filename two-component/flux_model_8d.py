import numpy as np
from joblib import Parallel, delayed
import time

start_time = time.time()
print('Loading 2D models...')

# Input Settings
sou_model = 'radex_model/'
model = '4d_2comp'
size_N = 13
size_T = 11
size_n = 13
size_phi = 13
size_4d = size_N*size_T*size_n*size_phi

flux_2comp_co10 = np.load(sou_model+'fluxsum_'+model+'_co10_2d.npy')
flux_2comp_co21 = np.load(sou_model+'fluxsum_'+model+'_co21_2d.npy')
flux_2comp_13co21 = np.load(sou_model+'fluxsum_'+model+'_13co21_2d.npy')
flux_2comp_13co32 = np.load(sou_model+'fluxsum_'+model+'_13co32_2d.npy')
flux_2comp_c18o21 = np.load(sou_model+'fluxsum_'+model+'_c18o21_2d.npy')
flux_2comp_c18o32 = np.load(sou_model+'fluxsum_'+model+'_c18o32_2d.npy')    

convert_time = time.time()
print(convert_time - start_time, 'sec elapsed. Constructing 8D models...')

# Convert 2D fluxsum models to 8D models
fluxsum_8d_co10 = np.full((size_N,size_T,size_n,size_phi,size_N,size_T,size_n,size_phi), np.nan, dtype='float32')
fluxsum_8d_co21 = np.full((size_N,size_T,size_n,size_phi,size_N,size_T,size_n,size_phi), np.nan, dtype='float32')
fluxsum_8d_13co21 = np.full((size_N,size_T,size_n,size_phi,size_N,size_T,size_n,size_phi), np.nan, dtype='float32')
fluxsum_8d_13co32 = np.full((size_N,size_T,size_n,size_phi,size_N,size_T,size_n,size_phi), np.nan, dtype='float32')
fluxsum_8d_c18o21 = np.full((size_N,size_T,size_n,size_phi,size_N,size_T,size_n,size_phi), np.nan, dtype='float32')
fluxsum_8d_c18o32 = np.full((size_N,size_T,size_n,size_phi,size_N,size_T,size_n,size_phi), np.nan, dtype='float32')

for i in range(size_4d):
    idx_comp0 = np.unravel_index(i, (size_N,size_T,size_n,size_phi)) 
    for j in range(i+1):
        idx_comp1 = np.unravel_index(j, (size_N,size_T,size_n,size_phi)) 
        # constraining T1 >= T2
        if idx_comp0[1] < idx_comp1[1]:  
            fluxsum_8d_co10[idx_comp1+idx_comp0] = flux_2comp_co10[j,i]
            fluxsum_8d_co21[idx_comp1+idx_comp0] = flux_2comp_co21[j,i]
            fluxsum_8d_13co21[idx_comp1+idx_comp0] = flux_2comp_13co21[j,i]
            fluxsum_8d_13co32[idx_comp1+idx_comp0] = flux_2comp_13co32[j,i]
            fluxsum_8d_c18o21[idx_comp1+idx_comp0] = flux_2comp_c18o21[j,i]
            fluxsum_8d_c18o32[idx_comp1+idx_comp0] = flux_2comp_c18o32[j,i]
        else:
            fluxsum_8d_co10[idx_comp0+idx_comp1] = flux_2comp_co10[j,i]
            fluxsum_8d_co21[idx_comp0+idx_comp1] = flux_2comp_co21[j,i]
            fluxsum_8d_13co21[idx_comp0+idx_comp1] = flux_2comp_13co21[j,i]
            fluxsum_8d_13co32[idx_comp0+idx_comp1] = flux_2comp_13co32[j,i]
            fluxsum_8d_c18o21[idx_comp0+idx_comp1] = flux_2comp_c18o21[j,i]
            fluxsum_8d_c18o32[idx_comp0+idx_comp1] = flux_2comp_c18o32[j,i]

np.save(sou_model+'fluxsum_'+model+'_co10_8d.npy', fluxsum_8d_co10)
np.save(sou_model+'fluxsum_'+model+'_co21_8d.npy', fluxsum_8d_co21)
np.save(sou_model+'fluxsum_'+model+'_13co21_8d.npy', fluxsum_8d_13co21)
np.save(sou_model+'fluxsum_'+model+'_13co32_8d.npy', fluxsum_8d_13co32)
np.save(sou_model+'fluxsum_'+model+'_c18o21_8d.npy', fluxsum_8d_c18o21)
np.save(sou_model+'fluxsum_'+model+'_c18o32_8d.npy', fluxsum_8d_c18o32)

end_time = time.time()
print(end_time - convert_time, 'sec elapsed. 8D models saved.')


