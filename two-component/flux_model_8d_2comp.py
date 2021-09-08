import numpy as np
from joblib import Parallel, delayed
import time

start_time = time.time()
print('Loading models...')

# Input Settings
sou_model = 'radex_model/'
model = '4d_2comp'
size_N = 13
size_T = 11
size_n = 13
size_phi = 13
size_4d = size_N*size_T*size_n*size_phi

model_4d = np.array(( np.load(sou_model+'flux_'+model+'_co10.npy'), np.load(sou_model+'flux_'+model+'_co21.npy'),
                    np.load(sou_model+'flux_'+model+'_13co21.npy'), np.load(sou_model+'flux_'+model+'_13co32.npy'),
                    np.load(sou_model+'flux_'+model+'_c18o21.npy'), np.load(sou_model+'flux_'+model+'_c18o32.npy') ))
                    
mask = np.isnan(np.load(sou_model+'fluxsum_'+model+'_co10_8d.npy').reshape(-1))

flux_8d_comp1 = np.full((6, size_4d**2), np.nan, dtype='float16')
flux_8d_comp2 = np.full((6, size_4d**2), np.nan, dtype='float16')

convert_time = time.time()
print(convert_time - start_time, 'sec elapsed. Constructing 8D models...')
for i in range(size_4d**2):
    if i%1000000 == 9:
        print(i//1000000, '/ 584 done.')
    idx = np.unravel_index(i, (size_N,size_T,size_n,size_phi,size_N,size_T,size_n,size_phi)) 
    if mask[i] == 0:  
        flux_8d_comp1[:,i] = model_4d[:, idx[0], idx[1], idx[2], idx[3]]
        flux_8d_comp2[:,i] = model_4d[:, idx[4], idx[5], idx[6], idx[7]]

np.save(sou_model+'flux_'+model+'_comp1.npy', flux_8d_comp1)
np.save(sou_model+'flux_'+model+'_comp2.npy', flux_8d_comp2)

end_time = time.time()
print(end_time - convert_time, 'sec elapsed. 8D models saved.')
   

