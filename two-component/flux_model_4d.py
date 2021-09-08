import numpy as np
from joblib import Parallel, delayed
import time

'''
This python script expands all the flux models returned by 'radex_pipeline_2comp.py' to 4D 
flux models by adding a new axis varying the beam-filling factor.

Inputs:
    sou_model: (str) directory name for output model grids
    base_model: (str) name of the original flux model to expand
    model: (str) name of the output 4D model after expansion
    num_Nco,num_Tk,num_nH2: (int) shape of the 3D base model
    beam_fill: (array) sampling values for beam-filling factors, 
                       e.g. np.arange(0.05, 1.01, 0.05)

Outputs:
    expanded 4D models: six numpy array grids saved to the given sou_model directory as 
                        e.g. 'flux_4d_2comp_co10.npy'
'''

time_ini = time.time()

# Input Settings
sou_model = 'radex_model/'
base_model = '3d_2comp'
model = '4d_2comp'
num_Nco = 13
num_Tk = 11
num_nH2 = 13
beam_fill = 10**np.arange(-1.3, -0.05, 0.1)
num_phi = beam_fill.shape[0]
num_4d = num_Nco*num_Tk*num_nH2*num_phi

# import flux data
flux_co10 = np.load(sou_model+'flux_'+base_model+'_co10.npy')
flux_co21 = np.load(sou_model+'flux_'+base_model+'_co21.npy')
flux_13co21 = np.load(sou_model+'flux_'+base_model+'_13co21.npy')
flux_13co32 = np.load(sou_model+'flux_'+base_model+'_13co32.npy')
flux_c18o21 = np.load(sou_model+'flux_'+base_model+'_c18o21.npy')
flux_c18o32 = np.load(sou_model+'flux_'+base_model+'_c18o32.npy')

# Construct 4d flux models from 3d by adding the beam filling factor dimension
flux_co10_4d = flux_co10.reshape(num_Nco,num_Tk,num_nH2,1) * beam_fill.reshape(1,1,1,num_phi)
flux_co21_4d = flux_co21.reshape(num_Nco,num_Tk,num_nH2,1) * beam_fill.reshape(1,1,1,num_phi)
flux_13co21_4d = flux_13co21.reshape(num_Nco,num_Tk,num_nH2,1) * beam_fill.reshape(1,1,1,num_phi)
flux_13co32_4d = flux_13co32.reshape(num_Nco,num_Tk,num_nH2,1) * beam_fill.reshape(1,1,1,num_phi)
flux_c18o21_4d = flux_c18o21.reshape(num_Nco,num_Tk,num_nH2,1) * beam_fill.reshape(1,1,1,num_phi)
flux_c18o32_4d = flux_c18o32.reshape(num_Nco,num_Tk,num_nH2,1) * beam_fill.reshape(1,1,1,num_phi)

np.save(sou_model+'flux_'+model+'_co10.npy', flux_co10_4d)
np.save(sou_model+'flux_'+model+'_co21.npy', flux_co21_4d)
np.save(sou_model+'flux_'+model+'_13co21.npy', flux_13co21_4d)
np.save(sou_model+'flux_'+model+'_13co32.npy', flux_13co32_4d)
np.save(sou_model+'flux_'+model+'_c18o21.npy', flux_c18o21_4d)
np.save(sou_model+'flux_'+model+'_c18o32.npy', flux_c18o32_4d)

# Construct 2-component flux models by adding up all combinations of fluxes
'''shape = np.triu_indices(num_4d)[0].shape
flux_2comp_co10 = np.full(shape, np.nan, dtype='float32')
flux_2comp_co21 = np.full(shape, np.nan, dtype='float32')
flux_2comp_13co21 = np.full(shape, np.nan, dtype='float32')
flux_2comp_13co32 = np.full(shape, np.nan, dtype='float32')
flux_2comp_c18o21 = np.full(shape, np.nan, dtype='float32')
flux_2comp_c18o32 = np.full(shape, np.nan, dtype='float32')
'''
flux_2comp_co10 = np.full((num_4d,num_4d), np.nan, dtype='float32')
flux_2comp_co21 = np.full((num_4d,num_4d), np.nan, dtype='float32')
flux_2comp_13co21 = np.full((num_4d,num_4d), np.nan, dtype='float32')
flux_2comp_13co32 = np.full((num_4d,num_4d), np.nan, dtype='float32')
flux_2comp_c18o21 = np.full((num_4d,num_4d), np.nan, dtype='float32')
flux_2comp_c18o32 = np.full((num_4d,num_4d), np.nan, dtype='float32')

def model_2comp(i,j):
    fluxsum_co10 = (flux_co10_4d.reshape(-1))[i] + (flux_co10_4d.reshape(-1))[j]
    fluxsum_co21 = (flux_co21_4d.reshape(-1))[i] + (flux_co21_4d.reshape(-1))[j]
    fluxsum_13co21 = (flux_13co21_4d.reshape(-1))[i] + (flux_13co21_4d.reshape(-1))[j]
    fluxsum_13co32 = (flux_13co32_4d.reshape(-1))[i] + (flux_13co32_4d.reshape(-1))[j]
    fluxsum_c18o21 = (flux_c18o21_4d.reshape(-1))[i] + (flux_c18o21_4d.reshape(-1))[j]
    fluxsum_c18o32 = (flux_c18o32_4d.reshape(-1))[i] + (flux_c18o32_4d.reshape(-1))[j]
    return i, j, fluxsum_co10, fluxsum_co21, fluxsum_13co21, fluxsum_13co32, fluxsum_c18o21, fluxsum_c18o32

results = Parallel(n_jobs=10, verbose=5)(delayed(model_2comp)(i,j) for i in range(num_4d) for j in range(i+1))

for result in results:
    i, j, fluxsum_co10, fluxsum_co21, fluxsum_13co21, fluxsum_13co32, fluxsum_c18o21, fluxsum_c18o32 = result
    flux_2comp_co10[j,i] = fluxsum_co10
    flux_2comp_co21[j,i] = fluxsum_co21
    flux_2comp_13co21[j,i] = fluxsum_13co21
    flux_2comp_13co32[j,i] = fluxsum_13co32
    flux_2comp_c18o21[j,i] = fluxsum_c18o21
    flux_2comp_c18o32[j,i] = fluxsum_c18o32

np.save(sou_model+'fluxsum_'+model+'_co10_2d.npy', flux_2comp_co10)
np.save(sou_model+'fluxsum_'+model+'_co21_2d.npy', flux_2comp_co21)
np.save(sou_model+'fluxsum_'+model+'_13co21_2d.npy', flux_2comp_13co21)
np.save(sou_model+'fluxsum_'+model+'_13co32_2d.npy', flux_2comp_13co32)
np.save(sou_model+'fluxsum_'+model+'_c18o21_2d.npy', flux_2comp_c18o21)
np.save(sou_model+'fluxsum_'+model+'_c18o32_2d.npy', flux_2comp_c18o32)    

time = round(time.time() - time_ini, 1)
print('Total elapsed time:', time, 'sec.') 

