import numpy as np

'''
This python script expands all the flux models returned by 'radex_pipeline.py' to 6D 
flux models by adding a new axis varying the beam-filling factor.

Inputs:
    sou_model: (str) directory name for output model grids
    base_model: (str) name of the original flux model to expand
    model: (str) name of the output 6D model after expansion
    num_Nco,num_Tk,num_nH2,num_X12to13,num_X13to18: (int) shape of the 5D base model
    beam_fill: (array) sampling values for beam-filling factors, 
                       e.g. np.arange(0.05, 1.01, 0.05)

Outputs:
    expanded 6D models: six numpy array grids saved to the given sou_model directory as 
                        e.g. 'flux_6d_coarse_co10.npy'
'''

# Input Settings
sou_model = 'radex_model/'
base_model = '5d_coarse'
model = '6d_coarse'
num_Nco = 26
num_Tk = 14
num_nH2 = 16
num_X12to13 = 20
num_X13to18 = 13
beam_fill = np.arange(0.05, 1.01, 0.05)

# import flux data
flux_co10 = np.load(sou_model+'flux_'+base_model+'_co10.npy')
flux_co21 = np.load(sou_model+'flux_'+base_model+'_co21.npy')
flux_13co21 = np.load(sou_model+'flux_'+base_model+'_13co21.npy')
flux_13co32 = np.load(sou_model+'flux_'+base_model+'_13co32.npy')
flux_c18o21_5d = np.load(sou_model+'flux_'+base_model+'_c18o21.npy')
flux_c18o32_5d = np.load(sou_model+'flux_'+base_model+'_c18o32.npy')

temp = np.repeat(flux_co21[:, :, :, np.newaxis], num_X12to13, axis=3)
flux_co21_5d = np.repeat(temp[:, :, :, :, np.newaxis], num_X13to18, axis=4)
temp2 = np.repeat(flux_co10[:, :, :, np.newaxis], num_X12to13, axis=3)
flux_co10_5d = np.repeat(temp2[:, :, :, :, np.newaxis], num_X13to18, axis=4)
flux_13co21_5d = np.repeat(flux_13co21[:, :, :, :, np.newaxis], num_X13to18, axis=4)
flux_13co32_5d = np.repeat(flux_13co32[:, :, :, :, np.newaxis], num_X13to18, axis=4) 

# Construct 6d flux models from 5d by adding the beam filling factor dimension
flux_co10_6d = flux_co10_5d.reshape(num_Nco,num_Tk,num_nH2,num_X12to13,num_X13to18,1) * beam_fill.reshape(1,1,1,1,1,beam_fill.shape[0])
flux_co21_6d = flux_co21_5d.reshape(num_Nco,num_Tk,num_nH2,num_X12to13,num_X13to18,1) * beam_fill.reshape(1,1,1,1,1,beam_fill.shape[0])
flux_13co21_6d = flux_13co21_5d.reshape(num_Nco,num_Tk,num_nH2,num_X12to13,num_X13to18,1) * beam_fill.reshape(1,1,1,1,1,beam_fill.shape[0])
flux_13co32_6d = flux_13co32_5d.reshape(num_Nco,num_Tk,num_nH2,num_X12to13,num_X13to18,1) * beam_fill.reshape(1,1,1,1,1,beam_fill.shape[0])
flux_c18o21_6d = flux_c18o21_5d.reshape(num_Nco,num_Tk,num_nH2,num_X12to13,num_X13to18,1) * beam_fill.reshape(1,1,1,1,1,beam_fill.shape[0])
flux_c18o32_6d = flux_c18o32_5d.reshape(num_Nco,num_Tk,num_nH2,num_X12to13,num_X13to18,1) * beam_fill.reshape(1,1,1,1,1,beam_fill.shape[0])

np.save(sou_model+'flux_'+model+'_co10.npy', flux_co10_6d)
np.save(sou_model+'flux_'+model+'_co21.npy', flux_co21_6d)
np.save(sou_model+'flux_'+model+'_13co21.npy', flux_13co21_6d)
np.save(sou_model+'flux_'+model+'_13co32.npy', flux_13co32_6d)
np.save(sou_model+'flux_'+model+'_c18o21.npy', flux_c18o21_6d)
np.save(sou_model+'flux_'+model+'_c18o32.npy', flux_c18o32_6d)