import numpy as np
import matplotlib.pyplot as plt

model = '4d_2comp'
output_map = '1dmax'
sou_model = 'radex_model/'
mask = np.load('mask_whole_recovered.npy')

# Constraint on the first component
constraint = 'largernH2'  #'largerTk'  #

N_co = np.load(sou_model+'Nco_'+model+'_'+output_map+'.npy') * mask
N_co[N_co == 0] = np.nan
T_k = np.load(sou_model+'Tk_'+model+'_'+output_map+'.npy') * mask
T_k[T_k == 0] = np.nan
n_h2 = np.load(sou_model+'nH2_'+model+'_'+output_map+'.npy') * mask
n_h2[n_h2 == 0] = np.nan
phi = np.load(sou_model+'phi_'+model+'_'+output_map+'.npy') * mask
phi[phi == 0] = np.nan

if constraint == 'largerTk':
    indices = T_k[0] < T_k[1]
elif constraint == 'largerPhi':
    indices = phi[0] < phi[1]
elif constraint == 'largernH2':
    indices = n_h2[0] <= n_h2[1]
    
N_co[0,indices], N_co[1,indices] = N_co[1,indices], N_co[0,indices]
T_k[0,indices], T_k[1,indices] = T_k[1,indices], T_k[0,indices]
n_h2[0,indices], n_h2[1,indices] = n_h2[1,indices], n_h2[0,indices]
phi[0,indices], phi[1,indices] = phi[1,indices], phi[0,indices]

np.save(sou_model+'Nco_'+constraint+'_'+model+'_'+output_map+'.npy', N_co)
np.save(sou_model+'Tk_'+constraint+'_'+model+'_'+output_map+'.npy', T_k)
np.save(sou_model+'nH2_'+constraint+'_'+model+'_'+output_map+'.npy', n_h2)
np.save(sou_model+'phi_'+constraint+'_'+model+'_'+output_map+'.npy', phi)

