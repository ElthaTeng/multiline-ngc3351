import numpy as np
import matplotlib.pyplot as plt
import corner
import time

'''
This script generates a corner plot showing 1D and 2D likelihood distributions for
all six parameters (dimensions) by loading in the chi^2/probability arrays returned 
from 'bestfit_onepix.py' or 'bestfit_onepix_cov.py'. Note that 'bestfit_onepix.py'
only saves the chi^2 array, so the probability should be computed in this script.

'''

start_time = time.time()
print('Constructing inputs for corner plot...')

idx_x = 36
idx_y = 43
model = '6d_coarse_cov'
sou_model = 'radex_model/'
chi2 = np.load(sou_model+'chi2_'+model+'_'+str(idx_x)+'_'+str(idx_y)+'.npy')

'''If use the output from 'bestfit_onepix_cov.py', just load in. Otherwise, compute probability here. '''
prob = np.load(sou_model+'prob_'+model+'_'+str(idx_x)+'_'+str(idx_y)+'.npy')
#prob = np.exp(-0.5*chi2).reshape(-1)

N_co = np.arange(16., 21.1, 0.2).astype('float16')
T_k = np.arange(1.,2.4,0.1).astype('float16') 
n_h2 = np.arange(2., 5.1, 0.2).astype('float16')
X_12to13 = np.arange(10, 201, 10).astype('float16')
X_13to18 = np.arange(2, 21, 1.5).astype('float16')
phi = np.arange(0.05, 1.01, 0.05).astype('float16')

size_N = N_co.shape[0]
size_T = T_k.shape[0]
size_n = n_h2.shape[0]
size_x12to13 = X_12to13.shape[0]
size_x13to18 = X_13to18.shape[0]
size_phi = phi.shape[0]

N_co = N_co.reshape(size_N,1,1,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi),dtype='float16')
T_k = T_k.reshape(1,size_T,1,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi),dtype='float16')
n_h2 = n_h2.reshape(1,1,size_n,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi),dtype='float16')
X_12to13 = X_12to13.reshape(1,1,1,size_x12to13,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi),dtype='float16')
X_13to18 = X_13to18.reshape(1,1,1,1,size_x13to18,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi),dtype='float16')
phi = phi.reshape(1,1,1,1,1,size_phi)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi),dtype='float16')

N_co = N_co.reshape(-1)
T_k = T_k.reshape(-1)
n_h2 = n_h2.reshape(-1)
X_12to13 = X_12to13.reshape(-1)
X_13to18 = X_13to18.reshape(-1)
phi = phi.reshape(-1)

input = np.stack((N_co, T_k, n_h2, X_12to13, X_13to18, phi), axis=-1)

corner_time = time.time()
print(corner_time-start_time, 'sec elapsed. Generating corner plot')
figure = corner.corner(input, bins=[size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi], plot_datapoints=False, 
                       labels=[r"$N_{CO}$", r"$T_{kin}$", r"$n_{H_2}$", r"$X_{12/13}$", r"$X_{13/18}$", r"$\Phi_{bf}$"], 
                       show_titles=True, title_kwargs={"fontsize": 12}, weights=prob)

plt.savefig(sou_model+'corner_'+model+'_'+str(idx_x)+'_'+str(idx_y)+'.pdf')
end_time = time.time()
print(end_time-corner_time, 'sec elapsed. Corner plot saved.')
print('Total time elapsed =', end_time-start_time, 'sec')

plt.show(figure)

