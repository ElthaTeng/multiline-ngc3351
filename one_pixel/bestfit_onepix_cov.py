import numpy as np
import time

'''
Including the covariance matrix from calibration uncertainties, this script returns 
the minimum chi^2 value and the corresponding best-fit parameter set. The chi^2 and 
probability values of all parameter sets are saved, respectively, as 1D numpy arrays.  

'''

start_time = time.time()

idx_x = 35
idx_y = 39
nobs = 6
two_pi_n = (2 * np.pi)**nobs

model = '6d_coarse'
num_Nco = 26
num_Tk = 14
num_nH2 = 16
num_X12to13 = 20
num_X13to18 = 13
num_Phi = 20

sou_model = 'radex_model/'
sou_data = 'data_image/'

flux_mod = np.array((np.load(sou_model+'flux_'+model+'_co10.npy').reshape(-1),   
                     np.load(sou_model+'flux_'+model+'_co21.npy').reshape(-1),
                     np.load(sou_model+'flux_'+model+'_13co21.npy').reshape(-1),
                     np.load(sou_model+'flux_'+model+'_13co32.npy').reshape(-1),
                     np.load(sou_model+'flux_'+model+'_c18o21.npy').reshape(-1),
                     np.load(sou_model+'flux_'+model+'_c18o32.npy').reshape(-1)))
shape = flux_mod.shape[1]

err_co10 = np.load(sou_data+'errors/NGC3351_CO10_emom0_broad_nyq.npy')[idx_y,idx_x]
err_co21 = np.load(sou_data+'errors/NGC3351_CO21_emom0_broad_nyq.npy')[idx_y,idx_x]
err_13co21 = np.load(sou_data+'errors/NGC3351_13CO21_emom0_broad_nyq.npy')[idx_y,idx_x]
err_13co32 = np.load(sou_data+'errors/NGC3351_13CO32_emom0_broad_nyq.npy')[idx_y,idx_x]
err_c18o21 = np.load(sou_data+'errors/NGC3351_C18O21_emom0_broad_nyq.npy')[idx_y,idx_x]
err_c18o32 = np.load(sou_data+'errors/NGC3351_C18O32_emom0_broad_nyq.npy')[idx_y,idx_x]
err = np.array((err_co10,err_co21,err_13co21,err_13co32,err_c18o21,err_c18o32))

flux_co10 = np.load(sou_data+'NGC3351_CO10_mom0.npy')[idx_y,idx_x]
flux_co21 = np.load(sou_data+'NGC3351_CO21_mom0.npy')[idx_y,idx_x]
flux_13co21 = np.load(sou_data+'NGC3351_13CO21_mom0.npy')[idx_y,idx_x]
flux_13co32 = np.load(sou_data+'NGC3351_13CO32_mom0.npy')[idx_y,idx_x]
flux_c18o21 = np.load(sou_data+'NGC3351_C18O21_mom0.npy')[idx_y,idx_x]
flux_c18o32 = np.load(sou_data+'NGC3351_C18O32_mom0.npy')[idx_y,idx_x]
flux_obs = np.array((flux_co10,flux_co21,flux_13co21,flux_13co32,flux_c18o21,flux_c18o32)).reshape(nobs,1)

data_time = time.time()
print('Models & data loaded;', data_time-start_time, 'sec elapsed.')

A_cor = np.zeros((nobs,nobs))
A_cor[0,0] = 0.1**2
A_cor[1,1] = 0.1**2
for i in [2,4]:
    for j in [2,4]:
        A_cor[i,j] = 0.1**2
for i in [3,5]:
    for j in [3,5]:
        A_cor[i,j] = 0.1**2

C_bkg = np.zeros((nobs,nobs))
for i in range(nobs):
    C_bkg[i,i] = err[i]**2

C_cal = np.matmul(flux_mod.T.reshape(shape, nobs, 1), flux_mod.T.reshape(shape, 1, nobs)) * A_cor
C_tot = C_cal + C_bkg
C_inv = np.full([shape, nobs, nobs], np.nan)
Q = np.full(shape, np.nan)

for i, mat in enumerate(C_tot):
    try:
        C_inv[i] = np.linalg.inv(mat)
        Q[i] = np.sqrt(np.linalg.det(mat) * two_pi_n)
    except np.linalg.LinAlgError:
        pass

cov_time = time.time()
print('Finished computing covariance matrices;', cov_time-data_time, 'sec elapsed.')

diff = (flux_mod - flux_obs).T
chi2_grid = np.matmul(np.matmul(diff.reshape(shape, 1, nobs), C_inv), diff.reshape(shape, nobs, 1)).reshape(-1)
np.save(sou_model+'chi2_'+model+'_cov_'+str(idx_x)+'_'+str(idx_y), chi2_grid)
prob_grid = np.nan_to_num(np.exp(-0.5 * chi2_grid) / Q)
np.save(sou_model+'prob_'+model+'_cov_'+str(idx_x)+'_'+str(idx_y), prob_grid)

chi_time = time.time()
print('Finished computing chi^2 and probability grid;', chi_time-cov_time, 'sec elapsed.')

idx_min = np.unravel_index(np.nanargmin(chi2_grid), (num_Nco,num_Tk,num_nH2,num_X12to13,num_X13to18,num_Phi))
print('Minumum chi^2 =', np.nanmin(chi2_grid), 'at', idx_min) 

par_min = np.asarray(idx_min)
Nco = np.round(0.2*par_min[0] + 16., 1)
Tk = np.round(0.1*par_min[1] + 1., 1)
nH2 = np.round(0.2*par_min[2] + 2., 1)
X12to13 = np.round(10*par_min[3] + 10., 1)
X13to18 = np.round(1.5*par_min[4] + 2., 1)
Phi = np.round(0.05*par_min[5] + 0.05, 2)

print('i.e. (Nco, Tk, nH2, X(12/13), X(13/18), Phi) =', Nco, Tk, nH2, X12to13, X13to18, Phi)
end_time = time.time()
print(end_time-chi_time, 'sec elapsed. Total time:', end_time-start_time, 'sec')

