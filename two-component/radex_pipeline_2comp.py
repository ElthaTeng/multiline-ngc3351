import os
import numpy as np
from joblib import Parallel, delayed
import time

start_time = time.time()

# Parameter Settings
molecule_0 = 'co'
molecule_1 = '13co'
molecule_2 = 'c18o'
model = '3d_2comp'

sou_model = 'radex_model/'
num_cores = 20

linewidth = 15
Nco = np.arange(16.,19.1,0.25)
Tkin = np.arange(1.,2.1,0.1)
nH2 = np.arange(2.,5.1,0.25)
X_13co = 25
X_c18o = 8
round_dens = 1
round_temp = 1

# Pre-processing
incr_dens = round(Nco[1] - Nco[0],2)
incr_temp = round(Tkin[1] - Tkin[0],1)
co_dex = np.round(10**np.arange(0.,1.,incr_dens), 4)
Tk_dex = np.round(10**np.arange(0.,1.,incr_temp), 4)
num_Nco = Nco.shape[0]
num_Tk = Tkin.shape[0]
num_nH2 = nH2.shape[0]
factors_13co = 1./X_13co  
factors_c18o = 1./X_c18o
cycle_dens = co_dex.shape[0]
cycle_temp = Tk_dex.shape[0]
diff_Tk = Tkin[1] - Tkin[0]

def write_inputs(i,j,k):
    powi = str(i//cycle_temp + int(Tkin[0]))
    Tk = str(round(diff_Tk*i + int(Tkin[0]), round_temp))
    powj = j//cycle_dens + int(nH2[0])
    n_h2 = str(powj)
    N_co = str(k//cycle_dens + int(Nco[0]))
    
    prei = str(Tk_dex[i%cycle_temp])
    prej = str(co_dex[j%cycle_dens])
    prej_r = str(round(co_dex[j%cycle_dens], round_dens))
    prek_m0 = str(co_dex[k%cycle_dens])
    codex = str(round(co_dex[k%cycle_dens], round_dens))

    file = open('input_'+model+'_'+molecule_0+'/'+Tk+'_'+prej_r+'e'+n_h2+'_'+codex+'e'+N_co+'.inp','w')
    file.write(molecule_0+'.dat\n')
    file.write('output_'+model+'_'+molecule_0+'/'+Tk+'_'+prej_r+'e'+n_h2+'_'+codex+'e'+N_co+'.out\n')
    file.write('100'+' '+'300'+'\n')
    file.write(prei+'e'+powi+'\n')
    file.write('1\n')
    file.write('H2\n')
    file.write(prej+'e'+n_h2+'\n')
    file.write('2.73'+'\n')
    file.write(prek_m0+'e'+N_co+'\n')
    file.write(str(linewidth)+'\n')
    file.write('0\n')
    file.close() 
    
    prek_m1 = str(round(factors_13co*co_dex[k%cycle_dens],4))
    x13co = str(X_13co)
    file = open('input_'+model+'_'+molecule_1+'/'+Tk+'_'+prej_r+'e'+n_h2+'_'+codex+'e'+N_co+'_'+x13co+'.inp','w')
    file.write(molecule_1+'.dat\n')
    file.write('output_'+model+'_'+molecule_1+'/'+Tk+'_'+prej_r+'e'+n_h2+'_'+codex+'e'+N_co+'_'+x13co+'.out\n')
    file.write('200'+' '+'400'+'\n')
    file.write(prei+'e'+powi+'\n')
    file.write('1\n')
    file.write('H2\n')
    file.write(prej+'e'+n_h2+'\n')
    file.write('2.73'+'\n')
    file.write(prek_m1+'e'+N_co+'\n')
    file.write(str(linewidth)+'\n')
    file.write('0\n')
    file.close() 
    
    prek_m2 = str(round(factors_c18o*factors_13co*co_dex[k%cycle_dens],6))
    xc18o = str(X_c18o)
    file = open('input_'+model+'_'+molecule_2+'/'+Tk+'_'+prej_r+'e'+n_h2+'_'+codex+'e'+N_co+'_'+x13co+'_'+xc18o+'.inp','w')
    file.write(molecule_2+'.dat\n')
    file.write('output_'+model+'_'+molecule_2+'/'+Tk+'_'+prej_r+'e'+n_h2+'_'+codex+'e'+N_co+'_'+x13co+'_'+xc18o+'.out\n')
    file.write('200'+' '+'400'+'\n')
    file.write(prei+'e'+powi+'\n')
    file.write('1\n')
    file.write('H2\n')
    file.write(prej+'e'+n_h2+'\n')
    file.write('2.73'+'\n')
    file.write(prek_m2+'e'+N_co+'\n')
    file.write(str(linewidth)+'\n')
    file.write('0\n')
    file.close()   

def run_radex(i,j,k):
    powj = j//cycle_dens + int(nH2[0])
    Tk = str(round(diff_Tk*i + int(Tkin[0]), round_temp))
    n_h2 = str(powj)
    N_co = str(k//cycle_dens + int(Nco[0]))
    
    prej_r = str(round(co_dex[j%cycle_dens], round_dens))
    codex = str(round(co_dex[k%cycle_dens], round_dens))
    x13co = str(X_13co)
    xc18o = str(X_c18o)
    
    os.system('radex < input_'+model+'_'+molecule_0+'/'+Tk+'_'+prej_r+'e'+n_h2+'_'+codex+'e'+N_co+'.inp')
    os.system('radex < input_'+model+'_'+molecule_1+'/'+Tk+'_'+prej_r+'e'+n_h2+'_'+codex+'e'+N_co+'_'+x13co+'.inp')
    os.system('radex < input_'+model+'_'+molecule_2+'/'+Tk+'_'+prej_r+'e'+n_h2+'_'+codex+'e'+N_co+'_'+x13co+'_'+xc18o+'.inp')

def radex_flux(i,j,k):
    powj = j//cycle_dens + int(nH2[0])
    Tk = str(round(diff_Tk*i + int(Tkin[0]), round_temp))
    n_h2 = str(powj)
    N_co = str(k//cycle_dens + int(Nco[0]))
    
    prej_r = str(round(co_dex[j%cycle_dens], round_dens))   
    x13co = str(X_13co)
    xc18o = str(X_c18o)
    codex = str(round(co_dex[k%cycle_dens], round_dens))
    
    flux_0 = np.genfromtxt('output_'+model+'_'+molecule_0+'/'+Tk+'_'+prej_r+'e'+n_h2+'_'+codex+'e'+N_co+'.out', skip_header=13)[:,11]
    flux_1 = np.genfromtxt('output_'+model+'_'+molecule_1+'/'+Tk+'_'+prej_r+'e'+n_h2+'_'+codex+'e'+N_co+'_'+x13co+'.out', skip_header=13)[:,11]
    flux_2 = np.genfromtxt('output_'+model+'_'+molecule_2+'/'+Tk+'_'+prej_r+'e'+n_h2+'_'+codex+'e'+N_co+'_'+x13co+'_'+xc18o+'.out', skip_header=13)[:,11]
    
    return k,i,j,flux_0,flux_1,flux_2
 
 
# Run input files for molecules 0,1,2    
Parallel(n_jobs=num_cores)(delayed(write_inputs)(i,j,k) for k in range(num_Nco) for j in range(num_nH2) for i in range(num_Tk))              
input_time = time.time()
print('Elapsed time for writing inputs: %s sec' % ((input_time - start_time)))

# Run RADEX for molecules 0,1,2
Parallel(n_jobs=num_cores)(delayed(run_radex)(i,j,k) for k in range(num_Nco) for j in range(num_nH2) for i in range(num_Tk))  
radex_time = time.time() 
print('Elapsed time for running RADEX: %s sec' % ((radex_time - input_time)))

# Construct 3D flux models
results = Parallel(n_jobs=num_cores)(delayed(radex_flux)(i,j,k) for k in range(num_Nco) for j in range(num_nH2) for i in range(num_Tk))

flux_co10 =  np.full((num_Nco,num_Tk,num_nH2),np.nan)
flux_co21 = np.full((num_Nco,num_Tk,num_nH2),np.nan)
flux_13co21 = np.full((num_Nco,num_Tk,num_nH2),np.nan)
flux_13co32 = np.full((num_Nco,num_Tk,num_nH2),np.nan)
flux_c18o21 = np.full((num_Nco,num_Tk,num_nH2),np.nan)
flux_c18o32 = np.full((num_Nco,num_Tk,num_nH2),np.nan)

for result in results:
    k, i, j, flux_0, flux_1, flux_2 = result
    flux_co10[k,i,j] = flux_0[0]
    flux_co21[k,i,j] = flux_0[1]
    flux_13co21[k,i,j] = flux_1[0]
    flux_13co32[k,i,j] = flux_1[1]
    flux_c18o21[k,i,j] = flux_2[0]
    flux_c18o32[k,i,j] = flux_2[1]

np.save(sou_model+'flux_'+model+'_co10.npy', flux_co10)
np.save(sou_model+'flux_'+model+'_co21.npy', flux_co21)
np.save(sou_model+'flux_'+model+'_13co21.npy', flux_13co21)
np.save(sou_model+'flux_'+model+'_13co32.npy', flux_13co32)
np.save(sou_model+'flux_'+model+'_c18o21.npy', flux_c18o21)
np.save(sou_model+'flux_'+model+'_c18o32.npy', flux_c18o32) 
print('Flux models saved; elapsed time for construction: %s sec' % ((time.time() - radex_time)))

# Construct 3D ratio models
'''temp = np.repeat(flux_co21[:, :, :, np.newaxis], num_X12to13, axis=3)
co21_5d = np.repeat(temp[:, :, :, :, np.newaxis], num_X13to18, axis=4)
temp2 = np.repeat(flux_co10[:, :, :, np.newaxis], num_X12to13, axis=3)
co10_5d = np.repeat(temp2[:, :, :, :, np.newaxis], num_X13to18, axis=4)
c13o_21_5d = np.repeat(flux_13co21[:, :, :, :, np.newaxis], num_X13to18, axis=4)
c13o_32_5d = np.repeat(flux_13co32[:, :, :, :, np.newaxis], num_X13to18, axis=4) 
'''
ratio_co = flux_co21/flux_co10
ratio_13co = flux_13co32/flux_13co21
ratio_c18o = flux_c18o32/flux_c18o21
ratio_co213co = flux_co21/flux_13co21
ratio_co2c18o = flux_co21/flux_c18o21
ratio_13co2c18o_21 = flux_13co21/flux_c18o21
ratio_13co2c18o_32 = flux_13co32/flux_c18o32

np.save(sou_model+'ratio_'+model+'_co_21_10.npy',ratio_co)
np.save(sou_model+'ratio_'+model+'_13co_32_21.npy',ratio_13co)
np.save(sou_model+'ratio_'+model+'_c18o_32_21.npy',ratio_c18o)
np.save(sou_model+'ratio_'+model+'_co_13co_21.npy',ratio_co213co)
np.save(sou_model+'ratio_'+model+'_co_c18o_21.npy',ratio_co2c18o)
np.save(sou_model+'ratio_'+model+'_13co_c18o_21.npy',ratio_13co2c18o_21)
np.save(sou_model+'ratio_'+model+'_13co_c18o_32.npy',ratio_13co2c18o_32)
print('Ratio models saved.')

end_time = time.time()
print('Total lapsed time: %s sec' % ((end_time - start_time)))
