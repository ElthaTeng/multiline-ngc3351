import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

'''
Assuming zero covariances between data, this script returns the minimum chi^2 
value, the corresponding best-fit parameter set, and a contour plot showing the 
six observational constraints and the best-fit solution. The chi^2 values of
all parameter sets are saved as a 1D numpy array.  

'''

idx_x = 35
idx_y = 39
extent_nT = (2.,5.25,1,2.4)

sou_model = 'radex_model/'
sou_data = 'data_image/'
source = 'NGC3351'
model = '6d_coarse'

model_co10 = np.load(sou_model+'flux_'+model+'_co10.npy')   
model_co21 = np.load(sou_model+'flux_'+model+'_co21.npy')
model_13co21 = np.load(sou_model+'flux_'+model+'_13co21.npy')
model_13co32 = np.load(sou_model+'flux_'+model+'_13co32.npy')
model_c18o21 = np.load(sou_model+'flux_'+model+'_c18o21.npy')
model_c18o32 = np.load(sou_model+'flux_'+model+'_c18o32.npy')

flux_co10 = np.load(sou_data+source+'_CO10_mom0.npy')[idx_y,idx_x]
flux_co21 = np.load(sou_data+source+'_CO21_mom0.npy')[idx_y,idx_x]
flux_13co21 = np.load(sou_data+source+'_13CO21_mom0.npy')[idx_y,idx_x]
flux_13co32 = np.load(sou_data+source+'_13CO32_mom0.npy')[idx_y,idx_x]
flux_c18o21 = np.load(sou_data+source+'_C18O21_mom0.npy')[idx_y,idx_x]
flux_c18o32 = np.load(sou_data+source+'_C18O32_mom0.npy')[idx_y,idx_x]

noise_co10 = np.load(sou_data+'errors/'+source+'_CO10_emom0_broad_nyq.npy')[idx_y,idx_x]
noise_co21 = np.load(sou_data+'errors/'+source+'_CO21_emom0_broad_nyq.npy')[idx_y,idx_x]
noise_13co21 = np.load(sou_data+'errors/'+source+'_13CO21_emom0_broad_nyq.npy')[idx_y,idx_x]
noise_13co32 = np.load(sou_data+'errors/'+source+'_13CO32_emom0_broad_nyq.npy')[idx_y,idx_x]
noise_c18o21 = np.load(sou_data+'errors/'+source+'_C18O21_emom0_broad_nyq.npy')[idx_y,idx_x]
noise_c18o32 = np.load(sou_data+'errors/'+source+'_C18O32_emom0_broad_nyq.npy')[idx_y,idx_x]

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

# Compute minimum chi^2 and its correspoding parameter set
chi_sum = ((model_co10 - flux_co10) / err_co10)**2
chi_sum = chi_sum + ((model_co21 - flux_co21) / err_co21)**2
chi_sum = chi_sum + ((model_13co21 - flux_13co21) / err_13co21)**2
chi_sum = chi_sum + ((model_13co32 - flux_13co32) / err_13co32)**2
chi_sum = chi_sum + ((model_c18o21 - flux_c18o21) / err_c18o21)**2
chi_sum = chi_sum + ((model_c18o32 - flux_c18o32) / err_c18o32)**2

idx_min = np.unravel_index(np.nanargmin(chi_sum, axis=None), chi_sum.shape)
par_min = np.asarray(idx_min)
Nco = np.round(0.2*par_min[0] + 16., 1)
T_best = 0.1*par_min[1] + 1.
n_best = 0.2*par_min[2] + 2.
X12to13 = np.round(10*par_min[3] + 10., 1)
X13to18 = np.round(1.5*par_min[4] + 2., 1)
Phi = np.round(0.05*par_min[5] + 0.05, 2)

print('Minumum chi^2 =', np.nanmin(chi_sum), 'at', idx_min)
print('i.e. (Nco, Tk, nH2, X(12/13), X(13/18), Phi) =', Nco, T_best, n_best, X12to13, X13to18, Phi)
np.save(sou_model+'chi2_'+model+'_'+str(idx_x)+'_'+str(idx_y), chi_sum)

idx_N = par_min[0]
idx_X1 = par_min[3]
idx_X2 = par_min[4]
idx_Phi = par_min[5]

# Contour plots of the n-to-T slices 
slice_co10 = model_co10[idx_N,:,:,idx_X1,idx_X2,idx_Phi]  
slice_co21 = model_co21[idx_N,:,:,idx_X1,idx_X2,idx_Phi]
slice_13co21 = model_13co21[idx_N,:,:,idx_X1,idx_X2,idx_Phi]
slice_13co32 = model_13co32[idx_N,:,:,idx_X1,idx_X2,idx_Phi]
slice_c18o21 = model_c18o21[idx_N,:,:,idx_X1,idx_X2,idx_Phi]
slice_c18o32 = model_c18o32[idx_N,:,:,idx_X1,idx_X2,idx_Phi]

con_co10 = plt.contour(slice_co10,origin='lower',levels=np.array((flux_co10-err_co10,flux_co10+err_co10)), extent=extent_nT, colors='gray')
con_co21 = plt.contour(slice_co21,origin='lower',levels=np.array((flux_co21-err_co21,flux_co21+err_co21)), extent=extent_nT, colors='k', linestyles='dashed')
con_13co21 = plt.contour(slice_13co21,origin='lower',levels=np.array((flux_13co21-err_13co21,flux_13co21+err_13co21)), extent=extent_nT, colors='c')
con_13co32 = plt.contour(slice_13co32,origin='lower',levels=np.array((flux_13co32-err_13co32,flux_13co32+err_13co32)), extent=extent_nT, colors='b', linestyles='dotted')
con_c18o21 = plt.contour(slice_c18o21,origin='lower',levels=np.array((flux_c18o21-err_c18o21,flux_c18o21+err_c18o32)), extent=extent_nT, colors='m')
con_c18o32 = plt.contour(slice_c18o32,origin='lower',levels=np.array((flux_c18o32-err_c18o32,flux_c18o32+err_c18o32)), extent=extent_nT, colors='r', linestyles='dashdot')

plt.gca().clabel(con_co10, inline=1, fontsize=10, fmt='%1.1f')
plt.gca().clabel(con_co21, inline=1, fontsize=10, fmt='%1.1f')
plt.gca().clabel(con_13co21, inline=1, fontsize=10, fmt='%1.1f')
plt.gca().clabel(con_13co32, inline=1, fontsize=10, fmt='%1.1f')
plt.gca().clabel(con_c18o21, inline=1, fontsize=10, fmt='%1.1f')
plt.gca().clabel(con_c18o32, inline=1, fontsize=10, fmt='%1.1f')

line_co10 = mlines.Line2D([], [], color='gray', label='CO 1-0')
line_co21 = mlines.Line2D([], [], color='k', label='CO 2-1', ls='--')
line_13co21 = mlines.Line2D([], [], color='c', label='13CO 2-1')
line_13co32 = mlines.Line2D([], [], color='b', label='13CO 3-2', ls=':')
line_c18o21 = mlines.Line2D([], [], color='m', label='C18O 2-1')
line_c18o32 = mlines.Line2D([], [], color='r', label='C18O 3-2', ls='-.')
legend = plt.legend(handles=[line_co10,line_co21,line_13co21,line_13co32,line_c18o21,line_c18o32], loc='upper right')  #, prop=lprop

plt.title(r'$N_{CO}$ ='+str(Nco)+r'; $X_{12/13}$ ='+str(X12to13)+r'; $X_{13/18}$ ='+str(X13to18)+r'; $\Phi_{bf}$ ='+str(Phi))
plt.fill_between([n_best,n_best+0.2], [T_best,T_best], [T_best+0.1,T_best+0.1], color='red', alpha='0.7')
plt.ylabel(r'$\log\ T_k\ (K)$', fontsize=12)
plt.xlabel(r'$\log\ n_{H_2}\ (cm^{-3})$', fontsize=12)
plt.savefig(sou_model+'flux_'+model+'_contours_'+str(idx_x)+'_'+str(idx_y)+'.pdf', bbox_inches='tight', format='pdf')
plt.tight_layout()
plt.show()

# plt.imshow(slice_c18o21,origin='lower')
# plt.colorbar()
# plt.show()
