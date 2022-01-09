import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

idx_x = 34
idx_y = 43

model = '6d_coarse_rmcor'
sou_model = 'radex_model/ngc3351_npy/'

# Set parameter ranges
samples_Nco = np.arange(16.,21.1,0.2)
samples_Tk = np.arange(1.,2.4,0.1)
samples_nH2 = np.arange(2.,5.1,0.2)
samples_X12to13 = np.arange(10,205,10)
samples_X13to18 = np.arange(2,21,1.5)
samples_phi = np.arange(0.05, 1.01, 0.05)

size_N = samples_Nco.shape[0]
size_T = samples_Tk.shape[0]
size_n = samples_nH2.shape[0]
size_x12to13 = samples_X12to13.shape[0]
size_x13to18 = samples_X13to18.shape[0]
size_phi = samples_phi.shape[0]

Nco = samples_Nco.reshape(size_N,1,1,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
Nco = Nco.reshape(-1)
nH2 = samples_nH2.reshape(1,1,size_n,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
nH2 = nH2.reshape(-1)
phi = samples_phi.reshape(1,1,1,1,1,size_phi)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
phi = phi.reshape(-1)

# Set up constraints if using priors
los_max = 100.
x_co = 3 * 10**(-4)
map_ew = fits.open('data_image/NGC3351_CO10_ew_broad_nyq.fits')[0].data
map_fwhm = map_ew * 2.35  # Conversion of 1 sigma to FWHM assuming Gaussian
los_length = (10**Nco / 15. * map_fwhm[idx_y,idx_x]) / (np.sqrt(phi) * 10**nH2 * x_co)  
mask = los_length < los_max * (3.086 * 10**18) 
print(np.sum(mask),'parameter sets have line-of-sight length smaller than',los_max,'pc')

# Generate masked alpha_co grid
Xco2alpha = 1 / (4.5 * 10**19) / x_co
Imod_co10 = np.load(sou_model+'flux_6d_coarse_co10.npy').reshape(-1)
alpha = (10**Nco / 15. * map_fwhm[idx_y,idx_x] * phi / Imod_co10 * mask * Xco2alpha).reshape(-1)
alpha[mask == 0] = np.nan
alpha = np.log10(alpha)
print(np.nanmax(alpha), np.nanmin(alpha))

# Compute masked chi2 and prob
chi2 = np.load(sou_model+'chi2_'+model+'_'+str(idx_x)+'_'+str(idx_y)+'.npy').reshape(-1)
prob = np.exp(-0.5*chi2).reshape(-1)
chi2_masked = chi2 * mask
chi2_masked[mask == 0] = np.nan
prob = prob * mask
print(prob.mean(), prob.max(), prob.min())

# 1D alpha_co likelihood
num_bins = 40
counts_noweight, bins = np.histogram(alpha, bins=num_bins, range=(-2.5,2.5), weights=None, density=True)
counts_weighted, bins = np.histogram(alpha, bins=num_bins, range=(-2.5,2.5), weights=prob)
counts_norm = np.nan_to_num(counts_weighted / counts_noweight)
counts = np.array((counts_noweight, counts_weighted, counts_norm))
titles = np.array(('uniformly-weighted','probability-weighted','normalized'))

plt.figure(figsize=(8,3))
plt.suptitle('(x,y) = ('+str(idx_x)+','+str(idx_y)+')')
for i in range(3):
	plt.subplot(1,3,i+1)
	plt.title(titles[i])
	plt.tick_params(axis="y", labelleft=False)
	plt.hist(bins[:-1], bins, weights=counts[i], log=False, histtype='step', color='k')
	plt.xlabel(r'$\log(\alpha_{CO})$')
plt.subplots_adjust(wspace=0.3)
plt.savefig('radex_model/ngc3351_pdf/prob1d_alpha_'+str(idx_x)+'_'+str(idx_y)+'.png', bbox_inches='tight', pad_inches=0.1)
plt.show()

