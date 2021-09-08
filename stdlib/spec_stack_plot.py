from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

num_chan = 76  # number of channels per side 

def spec_extract(file, idx_x, idx_y):
    mom1 = fits.open('data_image/NGC3351_CO21_mom1_broad_nyq.fits')[0].data[idx_y,idx_x]
    cube = file[0].data
    header = file[0].header
    v_ref = header['CRVAL3']/1000
    v_del = header['CDELT3']/1000
    RefPix = header['CRPIX3']
    N_vel = header['NAXIS3']
    
    v_list = np.arange(v_ref-v_del*(RefPix-1), v_ref+v_del*(-RefPix+N_vel+0.5), v_del)
    mom1_idx = (np.abs(v_list - mom1)).argmin()
    spec = cube[mom1_idx-num_chan:mom1_idx+num_chan,idx_y,idx_x]
    return spec

region = 'center'  # center, ring, arms
line = 'C18O10'

if region == 'center':
    mask = np.load('mask_cent3sig.npy')
elif region == 'ring':
    mask = np.load('mask_ring.npy')
elif region == 'arms':
    mask = np.load('mask_arms.npy') * np.load('old_masks/mask_whole.npy')

indices = np.stack((np.nonzero(mask)[0],np.nonzero(mask)[1]),axis=1)
N_pix = indices.shape[0]

v_extract = np.arange(-num_chan,num_chan)

spec_stacked = np.zeros((num_chan*2,))
file = fits.open('data_cube/NGC3351_'+line+'_cube_nyq.fits')

for i in range(N_pix):
    spec = spec_extract(file, indices[i,1],indices[i,0])
    spec_stacked += spec

spec_avg = spec_stacked/N_pix
num_chan_noise = int((spec_avg.shape[0] - 80)/2)
print(num_chan_noise)

print(spec_avg[:num_chan_noise].mean(), spec_avg[:num_chan_noise].std())
print(spec_avg[-num_chan_noise:].mean(), spec_avg[-num_chan_noise:].std())
#print(spec_avg.mean(), spec_avg.std())

#np.save('data_cube/spec_stack_'+line+'_'+region+'.npy', spec_avg)

plt.step(v_extract,spec_avg,c='#8E8E8E',lw=2,where='mid')
plt.show()