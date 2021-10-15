import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import Angle

def radius_arcsec(shape, w, ra, dec, pa, incl,
                  incl_correction=False, cosINCL_limit=0.5):
    # All inputs assumed as Angle
    if incl_correction and (np.isnan(pa.rad + incl.rad)):
        pa = Angle(0 * u.rad)
        incl = Angle(0 * u.rad)
        # Not written to the header
        msg = '\n::z0mgs:: PA or INCL is NaN in ' + \
            'radius calculation \n' + \
            '::z0mgs:: Setting both to zero.'
        # Warning message ends
        warnings.warn(msg, UserWarning)
        # Warning ends
    cosPA, sinPA = np.cos(pa.rad), np.sin(pa.rad)
    cosINCL = np.cos(incl.rad)
    if incl_correction and (cosINCL < cosINCL_limit):
        cosINCL = cosINCL_limit
        # Not written to the header
        msg = '\n::z0mgs:: Large inclination encountered in ' + \
            'radius calculation \n' + \
            '::z0mgs:: Input inclination: ' + str(incl.deg) + \
            ' degrees. \n' + \
            '::z0mgs:: cos(incl) is set to ' + str(cosINCL_limit)
        # Warning message ends
        warnings.warn(msg, UserWarning)
        # Warning ends
    xcm, ycm = ra.rad, dec.rad

    dp_coords = np.zeros(list(shape) + [2])
    # Original coordinate is (y, x)
    # :1 --> x, RA --> the one needed to be divided by cos(incl)
    # :0 --> y, Dec
    dp_coords[:, :, 0], dp_coords[:, :, 1] = \
        np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    # Now, value inside dp_coords is (x, y)
    # :0 --> x, RA --> the one needed to be divided by cos(incl)
    # :1 --> y, Dec
    for i in range(shape[0]):
        dp_coords[i] = Angle(w.wcs_pix2world(dp_coords[i], 1) * u.deg).rad
    dp_coords[:, :, 0] = 0.5 * (dp_coords[:, :, 0] - xcm) * \
        (np.cos(dp_coords[:, :, 1]) + np.cos(ycm))
    dp_coords[:, :, 1] -= ycm
    # Now, dp_coords is (dx, dy) in the original coordinate
    # cosPA*dy-sinPA*dx is new y
    # cosPA*dx+sinPA*dy is new x
    radius = np.sqrt((cosPA * dp_coords[:, :, 1] +
                      sinPA * dp_coords[:, :, 0])**2 +
                     ((cosPA * dp_coords[:, :, 0] -
                       sinPA * dp_coords[:, :, 1]) / cosINCL)**2)
    radius = Angle(radius * u.rad).arcsec
    return radius
    
fits_map = fits.open('data_image/NGC3351_CO21_mom0_broad_nyq.fits')
wcs = WCS(fits_map[0])
dshape = fits_map[0].data.shape
angles = Angle(['160.99064d','11.70367d','188.4d','45.1d'])  #ra, dec, pa, incl

radii = radius_arcsec(dshape, wcs, angles[0], angles[1], angles[2], angles[3])
np.save('ngc3351_radius_arcsec_new.npy',radii)
plt.imshow(radii, origin='lower')
plt.colorbar()
plt.show()
