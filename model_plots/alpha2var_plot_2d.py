import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.lines as mlines
from matplotlib.patches import Patch

'''This script plots the density contours of the random-drawed points returned from random_draw_correlation.py'''

input = 'tau10'  # tau10 or Tk
plot_type = 'density'  # scatter or density
two_category = True
model = '6d_coarse'

if input == 'tau10':
    var = np.log10(np.load('radex_model/random_draw_correlation_'+model+'_'+input+'_center.npy')).reshape(-1)
else:
    var =np.load('radex_model/random_draw_correlation_'+model+'_'+input+'_center.npy').reshape(-1)
alpha = np.log10(np.load('radex_model/random_draw_correlation_'+model+'_'+input+'_alpha_center.npy')).reshape(-1)
idx_eff = ~np.isnan(var)*(~np.isnan(alpha))
var = var[idx_eff]
alpha = alpha[idx_eff]

if plot_type == 'density':
    xmin = np.nanmin(var)
    xmax = np.nanmax(var)
    ymin = np.nanmin(alpha)
    ymax = np.nanmax(alpha)

    # Perform a kernel density estimate on the data
    if input == 'Tk':
        X, Y = np.mgrid[xmin:xmax:14j, ymin:ymax:20j]
    else:
        X, Y = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([var, alpha])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

if two_category:
    if input == 'tau10':
        var_2 = np.log10(np.load('radex_model/random_draw_correlation_'+model+'_'+input+'_arms.npy')).reshape(-1)
    else:
        var_2 = np.load('radex_model/random_draw_correlation_'+model+'_'+input+'_arms.npy').reshape(-1)
    alpha_2 = np.log10(np.load('radex_model/random_draw_correlation_'+model+'_'+input+'_alpha_arms.npy'))
    alpha_2[104, :] = np.full((1000,), np.nan)
    alpha_2 = alpha_2.reshape(-1)
    idx_eff_2 = ~np.isnan(var_2)*(~np.isnan(alpha_2))*(~np.isinf(alpha_2))
    var_2 = var_2[idx_eff_2]
    alpha_2 = alpha_2[idx_eff_2]

    if plot_type == 'density':
        xmin_2 = np.nanmin(var_2)
        xmax_2 = np.nanmax(var_2)
        ymin_2 = np.nanmin(alpha_2)
        ymax_2 = np.nanmax(alpha_2)

        # Perform a kernel density estimate on the data
        if input == 'Tk':
            X_2, Y_2 = np.mgrid[xmin_2:xmax_2:14j, ymin_2:ymax_2:40j]
        else:
            X_2, Y_2 = np.mgrid[xmin_2:xmax_2:100j, ymin_2:ymax_2:100j]
        positions_2 = np.vstack([X_2.ravel(), Y_2.ravel()])
        values_2 = np.vstack([var_2, alpha_2])
        kernel_2 = stats.gaussian_kde(values_2)
        Z_2 = np.reshape(kernel_2(positions_2).T, X_2.shape)

fig, ax = plt.subplots()
if two_category:
    if plot_type == 'density':
        if input == 'tau10':
            ax.contour(X_2, Y_2, Z_2, colors='darkred', label='Arms', levels=[0.1,0.2,0.3,0.4,0.5], alpha=0.5)
            ax.contourf(X, Y, Z, cmap='Blues', label='Center', levels=[0.2,0.4,0.6,0.8,0.9,1.])
            
            ax.set_xlim(-3, 2)
        else:
            ax.contour(X_2, Y_2, Z_2, colors='darkred', label='Arms', levels=[0.1,0.2,0.3,0.4,0.5,0.6,0.7], alpha=0.5)
            ax.contourf(X, Y, Z, cmap='Blues', label='Center', levels=[0.2,0.4,0.6,0.8,1.0,1.2,1.4])           
            ax.set_xlim(1., 2.3)
            # ax.imshow(np.rot90(Z), extent=[xmin, xmax, ymin, ymax])
        
        ax.set_ylim(-2, 1.5)    
        center = Patch(facecolor='tab:blue', label='Center')
        arms = mlines.Line2D([], [], color='darkred', label='Arms')

        lprop = {'weight':'bold', 'size':'large'}
        plt.legend(handles=[center, arms],
                labels=['Center', 'Arms'],
                ncol=1, handletextpad=0.5, handlelength=1.0, columnspacing=0.5, prop=lprop,
                loc='lower right', fontsize=18)
    else:
        ax.plot(var_2, alpha_2, c='darkblue', marker='.', linestyle='', label='Arms')
        ax.plot(var, alpha, c='darkred', marker='.', linestyle='', label='Center')  
        ax.legend(fontsize=12)  #, loc='lower right'  
        ax.set_ylim(-2.5, 1.5)
        if input == 'tau10':
            ax.set_xlim(-2.5, 1.5)

    ax.axhline(0.65, c='k', linestyle='--')

else:
    ax.plot(var, alpha, c='k', marker='.', linestyle='')
    ax.axhline(0.65, c='darkred', linestyle='--')
    ax.set_ylim(-2.5, 1.)
    ax.set_xlim(-2.5, 1.5)

if input == 'tau10':
    ax.set_xlabel(r'$\log\ \tau_{\rm CO(1-0)}$')
elif input == 'Tk':
    ax.set_xlabel(r'$\log\ T_{\rm k}$ (K)')
ax.set_ylabel(r'$\log\ \alpha_{CO}$ ($M_\odot\ (K\ km\ s^{-1}\ pc^2)^{-1}$)')

plt.tight_layout()
plt.show()

