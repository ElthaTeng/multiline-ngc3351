# RADEX flux / ratio analysis with multiple molecular lines

*radex_pipeline.py* combines the required steps for running RADEX analysis and builds up flux and ratio models based on the RADEX results. It considers six line transitions of three molecules (i.e. CO(1-0), CO(2-1), 13CO(2-1), 13CO(3-2), C18O(2-1), and C18O(3-2)) and five free parameters: CO column density (N_co), kinetic temperature (T_k), H2 density (n_H2), and the CO-to-13CO (X_12/13) and 13CO-to-C18O (X_13/18) abundance ratios. This code generates the input files for RADEX, runs RADEX on each file, and saves all the output files. Then, it extracts all the RADEX-predicted fluxes (in K*km/s) and constructs a 3D (N_co, T_k, n_H2) flux model grid for each CO line, a 4D (3D + X_12/13) grid for each 13CO line, and a 5D (4D + X_13/18) grid for each C18O line. Using these flux models, seven 5D line ratio grids are also generated and saved.  

## Required Installation

* RADEX: https://personal.sron.nl/~vdtak/radex/index.shtml
* joblib: https://joblib.readthedocs.io/en/latest/installing.html


## Prerequisites

* Set up molecules and model names at the beginning part of the script
  * e.g. molecule_0 = 'co', model = '5d_coarse'

* Create RADEX input and output directories for *each* molecule
  * directory name format: 'input_' + model name + '_' + molecule name
  * e.g. input_5d_coarse_co, output_5d_coarse_13co 

* Set up input parameters
  * linewidth: *(int/float)* representative line width (FWHM) of the molecular line, default = 15 (km/s) 
  * Nco: *(array)* sampling values for N_co in log scale, e.g. np.arange(16.,21.1,0.2)
  * Tkin: *(array)* sampling values for T_k in log scale, e.g. np.arange(1.,2.4,0.1)
  * nH2: *(array)* sampling values for n_H2 in log scale, e.g. np.arange(2.,5.1,0.2)
  * X_13co: *(array)* sampling values for X_12/13, e.g. np.arange(10,205,10)
  * X_c18o: *(array)* sampling values for X_13/18, e.g. np.arange(2,21,1.5) 
  * sou_model: *(str)* directory name for output model grids, e.g. 'radex_model/'
  * num_cores: *(int)* number of threads to use for multi-processing, default = 20
  * round_dens: *(int)* number of decimal places to round to for the base values of Nco and nH2 shown in the RADEX input/output file names, default = 1 
  * round_temp: *(int)* number of decimal places to round to for the Tkin values shown in the RADEX input/output file names, default = 1  

## Outputs
  * RADEX input files: saved to the prepared input directories as e.g. '1.5_1.0e3_2.5e17_50_10.inp'; values are in the order of T_k, n_H2, N_co, X_12/13, and X_13/18
  * RADEX output files: saved to the prepared output directories as e.g. '1.5_1.0e3_2.5e17_50_10.out'; same naming format as the RADEX input files
  * flux models: six numpy grids saved to the sou_model directory as e.g. 'flux_5d_coarse_co10.npy' 
  * ratio models: seven numpy grids saved to the sou_model directory as e.g. 'ratio_5d_coarse_13co32_21.npy'


## Notes

 1. N_co, T_k, and n_H2 should be sampled in log scale and with consistent step size in each parameter.
 2. The step size for Nco and nH2 should be the same.
 3. This code assumes a constant line width, since the variation of line widths do not have much effect on RADEX results.


## Paper

If you use the script(s) in your study, please cite Teng et al., "Molecular Gas Properties and CO-to-H2 Conversion Factors in the Central Kiloparsec of NGC 3351", 2022, *The Astrophysical Journal (ApJ)*, 925, 72.

[[homepage]](https://elthateng.github.io/projects/#galaxy-center)&nbsp;
[[paper]](https://iopscience.iop.org/article/10.3847/1538-4357/ac382f)&nbsp;
[[video]](https://youtu.be/aZUoEkiZJK4)&nbsp;
[[poster]](https://aas237-aas.ipostersessions.com/?s=BD-90-5D-5F-84-83-BA-95-9C-63-F3-E8-D6-DF-ED-62)&nbsp;
[[slides]](https://elthateng.github.io/files/cass_talk.pdf)&nbsp;
