.. image:: https://zenodo.org/badge/1097476719.svg
  :target: https://doi.org/10.5281/zenodo.18624731

# hfradar

Tools for analysing High-Frequency radar data.

This package contains, in one side, a set of core functions to analyze
HF radar data and, on the other, tutorials showing the usage of these 
tools. 

## Dependecies

The `hfradar` package is build on Numpy, Scipy and Xarray packages. 
Besides, tutorials have additional dependencies on the Matplotlib 
and Cartopy packages.

## Core functions

| Function         | Description                                                                                                                       |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `hfr_noise`      | Estimate the noise level of High-Frequency radar measurements of radial velocities                                                |
| `hfr_rmse_pairs` | Compute the Root Mean Square Error between two time-series of HF radar measurements of radial velocities                          |
| `hfr_rmse_fit`   | Fit a set of observed Root-Mean-Square-Error of the radial velocities between two diferen radar stations to a theoretical model   |
| `hfr_rmse_model` | Dependence of the Root-Mean-Square-Error of the radial velocity with the observing angle difference between two HF radar stations |
| `lmercator`      | Compute the Mercator projection centered at the center of the image                                                               |

## Tutorial

In addition to the core functions, this package contains some tutorials.

| Tutorial                       | Description                        |
|--------------------------------|------------------------------------|
| `estimate_noise_l2_data.ipynb` | Create the figure of the paper [1] |

## Constributors

Isern-Fontanet, J; Quirós-Collazos, L.; Iglesias, J.; Martínez, J; 
Ballabrera-Poy, J.; González-Haro, C.; and García-Ladona, E.


## Acknowledgements

This work was supported by the European Maritime, Fisheries and 
Aquaculture Fund (EMFAF) with the institutional support of the grant 
‘Severo Ochoa Centre of Excellence’ accreditation (CEX2024-001494-S) 
funded by AEI 10.13039/501100011033. This work has also the support of 
the DEMON project, grant PID2021-123457OB-C21 funded by 
MICIU/AEI/10.13039/501100011033 and ERDF/EU; the OPALE project, grant 
Interreg POCTEFA EFA146/03; and the ESA Contract number 
RFP/3-18663/24/NL/IB/ar.

## References

[1] Isern-Fontanet, J; Quirós-Collazos, L.; Iglesias, J.; Martínez, J;
    Ballabrera-Poy, J.; Agostinho, P.; González-Haro, C.; and 
    García-Ladona, E. (2026). **Data-Driven Noise Estimation for 
    Individual High-Frequency Radar Stations**. Submitted to J. Atmos. 
    Oceanic Technol.