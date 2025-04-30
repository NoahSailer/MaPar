# Maps to Parameters

[![](https://img.shields.io/badge/arXiv-2407.04606%20-red.svg)](https://arxiv.org/abs/2407.04606)
[![](https://img.shields.io/badge/arXiv-2407.04607%20-red.svg)](https://arxiv.org/abs/2407.04607)

Codebase for the cross-correlation analysis of DESI Luminous Red Galaxies with CMB lensing from *Planck* (PR3 and PR4) and ACT DR6. See `yamls/fiducial_noBAO_pr4-dr6.yaml` for the fiducial setup.

Chains, power spectra, window functions, covariance matrices and figure data are available on [zenodo](https://zenodo.org/records/12613408).

## Usage in Cobaya

Install the likelihood and associated data products with:
```
conda activate cobaya
python3 -m pip install -v git+https://github.com/NoahSailer/MaPar --user
cobaya-install MaPar.ckg_cgg.BGSLRG_PR4DR6 -p path/to/cobaya/packages
```
and optionally install the HEFT emulator with:
```
python3 -m pip install -v git+https://github.com/AemulusProject/aemulus_heft
```

**Currently the likelihood is only compatible with `CLASS`**. Example usage:

```
theory:
  classy: 

likelihood:
  MaPar.ckg_cgg.BGSLRG_PR4DR6:
    # OPTIONAL
    # listing defaults below
    model: linear # linear or heft
    kapNames: [PR4,DR6]
    galNames: [LRGz2,LRGz3,LRGz4]
    amin: [79,79,79]
    amax: [178,243,243]
    xmin: [[20,20,20],[44,44,44]]
    xmax: [[178,243,243],[178,243,243]]
    fidSN: [2.25e-6, 2.05e-6, 2.25e-6]
    fidsmag: [1.,1.,1.]
    fida0:  [0.,0.,0.]
    a0prior: [50.,50.,50.,50.]
    fidaX:  [3.,3.,3.]
    aXprior: [3.,3.,3.]
    nuisance: [b1,smag]
    chenprior: False
```
