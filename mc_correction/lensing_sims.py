# PR3/PR4 simulations are in galactic coords
# ACT simulations are in XXX coords

import numpy as np
import healpy as hp

def get_PR3_maps(simidx,nside):
    '''
    Returns reconstructed and true kappa map 
    from a simulation indexed by simidx:
    simidx = 0,...,299
    '''
    # get reconstructed map
    bdir = '/global/cfs/cdirs/cmb/data/planck2018/ffp10/lensing/MV/'
    fname = bdir + 'sim_klm_%03d.fits'%simidx
    kappa_sim_alm = np.nan_to_num(hp.read_alm(fname))
    kap_recon = hp.alm2map(kappa_sim_alm,nside)
    
    # get true map
    bdir = '/pscratch/sd/n/nsailer/mc_mult_corr/PR3_lensing_inputs/'
    fname = bdir + 'sky_klm_%03d.fits'%simidx
    true_map_alm = np.nan_to_num(hp.read_alm(fname))
    kap_true = hp.alm2map(true_map_alm,nside)
    
    return kap_recon,kap_true

def get_PR4_maps(simidx,nside):
    '''
    Returns reconstructed and true kappa map 
    from a simulation indexed by simidx:
    simidx = 60,...,300,360,...600
    (not sure why 301,...,359 don't exist)
    '''
    bdir = '/global/cfs/cdirs/cmb/data/'
    bdir_recon = 'planck2020/PR4_lensing/PR4_sims/'
    bdir_truth = 'generic/cmb/ffp10/mc/scalar/'
    
    # get reconstructed map
    fname = bdir+bdir_recon+'klm_sim_%04d_p.fits'%simidx
    kappa_sim_alm = np.nan_to_num(hp.read_alm(fname)) 
    kap_recon = hp.alm2map(kappa_sim_alm,nside)
    
    # get true map 
    fname = bdir+bdir_truth+'ffp10_unlensed_scl_cmb_000_tebplm_mc_%04d.fits'%(simidx + 200)
    true_map_alm,mmax = hp.read_alm(fname,hdu=4,return_mmax=True)
    pixel_idx = np.arange(len(true_map_alm))
    L = hp.sphtfunc.Alm.getlm(mmax,i=pixel_idx)[0]
    true_map_alm *= L*(L+1)/2 # phi -> kappa
    kap_true = hp.alm2map(true_map_alm,nside)
    
    return kap_recon,kap_true

def get_kappa_maps(simidx,nside,lensmap):
    if lensmap == 'PR3': return get_PR3_maps(simidx,nside)
    if lensmap == 'PR4': return get_PR4_maps(simidx,nside)