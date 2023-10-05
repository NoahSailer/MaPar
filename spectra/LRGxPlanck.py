from calc_cl import *
sys.path.append('../mc_correction/')
from do_mc_corr import apply_mc_corr

if len(sys.argv) != 2:
    print('Error: usage is "python LRGxPlanck.py #", where # = 3 or 4')
    sys.exit()

# Planck release version (either 3 or 4)
ver = int(sys.argv[1])

# load the maps, masks and CMB lensing noise curve
bdir      = '/pscratch/sd/m/mwhite/DESI/MaPar/maps/'
lrg_maps  = [hp.read_map(bdir+f'lrg_s0{isamp}_del.hpx2048.fits') for isamp in range(1,5)]
lrg_masks = [hp.read_map(bdir+f'lrg_s0{isamp}_msk.hpx2048.fits') for isamp in range(1,5)]
kap_map   = [hp.read_map(f'../maps/PR{ver}_lens_kap_filt.hpx2048.fits')]
kap_mask  = [hp.read_map(f'../maps/masks/PR{ver}_lens_mask.fits')]
nkk       = np.loadtxt(f'../data/PR{ver}_lens_nlkk_filt.txt')
fnout     = f'LRGxPR{ver}.json'

# define ell-bins, and give our maps+maps some names
ledges   = [25+50*i for i in range(21)]
kapName  = f'PR{ver}'
galNames = ['LRGz1','LRGz2','LRGz3','LRGz4']
msks     = kap_mask + lrg_masks
maps     = kap_map  + lrg_maps
# compute power spectra and window functions, save to json file
full_master(ledges,maps,msks,kapName+galNames,fnout)
# correct for the lensing normalization using
# the MC calculations (labeled by prefix), overwrite old json file
prefixs = [f'LRG_full_z1_PR{ver}']*4
apply_mc_corr(fnout,fnout,kapName,galNames,prefixs)
# Use polynomial fits to measured Ckg, Cgg for 
# the covariance. Update the ckk theory curve.
with open(fnout) as outfile:
    outdata = json.load(outfile) 
cij  = np.array(outdata['cij'])
ells = np.arange(cij.shape[-1])
cij[0,0,:] = np.interp(ells,nkk[:,0],nkk[:,2],right=0)
# now compute the covariance (only for the pairs of interest
# which correspond to ckgi, cgigi for i = 1,2,3,4)
pairs = [[0,1],[0,2],[0,3],[0,4],[1,1],[2,2],[3,3],[4,4]]
full_master(ledges,maps,msks,names,fnout,cij=cij,do_cov=True,pairs=pairs)