# Takes Rongpu's file and makes it slightly more "digestable" for our theory+likelihood code
import numpy as np

# areas used when weighting regions
fsky_north  = 0.113
fsky_decals = 0.204
fsky_des    = 0.123
fsky_south  = fsky_decals+fsky_des
# ACT overlap with decals and des
fsky_act_decals = 0.092
fsky_act_des    = 0.106

# generic header
header  = 'Angular number densities per z bin ("number per bin")\n'
header += 'i.e., the number of galaxies per sq.deg. that have zmin<z<zmax.\n'
header += 'Both imaging and spectroscopic weights are included.\n\n'
header += 'Columns are: redshift, dNdz'

# file location
version = '0.4'
bdir    = f'/global/cfs/cdirs/desi/users/rongpu/data/lrg_xcorr/dndz/iron_v{version}/'

# north, south and full sample
fname   = f'main_lrg_pz_dndz_iron_v{version}_dz_0.01.txt'
xxx = np.genfromtxt(bdir+fname)
z = (xxx[:,0]+xxx[:,1])/2
north = np.array([z,xxx[:,8],xxx[:,9],xxx[:,10],xxx[:,11]]).T
south = np.array([z,xxx[:,13],xxx[:,14],xxx[:,15],xxx[:,16]]).T
full  = (north*fsky_north+south*fsky_south)/(fsky_north+fsky_south)
for i in range(4): np.savetxt(f'dNdzs/LRGz{i+1}_dNdz_north.txt',north[:,[0,i+1]],header=header)
for i in range(4): np.savetxt(f'dNdzs/LRGz{i+1}_dNdz_south.txt',south[:,[0,i+1]],header=header)
for i in range(4): np.savetxt(f'dNdzs/LRGz{i+1}_dNdz.txt'      ,full[:,[0,i+1]] ,header=header)

# DECaLS and DES regions individually
fname   = f'main_lrg_pz_ngal_decals_des_iron_v{version}_dz_0.02.txt'
xxx = np.genfromtxt(bdir+fname)
z = (xxx[:,0]+xxx[:,1])/2
decals = np.array([z,xxx[:,3],xxx[:,4],xxx[:,5],xxx[:,6]]).T
des = np.array([z,xxx[:,8],xxx[:,9],xxx[:,10],xxx[:,11]]).T
for i in range(4): np.savetxt(f'dNdzs/LRGz{i+1}_dNdz_decals.txt',decals[:,[0,i+1]],header=header)
for i in range(4): np.savetxt(f'dNdzs/LRGz{i+1}_dNdz_des.txt',des[:,[0,i+1]],header=header)

# ACT region
act = (decals*fsky_act_decals+des*fsky_act_des)/(fsky_act_decals+fsky_act_des)
for i in range(4): np.savetxt(f'dNdzs/LRGz{i+1}_dNdz_act.txt',act[:,[0,i+1]],header=header)