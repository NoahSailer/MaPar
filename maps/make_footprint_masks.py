import numpy as np
import healpy as hp
import urllib.request
import os

def make_footprint_masks(NSIDE_OUT=2048, COORD_OUT='c', outdir='masks', 
                         deccuts=[-15], verbose_sffx=False):
    """
    Makes the following (binary) masks
        ngc_mask.fits          | NGC is defined as DEC >  0 
        sgc_mask.fits          | SGC is defined as DEC <= 0
        north_mask.fits        | (DEC > 32.375) and NGC
        des_mask.fits          | DES is defined as everywhere with positive "detection 
                                 fraction" in any of the DES DR2 g,r,i,z,Y bands.
        decals_mask.fits       | (not North) and (not DES) and (DEC > -15)
        dec[P or M]#_mask.fits | creates a mask for each # [integer] in DECcuts that 
                                 is defined by DEC <= \PM #
    at a given NSIDE_OUT and COORD_OUT system, and saves them to outdir/.
    """
    
    if not os.path.exists(outdir): os.mkdir(outdir)
    sffx = '' if (not verbose_sffx) else f'_cord.{COORD_OUT}_nside.{NSIDE_OUT}'

    # DEC and RA in galactic coords
    npix      = 12*NSIDE_OUT**2
    theta,phi = hp.pix2ang(NSIDE_OUT,np.arange(npix))
    DEC,RA    = 90-np.degrees(theta),np.degrees(phi)

    ## Make NGC/SGC masks in galactic coords
    ## and rotate to COORD_OUT coords
    ngc_mask = np.ones(npix)
    ngc_mask[np.where(DEC<=0.)] = 0.
    sgc_mask = np.ones(npix)
    sgc_mask[np.where(DEC>0.)]  = 0.
    rot = hp.rotator.Rotator(coord=f'g{COORD_OUT}')
    ngc_mask = np.round(rot.rotate_map_pixel(ngc_mask))
    sgc_mask = np.round(rot.rotate_map_pixel(sgc_mask))
    hp.write_map(f'{outdir}/ngc_mask{sffx}.fits',ngc_mask,overwrite=True,dtype=np.int32)
    hp.write_map(f'{outdir}/sgc_mask{sffx}.fits',sgc_mask,overwrite=True,dtype=np.int32)

    ## Make "North" mask in COORD_OUT coords
    north_mask = ngc_mask.copy()
    north_mask[np.where(DEC<=32.375)] = 0.
    hp.write_map(f'{outdir}/north_mask{sffx}.fits',north_mask,overwrite=True,dtype=np.int32)

    ## Make DES mask in COORD_OUT coords
    bands   = ['g','r','i','z','Y']
    website = 'https://desdr-server.ncsa.illinois.edu/despublic/dr2_tiles/Coverage_DR2/'
    fnames  = [f'dr2_hpix_4096_frac_detection_{b}.fits.fz' for b in bands]
    # fetch detection fraction files from the web (HEALPix maps in celestial coords)
    # and temporarily save them in the current directory  
    for fn in fnames: urllib.request.urlretrieve(website+fn, fn)
    maps = [hp.read_map(fn) for fn in fnames]
    for i in range(len(maps)): maps[i][np.where(maps[i]<=0)]=0.
    des_mask = hp.ud_grade(np.sum(maps,axis=0),NSIDE_OUT)
    des_mask[np.where(des_mask>0.)]=1.
    # rotate to COORD_OUT coordinates
    rot = hp.rotator.Rotator(coord=f'c{COORD_OUT}')
    des_mask = np.round(rot.rotate_map_pixel(des_mask))
    hp.write_map(f'{outdir}/des_mask{sffx}.fits',des_mask,overwrite=True,dtype=np.int32)
    # delete detection fraction files
    for fn in fnames: os.remove(fn)

    ## Make DECaLS mask in COORD_OUT coords
    decals_mask = np.ones(npix) - north_mask - des_mask
    decals_mask[np.where(DEC<=-15)] = 0.
    hp.write_map(f'{outdir}/decals_mask{sffx}.fits',decals_mask,overwrite=True,dtype=np.int32)
    
    ## Make DEC <= DECcuts masks in COORD_OUT coords
    for cut in np.array(deccuts,dtype=np.int32):
        mask = np.ones(npix)
        mask[np.where(DEC>cut)] = 0.
        sgn = 'p' if cut>=0 else 'm'
        hp.write_map(f'{outdir}/DEC{sgn}{np.abs(cut)}_mask{sffx}.fits',mask,overwrite=True,dtype=np.int32)

if __name__ == "__main__":
    make_footprint_masks()