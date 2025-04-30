import numpy as np
import sys, os, json
from cobaya.likelihoods.base_classes import InstallableLikelihood
from .limber    import limb 
from .gaussLike import gaussLike
from .pack_data import pack_cl_wl,pack_cov,pack_dndz

default_url = "https://zenodo.org/records/12613408/files/"
default_file_root = "DESI-LRG_x_Planck-PR4_ACT-DR6"

likelihood_defaults = {
'linear': {
    'kapNames': ['PR4','DR6'],
    'galNames': ['LRGz2','LRGz3','LRGz4'],
    'amin': [79,79,79],
    'amax': [178,243,243],
    'xmin': [[20,20,20],[44,44,44]],
    'xmax': [[178,243,243],[178,243,243]],
    'fidSN': [2.25e-6, 2.05e-6, 2.25e-6],
    'fidsmag': [1.,1.,1.],
    'fida0':  [0.,0.,0.],
    'a0prior': [50.,50.,50.,50.],
    'fidaX':  [3.,3.,3.],
    'aXprior': [3.,3.,3.],
    'nuisance': ['b1','smag'],
    'chenprior': False,
    }
'heft': {
    'kapNames': ['PR4','DR6'],
    'galNames': ['LRGz1','LRGz2','LRGz3','LRGz4'],
    'amin': [79,79,79,79],
    'amax': [600,600,600,600],
    'xmin': [[20,20,20,20],[44,44,44,44]],
    'xmax': [[600,600,600,600],[600,600,600,600]],
    'fidSN': [4.07e-6, 2.25e-6, 2.05e-6, 2.25e-6],
    'fidsmag': [1.,1.,1.,1.],
    'fida0':  [0.,0.,0.,0.],
    'a0prior': [50.,50.,50.,50.],
    'fidaX':  [0.,0.,0.,0.],
    'aXprior': [2.,2.,2.,2.],
    'nuisance': ['b1','b2','bs','smag']
    'chenprior': True,
    }
}

default_bias_priors = {
    'b1': {'prior': {'min': -1., 'max': 3.},
           'ref'  : {'dist': 'norm', 'loc': 1., 'scale': 0.05}},
    'b2': {'prior': {'min': -5., 'max': 5.},
           'ref'  : {'dist': 'norm', 'loc': 0., 'scale': 0.1}},
    'bs': {'prior': {'dist': 'norm', 'loc': 0., 'scale': 1.},
           'ref'  : {'dist': 'norm', 'loc': 0., 'scale': 0.1}}, 
}
smag_prior = lambda smag: {'dist': 'norm', 'loc': smag, 'scale': 0.1}

kval = np.logspace(np.log10(0.005),np.log10(5.),200) #h/Mpc

class ckg_cgg(InstallableLikelihood):
    """
    likelihood
    """
    install_options: dict = {
        "download_url": f"{default_url}zenodo_LRGxPR4-DR6.tar",
        "data_path": default_file_root
    }
    def initialize(self):
        # 
        if not (self.model in likelihood_defaults.keys()):
            raise RuntimeError(f'model must be one of {likelihood_defaults.keys()}')
        if self.data_base_path is None:
            if self.packages_path:
                self.data_base_path = self.get_path(self.packages_path)
            else:
                self.data_base_path = os.path.abspath(os.path.dirname(__file__))
        print("base path", self.data_base_path,flush=True)
        # 
        def background(thy_args,zs):
            pp  = self.provider
            OmM = np.float(pp.get_Omega_b(0)+pp.get_Omega_cdm(0)+pp.get_Omega_nu_massive(0))
            thermo  = pp.get_CLASS_thermodynamics()
            zstar   = thermo['z'][np.argmin(np.abs(np.array(thermo['x_e']) - 0.5))]
            chistar = np.vectorize(pp.get_comoving_radial_distance)(zstar)*h
            Ez  = np.vectorize(pp.get_Hubble)(zs)
            Ez  = Ez/Ez[0]
            chi = np.vectorize(pp.get_comoving_radial_distance)(zs)*h
            return OmM,chistar,Ez,chi
        def Pk(zz):
            pp    = self.provider
            h     = np.float(pp.get_Hubble(0, units="km/s/Mpc")/100)
            Pfunc = pp.get_Pk_interpolator()
            return np.array([np.float(Pfunc.P(zz,kk*h,grid=False)*h**3) for kk in kval])
        # 
        self.settings = None
        if self.model == 'linear': # thy_args = [b1]
            def pgm(thy_args,z): return np.array([kval,thy_args[0]*Pk(z)]).T
            def pgg(thy_args,z): return np.array([kval,thy_args[0]**2*Pk(z)]).T
            def pmm(thy_args,z):
                if isinstance(z, (np.floating, float)): return np.array([k,Pk(z)]).T
                return np.array([kval]+[Pk(zz) for zz in z]).T
            self.settings = likelihood_defaults['linear']
        if self.model == 'heft':  # thy_args = [omb,omc,ns,ln10As,H0,Mnu,b1,b2,bs]
            from .heft_emu import *
            pgm, pgg, pmm = pgmHEFT, pggHEFT, pmmHEFT
            self.settings = likelihood_defaults['heft']
        
        if self.custom: 
            self.settings = {
            'kapNames': self.kapNames,
            'galNames': self.galNames,
            'amin': self.amin,
            'amax': self.amax,
            'xmin': self.xmin,
            'xmax': self.xmax,
            'fidSN': self.fidSN,
            'fida0':  self.fida0,
            'a0prior': self.a0prior,
            'fidaX':  self.fidaX,
            'aXprior': self.aXprior,
            'nuisance': self.nuisance,
            'chenprior': self.chenprior,
            }        
        self.nsamp = len(self.settings['galNames']) 
        self.nkap  = len(self.settings['kapNames'])
        self.load_data()
        # set up theory class
        self.clPred = limb(self.dndz, pgm, pgg, pmm, background, zmin=0.001, zmax=1.8, Nz=80)
        # set up priors for analytically marginalized parameters and Gaussian likelihood
        def template_priors(isamp):
            a0 = self.fida0[isamp] ; a0p = self.a0prior[isamp]
            SN = self.fidSN[isamp] ; SNp = self.snfrac*SN
            aX = self.fidaX[isamp] ; aXp = self.aXprior[isamp]
            return [[a0,a0p],[SN,SNp],[aX,aXp]]
        tmp_priors = template_priors(0)
        for i in range(1,self.nsamp): tmp_priors += template_priors(i)
        self.tmp_priors = tmp_priors
        print('Using template priors =',tmp_priors,flush=True)
        self.glk = gaussLike(self.data, self.cov, tmp_priors=np.array(tmp_priors), jeffreys=self.jeffreys)
        
    # \checkmark
    def load_data(self):
        """Load the data."""
        jsonpath = os.path.join(self.data_base_path, "data/lrg_cross_pr4+dr6.json"))
        with open(jsonpath) as outfile: jsondata = json.load(outfile)
        keys = ['kapNames','galNames','amin','amax','xmin','xmax']
        kapNames,galNames,amin,amax,xmin,xmax = [self.settings[key] for key in keys]
        self.wla,self.wlx,self.data = pack_cl_wl(jsondata,kapNames,galNames,amin,amax,xmin,xmax)
        self.cov                    = pack_cov(  jsondata,kapNames,galNames,amin,amax,xmin,xmax)
        dndzs     = [np.loadtxt(os.path.join(self.data_base_path, f"data/dNdz/{name}_dNdz.txt"))) for name in galNames]
        self.dndz = pack_dndz(dndzs)
        self.pixwin = np.array(jsondata['pixwin'])

    # \checkmark
    def get_requirements(self):
        """What we require."""
        reqs = {'CLASS_thermodynamics': None}
        if self.model == 'linear':
            reqs['Pk_interpolator'] = {'z':self.clPred.z, 
                                       'k_max':10., 
                                       'nonlinear':True,
                                       'vars_pairs':([['delta_tot', 'delta_tot']])}
        if self.model == 'heft':
            reqs['omega_b']   = None
            reqs['omega_cdm'] = None
            reqs['n_s']       = None
            reqs['ln1e10As']  = None
            reqs['H0']        = None
            reqs['m_ncdm']    = None
        for suf in self.settings['galNames']:
            for pref in self.settings['nuisance']:
                reqs[pref+'_'+suf] = None
        return reqs
        
    # \checkmark
    def get_parameter_defaults(self):
        res = {}
        for i,suf in enumerate(self.settings['galNames']):
            for pref in self.settings['nuisance']:
                if pref != 'smag': 
                    res[pref+'_'+suf] = default_bias_priors[pref]
                else: 
                    smag_mean = self.settings['fidsmag'][i]
                    res[pref+'_'+suf] = {'prior':smag_prior(smag_mean),'ref':smag_prior(smag_mean)}
        return res

    # \checkmark
    def compute_full(self):
        """
        Do the full prediction (including [pixel] window functions)
        Returns a table with coefficients
        # (1, alpha_a(z1), SN(z1), alpha_x(z1), alpha_a(z2), SN(z2), alpha_x(z2), ...)
        """
        full_pred = []
        Nls       = []
        for i,suf in enumerate(self.settings['galNames']):            
            smag = self.provider.get_param(f'smag_{suf}')
            if self.model == 'linear': 
                params = np.array([self.provider.get_param(f'b1_{suf}')])
            if self.model == 'heft':
                pars   = ['omega_b','omega_cdm','n_s','ln1e10As','H0',
                          'm_ncdm',f'b1_{suf}',f'b2_{suf}',f'bs_{suf}'] 
                params = np.array([self.provider.get_param(p) for p in pars])
            # Cgg and Ckg are tables of shape (nell,4)
            # where the four columns correspond to 
            # 1, alpha_auto, shot noise, alpha_cross
            Cgg,Ckg = self.clPred.computeCggCkg(i,params,smag)
            if self.settings['chenprior']:
                Cgg[:,1] += Cgg[:,3]/(2.*(1.+b1))
                Ckg[:,1] += Ckg[:,3]/(2.*(1.+b1))
            Nkg = Ckg.shape[0] ; Ngg = Cgg.shape[0]
            # correct for pixel window function
            # shot noise is left untouched
            pixwin_idxs = [0,1,3] 
            for idx in pixwin_idxs:
                Ckg[:,idx] = Ckg[:,idx]*self.pixwin[:Nkg]
                Cgg[:,idx] = Cgg[:,idx]*self.pixwin[:Ngg]**2
            # multiply by the "mask window"
            wa     = self.wla[i][:,:Ngg]
            Cggkgs = np.dot(wa,Cgg)
            for j in range(self.nkap):
                wx     = self.wlx[j][i][:,:Nkg]
                Cggkgs = np.concatenate((Cggkgs,np.dot(wx,Ckg)))
            # stack the data vector
            Nl,Nmon = Cggkgs.shape
            res = np.zeros((Nl,1+(Nmon-1)*self.nsamp))
            res[:,0] = Cggkgs[:,0]
            res[:,1+i*(Nmon-1):1+(i+1)*(Nmon-1)] = Cggkgs[:,1:]
            full_pred.append(res)
        return np.concatenate(full_pred)
        
    # \checkmark
    def logp(self,**params_values):
        """Return the log-likelihood."""
        if self.maximize: return self.glk.maxLogLike(self.compute_full())
        return self.glk.margLogLike(self.compute_full())