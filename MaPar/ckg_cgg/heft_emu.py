import numpy as np
from aemulus_heft.heft_emu import NNHEFTEmulator

nnemu = NNHEFTEmulator()

def pmmHEFT(thy_args,z):
   """
   Assumes thy_args[:5] = [omb,omc,ns,ln10As,H0,Mnu] and z = ndarray
   
   Returns res = (Nk,1+Nz) table where the first column is k and the
   remaining Nz columns are the matter power spectrum evaluated
   at each z.
   """
   omb,omc,ns,ln10As,H0,Mnu = thy_args[:6]
   Mnu          = max(Mnu,0.01) # HEFT is only valid for 0.01 < Mnu < 0.5 
   cosmo        = np.zeros((len(z),8))
   cosmo[:,-1]  = z 
   cosmo[:,:-1] = np.array([omb, omc, -1., ns, np.exp(ln10As)/10., H0, Mnu])
   k_nn, spec_heft_nn = nnemu.predict(cosmo)
   res       = np.zeros((len(k_nn),len(z)+1))
   res[:,0]  = k_nn
   res[:,1:] = np.swapaxes(spec_heft_nn[:,0,:],0,1)
   return res

def ptableHEFT(thy_args,z):
   """
   Assumes thy_args[:5] = [omb,omc,ns,ln10As,H0,Mnu] and z = float
   
   Returns monomial table = (Nk,1+Nmono) ndarray. The first column is k, 
   while the order of the 15 monomials is:
   
   1-1, 1-cb, cb-cb, delta-1, delta-cb, delta-delta, delta2-1, delta2-cb, 
   delta2-delta, delta2-delta2, s2-1, s2-cb, s2-delta, s2-delta2, s2-s2.
   """
   omb,omc,ns,ln10As,H0,Mnu = thy_args[:6]
   Mnu   = max(Mnu,0.01) # HEFT is only valid for 0.01 < Mnu < 0.5 
   cosmo = np.atleast_2d([omb, omc, -1., ns, np.exp(ln10As)/10., H0, Mnu, z])
   k_nn, spec_heft_nn = nnemu.predict(cosmo)
   Nmono     = spec_heft_nn.shape[1]
   res       = np.zeros((len(k_nn),Nmono+1))
   res[:,0]  = k_nn
   res[:,1:] = np.swapaxes(spec_heft_nn[0,:,:],0,1)
   return res
   
def pgmHEFT(thy_args,z):
   """
   Assumes thy_args = [omb,omc,ns,ln10As,H0,Mnu,b1,b2,bs] and z = float
   
   Returns res = (Nk,3) ndarray, where the first column is k, the second column
   is the "bias contribution" (i.e. terms that cannot be analytically 
   marginalized over), while the third column is -0.5*k^2 P_{cb, 1}
   
   The full prediction is res[:,1] + alpha_x * res[:,2]
   """
   omb,omc,ns,ln10As,H0,Mnu,b1,b2,bs = thy_args
   bterms_gm = np.array([0, 1, 0, b1, 0, 0, 0.5*b2, 0, 0, 0, bs, 0, 0, 0, 0])
   T         = ptableHEFT(thy_args,z)
   res       = np.zeros((T.shape[0],3))
   res[:,0]  = T[:,0]
   res[:,1]  = np.dot(T[:,1:],bterms_gm) # bias-contribution
   res[:,2]  = -0.5 * T[:,0]**2 * T[:,1] # counterterm
   return res  
  
def pggHEFT(thy_args,z):
   """
   Assumes thy_args = [omb,omc,ns,ln10As,H0,Mnu,b1,b2,bs] and z = float
   
   Returns res = (Nk,3) ndarray, where the first column is k, the second column
   is the "bias contribution" (i.e. terms that cannot be analytically 
   marginalized over), while the third column is -0.5*k^2 P_{cb, cb}
   
   The full prediction is res[:,1] + alpha_a * res[:,2]
   """
   omb,omc,ns,ln10As,H0,Mnu,b1,b2,bs = thy_args
   bterms_gg = np.array([0, 0, 1, 0, 2*b1, b1**2, 0, b2, b2*b1, 0.25*b2**2, 0, 2*bs, 2*bs*b1, bs*b2, bs**2])
   T         = ptableHEFT(thy_args,z)
   res       = np.zeros((T.shape[0],3))
   res[:,0]  = T[:,0]
   res[:,1]  = np.dot(T[:,1:],bterms_gg) # bias-contribution
   res[:,2]  = -0.5 * T[:,0]**2 * T[:,2] # counterterm
   return res  