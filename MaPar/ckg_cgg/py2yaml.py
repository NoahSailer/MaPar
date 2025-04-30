import os,yaml
import numpy as np
from copy import deepcopy
from .ckg_cgg import likelihood_defaults,default_bias_priors,smag_prior

output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"bgs+lrg_cross+pr4+dr6.yaml")
indict  = yaml.safe_load(open(output_file,'r'))
model    = indict['model']
galNames = likelihood_defaults[model]['galNames']
nuisance = likelihood_defaults[model]['nuisance']
yamldict = {}
for i,suf in enumerate(galNames):
    for pref in nuisance:
        if pref != 'smag':
            yamldict[pref+'_'+suf] = deepcopy(default_bias_priors[pref])
        else:
            smag_mean = likelihood_defaults[model]['fidsmag'][i]
            yamldict[pref+'_'+suf] = {'prior':smag_prior(smag_mean),'ref':smag_prior(smag_mean)}
outdict = {**indict, **{'params': yamldict}}
if outdict != indict: 
    yaml.dump(outdict, open(output_file, 'w'),sort_keys=False)
