import numpy as np
import os
import yaml
import pickle
import matplotlib.pyplot as plt
from scipy import stats

output_dir = '/ec/res4/hpcperm/ecme3038/ecearth/ecearth4/ECtuner/mari/results/tuned_2d_LR/sobol_params/'
os.makedirs(output_dir, exist_ok=True)

# 2. parameters
param_structure = {
    'namcumf': {
        'RPRCON': 0.14E-02,
        'ENTRORG': 0.175E-02,
        'DETRPEN': 0.75E-04,
        'ENTRDD': 0.3E-03,
        'RMFDEPS': 0.3
    },
    'namcldp': {
        'RVICE': 0.13,
        'RLCRITSNOW': 2.0E-05,
        'RSNOWLIN2': 0.03,
        'RCLDIFF': 0.6E-05,
        'RCLDIFF_CONVI': 10.0,
        'RDEPLIQREFRATE': 0.5,
        'RDEPLIQREFDEPTH': 500.0,
        'RCL_OVERLAPLIQICE': 0.65,
        'RCL_INHOMOGAUT': 1.5,
        'RCL_INHOMOGACC': 3.0
    },
    'naerad': {
        'RMINICE': 60.0
    }
}

# Sflat structure: [(namelist, param_name, default_value), ...]
flat_param_meta = []
for namelist, params in param_structure.items():
    for p_name, p_val in params.items():
        flat_param_meta.append((namelist, p_name, p_val))

D = len(flat_param_meta)  # 16 dimensions (parametri)
m = 7                    # 2^7 = 128 simulation
max_pert = 0.50          # max perturb 50 %

# 3. sobol space generation
sob = stats.qmc.Sobol(d=D, scramble=True)
gino = sob.random_base2(m=m) 

# to save (from fede)
pickle.dump([sob, gino], open(os.path.join(output_dir, 'sobolset_16d.p'), 'wb'))
print(f"--- Sobol space generated and pickled: 128 combinations, 16 parameters ---")

# 4. diagnostic plot (first 9 dimensions)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')
ax.scatter(gino[:, 0], gino[:, 1], gino[:, 2], color='blue', alpha=0.6, label='Dims 0,1,2')
ax.scatter(gino[:, 3], gino[:, 4], gino[:, 5], color='red', alpha=0.6, label='Dims 3,4,5')
ax.scatter(gino[:, 6], gino[:, 7], gino[:, 8], color='green', alpha=0.6, label='Dims 6,7,8')
ax.set_title('Sobol Space Filling Diagnostic (First 9 Dimensions)')
ax.legend()
fig.savefig(os.path.join(output_dir, 'sobol_diagnostic_3d.pdf'))
plt.close()

# 5. file generation (Aggiornato con la struttura corretta di EC-Earth)
for co in range(len(gino)):
    # Costruiamo l'albero nidificato identico al tuo esempio
    # Nota la struttura a lista [] dentro cui inseriamo il dizionario con 'base.context'
    yaml_structure = [
        {
            'base.context': {
                'model_config': {
                    'oifs': {
                        'tuning': {nl: {} for nl in param_structure.keys()}
                    }
                }
            }
        }
    ]
    
    # Estraiamo una referenza comoda per popolare i parametri senza scrivere una riga infinita
    tuning_pointer = yaml_structure[0]['base.context']['model_config']['oifs']['tuning']
    
    for indpa, (namelist, p_name, default_val) in enumerate(flat_param_meta):
        gik = gino[co, indpa]  # quasi-random value
        
        # real range perturbation: [-max_pert, +max_pert]
        pert = (gik * 2.0 - 1.0) * max_pert
        
        # new perturbed value
        nuval = default_val * (1.0 + pert)
        
        # Popoliamo la namelist corretta dentro la struttura profonda
        tuning_pointer[namelist][p_name] = float(f"{nuval:.6e}")
        
    file_out_path = os.path.join(output_dir, f"sobol_run_{co:02d}.yml")
    with open(file_out_path, 'w') as yf:
        # Usiamo default_flow_style=False per avere la struttura incolonnata pulita
        yaml.dump(yaml_structure, yf, default_flow_style=False)

print(f"--- Success: 128 structured YAML files saved in: {output_dir} ---")