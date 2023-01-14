import numpy as np
import mdtraj as md
from tqdm import tqdm

from glob import glob

##########################################################################################################################################################################

## Number of simulation
n_sim = 7

## Type of simulation
type_sim = 'all'
#type_sim = 'neighbors'

## Type of saved data
data_type = 'protein'
#data_type = 'all'

# simulation parameters
n_replicas = 36
min_T = 280
max_T = 400

#########################################################################################################################################################################

# Define paths where position and forces from the simulaion are saved 
pos_path = f'/import/a12/users/jacopo/transf_temp/all_atom_replica_results_chignolin/exchange_{type_sim}/sim{n_sim}'
forces_path = f'/import/a12/users/jacopo/transf_temp/all_atom_replica_results_chignolin/exchange_{type_sim}/sim{n_sim}'

# Define paths where to save results
processed_pos = f'/local_scratch2/jacopo/trans_temp/all_atom_replica_results_chignolin/exchange_{type_sim}/coords_nowater'
processed_forces = f'/local_scratch2/jacopo/trans_temp/all_atom_replica_results_chignolin/exchange_{type_sim}/forces_nowater'

delta = (n_sim -1) * 10

print(f'Check delta should be ({n_sim} -1) * 10')
print(f'Value of delta -> {delta}')


# define different temperatures
Temperatures = np.geomspace(min_T, max_T, n_replicas)


## if we want to save some subset we need to define a topology
if data_type == 'all':
    # pdb file used for the simulation
    pdb_file = 'chi_sys.pdb'

    # find protein's atoms indeces
    topo = md.load(pdb_file).topology
    protein_index = topo.select("protein")
    protein_topo = topo.subset(protein_index)
else:
    protein_index = None




# load position and forces
pos_fns = sorted(glob(f"{pos_path}/position_*.npy"))
forces_fns = sorted(glob(f"{forces_path}/forces_*.npy"))

# verify position and forces have same length
assert len(pos_fns) == len(forces_fns)


print('Started')

if protein_index is not None:
    i = 0
    for pos_fn, force_fn in zip(pos_fns, forces_fns):    
        pos = np.load(pos_fn)[:, :, protein_index, :]
        force = np.load(force_fn)[:, :, protein_index, :]
        for idx in tqdm(range(n_replicas)):
            np.save(f"{processed_pos}/temperature_{Temperatures[idx]:.0f}/coor_{i+delta}", pos[idx]) 
            np.save(f"{processed_forces}/temperature_{Temperatures[idx]:.0f}/forces_{i+delta}", force[idx]) 
        print(f'Progress -> {i+1} over 10')
        i += 1

else:
    i = 0
    for pos_fn, force_fn in zip(pos_fns, forces_fns):    
        pos = np.load(pos_fn)
        force = np.load(force_fn)
        for idx in tqdm(range(n_replicas)):
            np.save(f"{processed_pos}/temperature_{Temperatures[idx]:.0f}/coor_{i+delta}", pos[idx]) 
            np.save(f"{processed_forces}/temperature_{Temperatures[idx]:.0f}/forces_{i+delta}", force[idx]) 
        print(f'Progress -> {i+1} over 10')
        i += 1




