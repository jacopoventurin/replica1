import numpy as np
import mdtraj as md
from tqdm import tqdm

from glob import glob


## number of previous simulation saved
delta = 0  

## path of the results from all thoms simulation 

## Exchange all
#pos_path = '/import/a12/users/jacopo/transf_temp/all_atom_replica_results_chignolin/exchange_all/sim1'
#forces_path = '/import/a12/users/jacopo/transf_temp/all_atom_replica_results_chignolin/exchange_all/sim1'
## Exchange neighbors
pos_path = '/import/a12/users/jacopo/transf_temp/all_atom_replica_results_chignolin/exchange_neighbors/sim1'
forces_path = '/import/a12/users/jacopo/transf_temp/all_atom_replica_results_chignolin/exchange_neighbors/sim1'

## pdb file used for the simulation
pdb_file = 'chi_sys.pdb'


# simulation parameters
n_replicas = 36
min_T = 280
max_T = 400
Temperatures = np.geomspace(min_T, max_T, n_replicas)

## paths where to save results of trajectories elaboration are created

## Excahge all 
#processed_pos = '/local_scratch2/jacopo/trans_temp/all_atom_replica_results_chignolin/exchange_all/coords_nowater'
#processed_forces = '/local_scratch2/jacopo/trans_temp/all_atom_replica_results_chignolin/exchange_all/forces_nowater'
# Exchange neighbors
processed_pos = '/local_scratch2/jacopo/trans_temp/all_atom_replica_results_chignolin/exchange_neighbors/coords_nowater'
processed_forces = '/local_scratch2/jacopo/trans_temp/all_atom_replica_results_chignolin/exchange_neighbors/forces_nowater'



# find protein's atoms indeces
topo = md.load(pdb_file).topology
protein_index = topo.select("protein")
protein_topo = topo.subset(protein_index)




# load position and forces
pos_fns = sorted(glob(f"{pos_path}/position_*.npy"))
forces_fns = sorted(glob(f"{forces_path}/forces_*.npy"))

# verify position and forces have same length
assert len(pos_fns) == len(forces_fns)

i = 0
print('Started')
for pos_fn, force_fn in zip(pos_fns, forces_fns):    
    pos = np.load(pos_fn)[:, :, protein_index, :]
    force = np.load(force_fn)[:, :, protein_index, :]
    for idx in tqdm(range(n_replicas)):
        np.save(f"{processed_pos}/temperature_{Temperatures[idx]:.0f}/coor_{i+delta}", pos[idx]) 
        np.save(f"{processed_forces}/temperature_{Temperatures[idx]:.0f}/forces_{i+delta}", force[idx]) 
    print(f'Progress -> {i+1} over 10')
    i += 1



##########################################################################################################################################################

## number of previous simulation saved
delta = 10  

## path of the results from all thoms simulation 

## Exchange all
#pos_path = '/import/a12/users/jacopo/transf_temp/all_atom_replica_results_chignolin/exchange_all/sim1'
#forces_path = '/import/a12/users/jacopo/transf_temp/all_atom_replica_results_chignolin/exchange_all/sim1'
## Exchange neighbors
pos_path = '/import/a12/users/jacopo/transf_temp/all_atom_replica_results_chignolin/exchange_neighbors/sim2'
forces_path = '/import/a12/users/jacopo/transf_temp/all_atom_replica_results_chignolin/exchange_neighbors/sim2'

## pdb file used for the simulation
pdb_file = 'chi_sys.pdb'


# simulation parameters
n_replicas = 36
min_T = 280
max_T = 400
Temperatures = np.geomspace(min_T, max_T, n_replicas)

## paths where to save results of trajectories elaboration are created

## Excahge all 
#processed_pos = '/local_scratch2/jacopo/trans_temp/all_atom_replica_results_chignolin/exchange_all/coords_nowater'
#processed_forces = '/local_scratch2/jacopo/trans_temp/all_atom_replica_results_chignolin/exchange_all/forces_nowater'
# Exchange neighbors
processed_pos = '/local_scratch2/jacopo/trans_temp/all_atom_replica_results_chignolin/exchange_neighbors/coords_nowater'
processed_forces = '/local_scratch2/jacopo/trans_temp/all_atom_replica_results_chignolin/exchange_neighbors/forces_nowater'



# find protein's atoms indeces
topo = md.load(pdb_file).topology
protein_index = topo.select("protein")
protein_topo = topo.subset(protein_index)




# load position and forces
pos_fns = sorted(glob(f"{pos_path}/position_*.npy"))
forces_fns = sorted(glob(f"{forces_path}/forces_*.npy"))

# verify position and forces have same length
assert len(pos_fns) == len(forces_fns)

i = 0
print('Started')
for pos_fn, force_fn in zip(pos_fns, forces_fns):    
    pos = np.load(pos_fn)[:, :, protein_index, :]
    force = np.load(force_fn)[:, :, protein_index, :]
    for idx in tqdm(range(n_replicas)):
        np.save(f"{processed_pos}/temperature_{Temperatures[idx]:.0f}/coor_{i+delta}", pos[idx]) 
        np.save(f"{processed_forces}/temperature_{Temperatures[idx]:.0f}/forces_{i+delta}", force[idx]) 
    print(f'Progress -> {i+1} over 10')
    i += 1

########################################################################################################################################################################################################################

## number of previous simulation saved
delta = 0  

## path of the results from all thoms simulation 

## Exchange all
pos_path = '/import/a12/users/jacopo/transf_temp/all_atom_replica_results_chignolin/exchange_all/sim1'
forces_path = '/import/a12/users/jacopo/transf_temp/all_atom_replica_results_chignolin/exchange_all/sim1'
## Exchange neighbors
#pos_path = '/import/a12/users/jacopo/transf_temp/all_atom_replica_results_chignolin/exchange_neighbors/sim2'
#forces_path = '/import/a12/users/jacopo/transf_temp/all_atom_replica_results_chignolin/exchange_neighbors/sim2'

## pdb file used for the simulation
pdb_file = 'chi_sys.pdb'


# simulation parameters
n_replicas = 36
min_T = 280
max_T = 400
Temperatures = np.geomspace(min_T, max_T, n_replicas)

## paths where to save results of trajectories elaboration are created

## Excahge all 
processed_pos = '/local_scratch2/jacopo/trans_temp/all_atom_replica_results_chignolin/exchange_all/coords_nowater'
processed_forces = '/local_scratch2/jacopo/trans_temp/all_atom_replica_results_chignolin/exchange_all/forces_nowater'
# Exchange neighbors
#processed_pos = '/local_scratch2/jacopo/trans_temp/all_atom_replica_results_chignolin/exchange_neighbors/coords_nowater'
#processed_forces = '/local_scratch2/jacopo/trans_temp/all_atom_replica_results_chignolin/exchange_neighbors/forces_nowater'



# find protein's atoms indeces
topo = md.load(pdb_file).topology
protein_index = topo.select("protein")
protein_topo = topo.subset(protein_index)




# load position and forces
pos_fns = sorted(glob(f"{pos_path}/position_*.npy"))
forces_fns = sorted(glob(f"{forces_path}/forces_*.npy"))

# verify position and forces have same length
assert len(pos_fns) == len(forces_fns)

i = 0
print('Started')
for pos_fn, force_fn in zip(pos_fns, forces_fns):    
    pos = np.load(pos_fn)[:, :, protein_index, :]
    force = np.load(force_fn)[:, :, protein_index, :]
    for idx in tqdm(range(n_replicas)):
        np.save(f"{processed_pos}/temperature_{Temperatures[idx]:.0f}/coor_{i+delta}", pos[idx]) 
        np.save(f"{processed_forces}/temperature_{Temperatures[idx]:.0f}/forces_{i+delta}", force[idx]) 
    print(f'Progress -> {i+1} over 10')
    i += 1

    ####################################################################################################################################################################################################





## number of previous simulation saved
delta = 10  

## path of the results from all thoms simulation 

## Exchange all
pos_path = '/import/a12/users/jacopo/transf_temp/all_atom_replica_results_chignolin/exchange_all/sim2'
forces_path = '/import/a12/users/jacopo/transf_temp/all_atom_replica_results_chignolin/exchange_all/sim2'
## Exchange neighbors
#pos_path = '/import/a12/users/jacopo/transf_temp/all_atom_replica_results_chignolin/exchange_neighbors/sim2'
#forces_path = '/import/a12/users/jacopo/transf_temp/all_atom_replica_results_chignolin/exchange_neighbors/sim2'

## pdb file used for the simulation
pdb_file = 'chi_sys.pdb'


# simulation parameters
n_replicas = 36
min_T = 280
max_T = 400
Temperatures = np.geomspace(min_T, max_T, n_replicas)

## paths where to save results of trajectories elaboration are created

## Excahge all 
processed_pos = '/local_scratch2/jacopo/trans_temp/all_atom_replica_results_chignolin/exchange_all/coords_nowater'
processed_forces = '/local_scratch2/jacopo/trans_temp/all_atom_replica_results_chignolin/exchange_all/forces_nowater'
# Exchange neighbors
#processed_pos = '/local_scratch2/jacopo/trans_temp/all_atom_replica_results_chignolin/exchange_neighbors/coords_nowater'
#processed_forces = '/local_scratch2/jacopo/trans_temp/all_atom_replica_results_chignolin/exchange_neighbors/forces_nowater'



# find protein's atoms indeces
topo = md.load(pdb_file).topology
protein_index = topo.select("protein")
protein_topo = topo.subset(protein_index)




# load position and forces
pos_fns = sorted(glob(f"{pos_path}/position_*.npy"))
forces_fns = sorted(glob(f"{forces_path}/forces_*.npy"))

# verify position and forces have same length
assert len(pos_fns) == len(forces_fns)

i = 0
print('Started')
for pos_fn, force_fn in zip(pos_fns, forces_fns):    
    pos = np.load(pos_fn)[:, :, protein_index, :]
    force = np.load(force_fn)[:, :, protein_index, :]
    for idx in tqdm(range(n_replicas)):
        np.save(f"{processed_pos}/temperature_{Temperatures[idx]:.0f}/coor_{i+delta}", pos[idx]) 
        np.save(f"{processed_forces}/temperature_{Temperatures[idx]:.0f}/forces_{i+delta}", force[idx]) 
    print(f'Progress -> {i+1} over 10')
    i += 1
##################################################################################################################################################################################################


## number of previous simulation saved
delta = 20  

## path of the results from all thoms simulation 

## Exchange all
pos_path = '/import/a12/users/jacopo/transf_temp/all_atom_replica_results_chignolin/exchange_all/sim3'
forces_path = '/import/a12/users/jacopo/transf_temp/all_atom_replica_results_chignolin/exchange_all/sim3'
## Exchange neighbors
#pos_path = '/import/a12/users/jacopo/transf_temp/all_atom_replica_results_chignolin/exchange_neighbors/sim2'
#forces_path = '/import/a12/users/jacopo/transf_temp/all_atom_replica_results_chignolin/exchange_neighbors/sim2'

## pdb file used for the simulation
pdb_file = 'chi_sys.pdb'


# simulation parameters
n_replicas = 36
min_T = 280
max_T = 400
Temperatures = np.geomspace(min_T, max_T, n_replicas)

## paths where to save results of trajectories elaboration are created

## Excahge all 
processed_pos = '/local_scratch2/jacopo/trans_temp/all_atom_replica_results_chignolin/exchange_all/coords_nowater'
processed_forces = '/local_scratch2/jacopo/trans_temp/all_atom_replica_results_chignolin/exchange_all/forces_nowater'
# Exchange neighbors
#processed_pos = '/local_scratch2/jacopo/trans_temp/all_atom_replica_results_chignolin/exchange_neighbors/coords_nowater'
#processed_forces = '/local_scratch2/jacopo/trans_temp/all_atom_replica_results_chignolin/exchange_neighbors/forces_nowater'


# find protein's atoms indeces
topo = md.load(pdb_file).topology
protein_index = topo.select("protein")
protein_topo = topo.subset(protein_index)




# load position and forces
pos_fns = sorted(glob(f"{pos_path}/position_*.npy"))
forces_fns = sorted(glob(f"{forces_path}/forces_*.npy"))

# verify position and forces have same length
assert len(pos_fns) == len(forces_fns)

i = 0
print('Started')
for pos_fn, force_fn in zip(pos_fns, forces_fns):    
    pos = np.load(pos_fn)[:, :, protein_index, :]
    force = np.load(force_fn)[:, :, protein_index, :]
    for idx in tqdm(range(n_replicas)):
        np.save(f"{processed_pos}/temperature_{Temperatures[idx]:.0f}/coor_{i+delta}", pos[idx]) 
        np.save(f"{processed_forces}/temperature_{Temperatures[idx]:.0f}/forces_{i+delta}", force[idx]) 
    print(f'Progress -> {i+1} over 10')
    i += 1