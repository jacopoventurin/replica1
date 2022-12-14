import numpy as np
import openmm as mm
import mdtraj as md
from openmm import unit
import openmm.app as app
import os
from openmmtools import states, mcmc
import openmmtools
import os
import os.path as osp
from ReplicaExchangeProtocol import ReplicaExchange


if osp.exists('output-1.nc'): os.system('rm output-1.nc')

# System creation
# Define force field
forcefield=['amber99sbildn.xml','tip3p.xml']
constraints = app.HBonds
nonbonded_cutoff = 0.9*unit.nanometer
switch_distance = 0.75*unit.nanometer
nonbonded_method = app.PME
implicit_solvent = False
solvated = True
hydrogen_mass = None

# Define initial constants of the input pdb file
friction = 0.1 / unit.picoseconds
pressure = 1.0 * unit.atmosphere
temperature = 300*unit.kelvin
timestep = 2.0 * unit.femtosecond

# Load pdb file 
pdb = app.PDBFile('chi_sys.pdb')
modeller = app.modeller.Modeller(pdb.topology, pdb.positions)

# Define system 
create_system_kwargs = dict(
    removeCMMotion=True,
    nonbondedMethod=nonbonded_method,
    nonbondedCutoff=nonbonded_cutoff,
    switchDistance=switch_distance,
    constraints=constraints,
    hydrogenMass=hydrogen_mass,
    rigidWater=True,
)


ff = app.ForceField(*forcefield)
# Add solvent
#modeller.addSolvent(ff)

# Create a system
system = ff.createSystem(modeller.topology, **create_system_kwargs)

#barostat = mm.MonteCarloBarostat(pressure, temperature)
#force_id = system.addForce(barostat)


platform = mm.Platform.getPlatformByName("CUDA")
platform_properties = {"DeviceIndex": "0", "Precision": "mixed"}

# Replica setup
n_replicas = 36
min_T = 280
max_T = 400
Temps = np.geomspace(min_T, max_T, n_replicas)

protocol = {'temperature': Temps * unit.kelvin}
thermodynamic_states = states.create_thermodynamic_state_protocol(system,protocol)

sampler_states = list()
for i_t,_ in enumerate(thermodynamic_states):
    sampler_states.append(openmmtools.states.SamplerState(positions=modeller.positions))

langevin_move = mcmc.LangevinSplittingDynamicsMove(
    timestep = timestep,
    collision_rate = friction,
    n_steps = 500, # 1ps
    reassign_velocities=False,
    n_restart_attempts=20,
    splitting="V R O R V"
)

# Define class for replica exchange
parallel_tempering = ReplicaExchange(
    thermodynamic_states=thermodynamic_states, 
    sampler_states=sampler_states, 
    mcmc_move=langevin_move
)

# Run symulation and save position and forces every 100 timesteps


sim_params ={
    'n_attempts': 1, 
    'equilibration_timesteps': 100, # 100 ps
    'production_timesteps': 500,  # 500 ps
    'save': True, 
    'save_interval': 2, # save every 2 ps
    'checkpoint_simulations': True, 
    'mixing': 'neighbors'   #try exchange between neighbors only
}

position, forces, acceptance = parallel_tempering.run(200, **sim_params) # 100 ns of production time 
np.save(f'position.npy', position)
np.save(f'forces.npy', forces)
np.save(f'acceptance.npy', acceptance)

    
    



