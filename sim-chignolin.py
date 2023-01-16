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
from Reporters import ReplicaStateReporter
import time


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
    mcmc_move=langevin_move,
    rescale_velocities=True,
)

# Load topology in order to allow 
parallel_tempering.load_topology(md.load_topology('chi_sys.pdb'))

# Define and load reporter 
reporter = ReplicaStateReporter('state.csv', reportInterval=20, time=True, potentialEnergy=True,
                                kineticEnergy=True, totalEnergy=True, volume=True, elapsedTime=True)

parallel_tempering.load_reporter(reporter)

# Load state of the simulation from checkpoint 
#parallel_tempering._load_contexts()

# Run symulation and save position and forces every 100 timesteps


sim_params ={
    'n_attempts': 129, #n_replicas*log(n_replicas)
    'md_timesteps': 600, #600 ps 
    'equilibration_timesteps': 100, # 100 ps
    'save': True, 
    'save_interval': 3, # save every 3 ps
    'checkpoint_simulations': False, 
    'mixing': 'all',   #try exchange between neighbors only
    'save_atoms': 'protein',   #save position and forces of protein's atoms only
    'reshape_for_TICA': 'True'  #save in format for TICA analysis 
}


print('Simulation of 20 ns trajectory trying 129 exchange between all replicas for each timesteps with rescale of velocities')
print('-' * 50)
start = time.time()
print(f'Simulation started at time {start}')
print('-' * 50)

for step in range(10):   # 20 ns of production time 
    # locally save position and forces every 2 ns
    position, forces, acceptance = parallel_tempering.run(4, **sim_params)  
    np.save(f'position_{step}.npy', position)
    np.save(f'forces_{step}.npy', forces)
    np.save(f'acceptance_{step}.npy', acceptance)
    
    del position
    del forces

    # save checkpoint after 10 ns
    if step == 4:
        parallel_tempering.save_checkpoint(code='10ns')
    
    print(f'{(step+1)*2} ns of simulation done')

# save final checkpoint
parallel_tempering.save_checkpoint(code='20ns')

temperature_history = parallel_tempering.get_temperature_history(asNpy=True)
np.save('temperature_history', temperature_history)
    
end = time.time()
print(f'Simulation ended at {end}')
print(f'Total simulation time {end-start}')
