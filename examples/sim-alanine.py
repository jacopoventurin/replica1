import numpy as np
from openmm import unit
import openmm.app as app
from openmmtools import states, mcmc
import openmmtools
from replica.ReplicaExchangeProtocol import ReplicaExchange
from replica.Reporters import ReplicaStateReporter
import time
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# System creation
# Define force field
forcefield=['amber99sbildn.xml','amber99_obc.xml']
constraints = app.HBonds
nonbonded_cutoff = 0.9*unit.nanometer
switch_distance = 0.75*unit.nanometer
# nonbonded_method = app.PME
implicit_solvent = False
hydrogen_mass = None

# Define initial constants of the input pdb file
friction = 0.1 / unit.picoseconds
pressure = 1.0 * unit.atmosphere
temperature_pdb = 300*unit.kelvin
timestep = 2.0 * unit.femtosecond 

# Define system 
create_system_kwargs = dict(
    removeCMMotion=True,
    # nonbondedMethod=nonbonded_method,
    nonbondedCutoff=nonbonded_cutoff,
    switchDistance=switch_distance,
    constraints=constraints,
    hydrogenMass=hydrogen_mass,
    # rigidWater=True,
)

# Replica setup
n_replicas = 3
min_T = 300
max_T = 500
Temps = np.geomspace(min_T, max_T, n_replicas)


init_system = openmmtools.testsystems.AlanineDipeptideImplicit()
ff = app.ForceField(*forcefield)
system = ff.createSystem(init_system.topology,**create_system_kwargs)


protocol = {'temperature': Temps * unit.kelvin}
thermodynamic_states = states.create_thermodynamic_state_protocol(system,protocol)

sampler_states = list()
for i_t,_ in enumerate(thermodynamic_states):
    sampler_states.append(openmmtools.states.SamplerState(positions=init_system.positions))

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
    save_temperatures_history=True,
)

# Define and load reporter 
reporter = ReplicaStateReporter('state.csv', reportInterval=1, time=True, potentialEnergy=True,
                                kineticEnergy=True, totalEnergy=True, bathTemperature=True, volume=True, elapsedTime=True)

parallel_tempering.load_reporter(reporter)



# Run symulation and save position and forces every 100 timesteps


sim_params ={
    'n_attempts': 1, 
    'md_timesteps': 520, #520 ps 
    'equilibration_timesteps': 20, # 20 ps
    'save': True, 
    'save_interval': 1, # save every 1 ps
    'mixing': 'neighbors',   #try exchange between neighbors only
    'save_atoms': 'protein',   #save position and forces of protein's atoms only
    'reshape_for_TICA': 'True',  #save in format for TICA analysis 
    'verbose': 'True'  #print time information
}


print('Simulation of 500 ns trajectory trying 4 exchange between neighbor replicas for each timesteps with rescale of velocities')
print('-' * 50)
start = time.time()
print(f'Simulation started at {time.ctime()}')
print('-' * 50)

for step in range(100):   # 500 ns of production time 
    # locally save position and forces every 5 ns
    partial_start = time.time()
    position, forces, acceptance = parallel_tempering.run(10, **sim_params)  
    
    if rank == 0:
        np.save(f'position_{step}.npy', position)
        np.save(f'forces_{step}.npy', forces)
        np.save(f'acceptance_{step}.npy', acceptance)
    
        if ((step+1)%10) == 0:
            parallel_tempering.save_temperature_history(filename=f'temperature_history_{(step+1)*5}ns.npy')
            parallel_tempering._save_context_checkpoints(f"checkpoint_{((step+1)*5)}ns")
    
        print(f'{(step+1)*5} ns of simulation done in {(time.time()-partial_start):.3f} s, simulation speed {86400*5/(time.time()-partial_start)}ns/day')

    del position
    del forces

# save temperature history
if rank == 0:
    parallel_tempering.save_temperature_history()


end = time.time()
print(f'Simulation ended at {time.ctime()}')
print(f'Total simulation time {end-start} [sec]')
