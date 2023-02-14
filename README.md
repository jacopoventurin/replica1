# replica1

This project contains code to perform parallel tempering simulation with openmm.

The ReplicaExchange class support multiple process simulation via MPI.
For example, if we whant to run sim.py usning 4 process

```
mpiexec --oversubscribe -n 4 python sim.py

```
The class also supports different swapping method: is possible to specify if
`neighbor` replicas only should be exchanged or if is possible to attempt swaps between `all` the couples to
improve the mixing. 
During the simulation, it is also possible to save position and forces of any specific
part of the system, using a `mdtraj`-compatible selection. For example, if we are running an explicit
solvent protein simulation, and we are interested in position and forces acting on the protein only, is
possible to specify `protein` as target atoms we want to track. 
Another quantity that is computed and
returned after the simulation is completed is the acceptance matrix that reports under the diagonal
the number of attempted swap, and above the diagonal the number of accepted swap for all possible
couples of replicas.
Like in the openmm simualtion, is possible to add a reporter in order to keep track of relevant
thermodynamic quantity during the simulation. The class `RepliaStateReporter` was created for this purpose. In
`RepliaStateReporter` is also possible to specify if `bathTemperatures` should be saved. This can be useful if we want
to unbias the simulation to the target temperature

# Example

```
import numpy as np
import openmm as mm
import mdtraj as md
from openmm import unit
import openmm.app as app
import os
from openmmtools import states, mcmc
import openmmtools
from ReplicaExchangeProtocol import ReplicaExchange
from Reporters import ReplicaStateReporter
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# Define the system...

pdb = app.PDBFile(...)
modeller = app.modeller.Modeller(pdb.topology, pdb.positions)

ff = app.ForceField(*forcefield)

system = ff.createSystem(modeller.topology, **create_system_kwargs)


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

langevin_move = mcmc.LangevinSplittingDynamicsMove(...)

# Define class for replica exchange
parallel_tempering = ReplicaExchange(
    thermodynamic_states=thermodynamic_states, 
    sampler_states=sampler_states, 
    mcmc_move=langevin_move,
    save_temperatures_history=True,
)

# Load topology in order to save protein positions only
parallel_tempering.load_topology(md.load_topology('chi_sys.pdb'))

# Define and load reporter 
reporter = ReplicaStateReporter('state.csv', reportInterval=1, time=True, potentialEnergy=True,
                                kineticEnergy=True, totalEnergy=True, bathTemperature=True)
parallel_tempering.load_reporter(reporter)


# Run symulation and save position and forces
position, forces, acceptance = parallel_tempering.run(10, **sim_params)
if rank == 0:
      np.save(f'position.npy', position)
      np.save(f'forces.npy', forces)
      np.save(f'acceptance.npy', acceptance)

# Empty memory
del position
del forces
    
# save temperature history
if rank == 0:
    parallel_tempering.save_temperature_history()

```