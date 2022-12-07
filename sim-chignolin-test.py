import numpy as np
import openmm as mm
import mdtraj as md
from openmm import unit
import openmm.app as app
import os
from openmmtools import states, mcmc
from openmmtools.multistate import ReplicaExchangeSampler, MultiStateReporter
import openmmtools
import os
import os.path as osp
import tempfile



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
barostat = mm.MonteCarloBarostat(pressure, temperature)
force_id = system.addForce(barostat)


platform = mm.Platform.getPlatformByName("CUDA")
platform_properties = {"DeviceIndex": "0", "Precision": "mixed"}

# Replica setup
n_replicas = 5
min_T = 300
max_T = 320
Temps = np.geomspace(min_T, max_T, n_replicas)


thermodynamic_states = [states.ThermodynamicState(system=system, temperature=T) for T in Temps]


#Initialize simulation object with options. Run with a Langevin integrator.

langevin_move = mcmc.LangevinDynamicsMove(
    timestep = timestep,
    collision_rate = friction,
    n_steps = 1000, # 2ps
    reassign_velocities=False,
)
simulation = ReplicaExchangeSampler(mcmc_moves=langevin_move, number_of_iterations=5000)

#Create simulation with its storage file and run.

storage_path = '/data/traj.nc'
reporter = MultiStateReporter(storage_path, checkpoint_interval=100)
simulation.create(thermodynamic_states=thermodynamic_states,
                  sampler_states=states.SamplerState(system.positions),
                  storage=reporter)

simulation.run(n_iterations=5000)


