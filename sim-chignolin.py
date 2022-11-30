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
forcefield=['amber99sb.xml','tip3p.xml']
forcefield=['amber96.xml','tip3p.xml']
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
pdb = app.PDBFile('chi_vac.pdb')

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

modeller.addSolvent(ff)

system = ff.createSystem(modeller.topology, **create_system_kwargs)

#sampler_state = mcmc.SamplerState(positions=system.positions)
#thermodynamic_state = mcmc.ThermodynamicState(system=system, temperature=298*unit.kelvin)


barostat = mm.MonteCarloBarostat(pressure, temperature)
force_id = system.addForce(barostat)

#integrator = mm.LangevinMiddleIntegrator(temperature , friction , timestep)
#integrator.setConstraintTolerance(1e-7)


# Replica setup
platform = mm.Platform.getPlatformByName("CUDA")
platform_properties = {"DeviceIndex": "0", "Precision": "mixed"}

n_replicas = 2

protocol = {'temperature': [300,302,304,306] * unit.kelvin}
thermodynamic_states = states.create_thermodynamic_state_protocol(system,protocol)


sampler_states = list()
for i_t,_ in enumerate(thermodynamic_states):
    sampler_states.append(openmmtools.states.SamplerState(positions=modeller.positions))

langevin_move = mcmc.LangevinSplittingDynamicsMove(
    timestep = timestep,
    collision_rate = friction,
    n_steps = 50,
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
positions, forces, acceptance = parallel_tempering.run(5000,save=True)

np.save('positions.npy', positions)
np.save('forces.npy', forces)
np.save('acceptance.npy', acceptance)


#for ii in range(4):
#    dist = compute_forces(positions[ii],forces[ii])
#    bins = np.linspace(0,0.02,25)
#    count,edges = np.histogram(dist,bins=bins)
#    px_dist = count / np.sum(count)
#    centers = (edges[1:]+edges[:-1])/2
#    plt.plot(centers,px_dist,alpha=0.5,label=ii)
#    energy = 0.5*big_k.value_in_unit(unit.kilocalorie_per_mole/unit.angstrom**2) * centers**2
#    beta = (1 / protocol['temperature'][ii] / unit.BOLTZMANN_CONSTANT_kB / 6.022e23).value_in_unit(unit.kilocalorie**-1)
#    px = np.exp(-beta*energy)
#    px = px / np.sum(px)
#    plt.plot(centers,px,label='{} exact'.format(ii))
#plt.legend()
#plt.show()
#x = 5