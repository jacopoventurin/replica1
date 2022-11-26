import numpy as np
import openmm as mm
import openmm.app as app
import os
from openmmtools import states, mcmc
from openmmtools.multistate import ReplicaExchangeSampler, ReplicaExchangeAnalyzer
from openmmtools import multistate
from openmmtools.mcmc import SequenceMove, MCMCSampler, GHMCMove
import openmmtools
from openmm.unit import dalton, kelvin, second, nanometer, kilojoule_per_mole, picosecond
import os
import os.path as osp


if osp.exists('output-1.nc'): os.system('rm output-1.nc')
# System creation
n_atoms = 2
mass = 1.0*dalton
temperature = 2.405*kelvin
timestep = 1e-15*second # (fs)
gamma =  0.5 # (1/ps)
spacing = 0.5
start_pos = np.zeros((n_atoms,3))
start_pos[0,0] = -0.5*spacing
start_pos[1,0] = 0.5*spacing
# 
system = mm.System()
for ix in range(0,n_atoms):
    system.addParticle(mass)
# Specify bonding connectivity
big_k = 500*kilojoule_per_mole/nanometer/nanometer
big_r0 = 0*spacing*nanometer
harmonic_force = mm.HarmonicBondForce()
harmonic_force.addBond(0,1,big_r0,big_k)
system.addForce(harmonic_force)
#

# Replica setup
integrator = mm.LangevinIntegrator(temperature,gamma,timestep)
# platform = mm.Platform.getPlatformByName("CUDA")
# platform_properties = {"DeviceIndex": "0", "Precision": "mixed"}

n_replicas = 10
T_min = 2.405 * kelvin
T_max = 5.440 * kelvin
dT = (T_max-T_min) / n_replicas
temperatures = [T_min + dT * i for i in range(n_replicas)]
value_temperatures = [T_min + (T_max-T_min)/n_replicas * i for i in range(n_replicas)]

sampler_states = list()
thermodynamic_states = list()

for temperature in temperatures:
    thermodynamic_state = openmmtools.states.ThermodynamicState(
        system=system, temperature=temperature
    )
    thermodynamic_states.append(thermodynamic_state)
    sampler_states.append(
        openmmtools.states.SamplerState(start_pos) 
    )

langevin_move = mcmc.LangevinSplittingDynamicsMove(
    n_steps = 100,
    collision_rate = 1.0 / picosecond,
)


simulation = ReplicaExchangeSampler(mcmc_moves=langevin_move, number_of_iterations=10)
reporter = openmmtools.multistate.MultiStateReporter("output-1.nc", checkpoint_interval=1)
simulation.create(thermodynamic_states, sampler_states, reporter)
simulation.minimize()
simulation.run()

def check_compute_forces(coords):
    F = np.zeros(3)
    for ix in range(3):
        F[ix] = -big_k.value_in_unit(kilojoule_per_mole/nanometer/nanometer) * (coords[1][ix]-coords[0][ix]).value_in_unit(nanometer)
    return F

for step in range(500):
    print(step)
    state = reporter.read_last_iteration(last_checkpoint=False)
    for ix in range(n_replicas):
        print("thermodynamic state {}".format(ix))
        r0 = reporter.read_sampler_states(state)[ix].positions
        # r1 = reporter.read_sampler_states(state)[ix].positions[1]
        s0 = simulation.sampler_states[ix].positions
        # s1 = simulation.sampler_states[ix].positions[1]
        print('coords from simulation sampler')
        print(s0)
        print('coords from reporter')
        print(r0)
        #
        local_context_cache = langevin_move._get_context_cache(None)
        integrator = langevin_move._get_integrator(thermodynamic_states[ix])
        context,integrator = local_context_cache.get_context(thermodynamic_states[ix],integrator)
        coords = context.getState(getPositions=True).getPositions(asNumpy=True)
        forces = context.getState(getForces=True).getForces(asNumpy=True)
        print('coords from context')
        print(coords)
        print('force from context')
        print(forces)
        print(check_compute_forces(coords))
        print('')

    # print("Coords and fores from context")
    # i0 = move._get_integrator(thermodynamic_states[0])
    # i1 = move._get_integrator(thermodynamic_states[1])
    # local_context_cache0 = move._get_context_cache(None)
    # c0,_ = local_context_cache0.get_context(thermodynamic_states[0],i0)
    # local_context_cache1 = move._get_context_cache(None)
    # c1,_ = local_context_cache1.get_context(thermodynamic_states[1],i1)
    # coords = c0.getState(getPositions=True).getPositions(asNumpy=True)
    # print(coords)
    # coords = c1.getState(getPositions=True).getPositions(asNumpy=True)
    # print(coords)
    simulation.extend(1)

