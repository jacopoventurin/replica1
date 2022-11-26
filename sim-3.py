import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit
from openmm.openmm import NonbondedForce
import os
from openmmtools import states, mcmc
from openmmtools.multistate import ReplicaExchangeSampler, ReplicaExchangeAnalyzer
from openmmtools import multistate
import openmmtools
import tempfile
from openmm.unit import dalton, kelvin, second, nanometer, kilojoule_per_mole, picosecond
import os
import os.path as osp

if osp.exists('output-1.nc'): os.system('rm output-1.nc')
# System creation
n_atoms = 1
mass = 1.0*dalton
temperature = 2.405*kelvin
timestep = 1e-15*second # (fs)
gamma =  0.5 # (1/ps)
mass = 12.0*dalton
zero_mass = 0.0*dalton
spacing = 0.0
start_pos = np.zeros((2,n_atoms,3))
start_pos[0,0,0] = 3.33
start_pos[1,0,0] = -3.33
# 
system = mm.System()
for ix in range(0,n_atoms):
    system.addParticle(mass)
# Specify bonding connectivity
force = mm.CustomExternalForce('E0*(C*r^4-r^2); r=sqrt(x^2+y^2+z^2)')
force.addGlobalParameter('E0', 0.2*unit.kilojoule_per_mole/unit.nanometers**2)
force.addGlobalParameter('C', 0.045/unit.nanometers**2)
force.addParticle(0,[])

forcexy = mm.CustomExternalForce('k*y^2+k*z^2')
forcexy.addGlobalParameter('k', 100*unit.kilojoule_per_mole/unit.nanometers**2)
forcexy.addParticle(0,[])


# system.addForce(harmonic_force)
system.addForce(force)
system.addForce(forcexy)
#

# Replica setup
integrator = mm.LangevinIntegrator(temperature,gamma,timestep)
platform = mm.Platform.getPlatformByName("CUDA")
platform_properties = {"DeviceIndex": "0", "Precision": "mixed"}

n_replicas = 2
T_min = 1e-9 * unit.kelvin
T_max = 10 * unit.kelvin
dT = (T_max-T_min) / n_replicas
temperatures = [T_min + dT * i for i in range(n_replicas)]
value_temperatures = [T_min + (T_max-T_min)/n_replicas * i for i in range(n_replicas)]

sampler_states = list()
thermodynamic_states = list()

for i_t,temperature in enumerate(temperatures):
    thermodynamic_state = openmmtools.states.ThermodynamicState(
        system=system, temperature=temperature
    )
    thermodynamic_states.append(thermodynamic_state)
    sampler_states.append(
        openmmtools.states.SamplerState(start_pos[i_t]) #,box_vectors=pdb.topology.getPeriodicBoxVectors())
    )
its0 = 0
ts0 = thermodynamic_states[0]
Tdiff = np.abs(temperatures[0].value_in_unit(mm.unit.kelvin)-300)
for i_temp,temperature in enumerate(temperatures):
    if np.abs(temperature.value_in_unit(mm.unit.kelvin)-300) < Tdiff:
        ts0 = thermodynamic_states[i_temp] 
        Tdiff = np.abs(temperatures[i_temp].value_in_unit(mm.unit.kelvin)-300)
        its0 = i_temp

move = mcmc.GHMCMove(
    timestep = timestep,
    collision_rate = gamma/picosecond,
    n_steps = 5,
)

simulation = ReplicaExchangeSampler(mcmc_moves=move, number_of_iterations=1)
reporter = openmmtools.multistate.MultiStateReporter("output-1.nc", checkpoint_interval=1)
simulation.create(thermodynamic_states, sampler_states, reporter)


# Extract forces, build new context to load in positions
topo = app.Topology()
topo.addChain()
chain = list(topo.chains())[0]
topo.addResidue('C1',chain)
residue = list(topo.residues())[0]
element = app.Element.getBySymbol('C')
for atom in range(n_atoms):
    topo.addAtom('C',element,residue)
force_simulation = app.Simulation(topo,system,integrator,platform,platform_properties)


def get_force(pos):
    force_simulation.context.setPositions(pos)
    force_simulation.step(0)
    forces = force_simulation.context.getState(getForces=True).getForces(asNumpy=True)
    forces = forces.value_in_unit(kilojoule_per_mole/nanometer)
    return forces

pos0 = []
pos1 = []

for sim in range(10):
    kT = []
    for ts in reporter.read_thermodynamic_states()[0]:
        kT.append(ts.kT)
    pos = simulation.sampler_states[0].positions
    if pos.unit.is_dimensionless():
        pos.unit = nanometer
    pos0.append(pos)
    force = get_force(pos)
    # print(sim,kT[0],pos[0],force[0])

    pos = simulation.sampler_states[1].positions
    if pos.unit.is_dimensionless():
        pos.unit = nanometer
    pos1.append(pos)
    force = get_force(pos)
    # print(sim,kT[1],pos[0],force[0])

    simulation.extend(1)



# for sim in range(1):
#     coords_list = {}
#     forces_list = {}
#     for i_t in range(len(thermodynamic_states)):
#         coords_list[i_t] = []
#         forces_list[i_t] = []
#     for ix in range(1): 
#         simulation.extend(1)
        # r0 = reporter.read_sampler_states(ix)
        # for i_t in range(len(thermodynamic_states)):
        #     coords_list[i_t] = r0[i_t].positions

    # coords = {}
    # forces = {}
    # for i_t in range(len(thermodynamic_states)):
    #     coords[i_t] = np.concatenate(coords_list[i_t],axis=0)
    #     forces[i_t] = np.concatenate(forces_list[i_t],axis=0)
    # np.save('coords-{}.npy'.format(sim),coords)
    # np.save('forces-{}.npy'.format(sim),forces)
    
#        for i_temp,temperature in enumerate(value_temperatures):
#            coords_list[temperature].append([])  
#            context = simulation.sampler_states[i_temp]
#            coords_list[temperature][-1].append(context.positions.value_in_unit(mm.unit.angstrom))
        # coords_list[-1].append(context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(mm.unit.angstrom)[atom_list]) 
        
#    coords = {}
#    for i_temp,temperature in enumerate(value_temperatures):
#        coords[temperature] = np.concatenate(coords_list[temperature],axis=0)
#    np.save('coords-{}.npy'.format(sim),coords)


analyzer = ReplicaExchangeAnalyzer(reporter)

# (   replica_energies,
#     unsampled_state_energies,
#     neighborhoods,
#     replica_state_indices,
# ) = analyzer.read_energies()
# def compute_forces(coords):
#     dist = np.linalg.norm(coords[0]-coords[1])
#     F = np.zeros(3)
#     for ix in range(3):
#         tmp = -big_k.value_in_unit(kilojoule_per_mole/nanometer/nanometer) * (coords[1][ix]-coords[0][ix]).value_in_unit(nanometer)
#         F[ix] = tmp
#     return F

# for step in range(20):
#     print(step)
#     state = reporter.read_last_iteration(last_checkpoint=False)
#     for ix in range(n_replicas):
#         r0 = reporter.read_sampler_states(state)[ix].positions[0]
#         r1 = reporter.read_sampler_states(state)[ix].positions[1]
#         s0 = simulation.sampler_states[ix].positions[0]
#         s1 = simulation.sampler_states[ix].positions[1]
#         # print(ix,np.linalg.norm(r0-r1),np.linalg.norm(s0-s1))
#         # print(r0)
#         # print(r1)
#         local_context_cache = move._get_context_cache(None)
#         context,integrator = local_context_cache.get_context(thermodynamic_states[ix])
#         coords = context.getState(getPositions=True).getPositions(asNumpy=True)
#         forces = context.getState(getForces=True).getForces(asNumpy=True)
#         print('coords')
#         print(coords)
#         print('force')
#         print(forces)
#         print(compute_forces(coords))
#         print('')

        # context, integrator = simulation.energy_context_cache.get_context(thermodynamic_states[ix])
        # forces = context.getState(getForces=True).getForces(asNumpy=True)
        # coords = context.getState(getPositions=True).getPositions(asNumpy=True)
        # print(coords)
        # print(forces)
        # print(compute_forces(coords))
        # print('')
        # print(coords-r0)
        # print('')

    # i0 = move._get_integrator(thermodynamic_states[0])
    # i1 = move._get_integrator(thermodynamic_states[1])
    # c0,_ = local_context_cache.get_context(thermodynamic_states[0],i0)
    # c1,_ = local_context_cache.get_context(thermodynamic_states[1],i1)
    # coords = c0.getState(getPositions=True).getPositions(asNumpy=True)
    # print(coords)
    # coords = c1.getState(getPositions=True).getPositions(asNumpy=True)
    # print(coords)
    # simulation.extend(1)