import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit
from openmm.openmm import NonbondedForce
import os
from torchvision.datasets.utils import download_url
from openmmtools import states, mcmc
from openmmtools.multistate import ReplicaExchangeSampler, ReplicaExchangeAnalyzer
from openmmtools import multistate
import openmmtools
import tempfile
from openmm.unit import dalton, kelvin, second, nanometer, kilojoule_per_mole, picosecond
import openmm.unit as unit
import os
import os.path as osp

# System creation
n_atoms = 1
mass = 1.0*dalton
temperature = 2.405*kelvin
timestep = 1e-15*second # (fs)
gamma =  0.5 # (1/ps)
mass = 12.0*dalton
zero_mass = 0.0*dalton
spacing = 0.0
start_pos = np.zeros((n_atoms,3))
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
temperatures = [temperature, 1.5*temperature, 2*temperature]
integrators = {}
for i_temp,temperature in enumerate(temperatures):
    integrators[i_temp] = mm.LangevinIntegrator(temperature,gamma,timestep)
platform = mm.Platform.getPlatformByName("CUDA")
platform_properties = {}
for i_temp,temperature in enumerate(temperatures):
    platform_properties[i_temp] = {"DeviceIndex": "{}".format(i_temp%4), "Precision": "mixed"}

topo = app.Topology()
topo.addChain()
chain = list(topo.chains())[0]
topo.addResidue('C1',chain)
residue = list(topo.residues())[0]
element = app.Element.getBySymbol('C')
for atom in range(n_atoms):
    topo.addAtom('C',element,residue)
n_replicas = len(temperatures)

simulations = {}
for i_temp,temperature in enumerate(temperatures):
    simulations[i_temp] = app.Simulation(
        topo,system,integrators[i_temp],platform,platform_properties[i_temp]
    )
    simulations[i_temp].reporters.append(app.DCDReporter('output-T{}.dcd'.format(i_temp), 1000))
    simulations[i_temp].context.setPositions(start_pos)
    simulations[i_temp].context.setVelocitiesToTemperature(temperature)

def swap_attempt(simulations):
    success = 0
    ix,iy = np.random.choice(range(n_replicas),2,replace=False)
    if iy > ix:
        tmp = ix
        ix = iy
        iy = tmp
    energy_i = simulations[ix].context.getState(getEnergy=True).getPotentialEnergy()
    energy_j= simulations[iy].context.getState(getEnergy=True).getPotentialEnergy()
    temperature_i = simulations[ix].context.getIntegrator().getTemperature()
    temperature_j = simulations[iy].context.getIntegrator().getTemperature()
    beta_i = 1/(unit.BOLTZMANN_CONSTANT_kB*temperature_i)
    beta_j = 1/(unit.BOLTZMANN_CONSTANT_kB*temperature_j)
    prob = np.exp((energy_i-energy_j)*(beta_i-beta_j)/unit.AVOGADRO_CONSTANT_NA)
    print(prob,ix,iy)
    print(temperature_i,energy_i,beta_i*energy_i/unit.AVOGADRO_CONSTANT_NA)
    print(temperature_j,energy_j,beta_j*energy_j/unit.AVOGADRO_CONSTANT_NA)
    if prob > np.random.rand():
        success = 1
        pos_i = simulations[ix].context.getState(getPositions=True).getPositions()
        pos_j = simulations[iy].context.getState(getPositions=True).getPositions()
        simulations[ix].context.setPositions(pos_j)
        simulations[iy].context.setPositions(pos_i)
        simulations[ix].context.setVelocitiesToTemperature(temperature_i)
        simulations[iy].context.setVelocitiesToTemperature(temperature_j)

    return simulations, success, ix, iy

# How long to run each simulation
n_steps = 1000
# How many microcycles to perform
n_sims = 1000
# How many macrocycles to perform
n_loops = 100
# Success in upper half, total attempts in lower
swap_attempt_matrix = np.zeros((n_replicas,n_replicas))
# which replica to extract coordinates and forces from
i_replica = 0
forces_list = []
coords_list = []
energies = []

for loop in range(n_loops):
    for sim in range(n_sims):
        forces_list.append([])
        coords_list.append([])
        for i_temp, temperature in enumerate(temperatures):
            simulations[i_temp].step(n_steps)

        # Extract coordinates/forces
        forces = simulations[i_replica].context.getState(getForces=True).getForces(asNumpy=True)
        forces = forces.value_in_unit(kilojoule_per_mole/nanometer)
        position = simulations[i_replica].context.getState(getPositions=True).getPositions(asNumpy=True)
        position = position.value_in_unit(nanometer)
        energy = simulations[i_replica].context.getState(getEnergy=True).getPotentialEnergy()
        energy = energy.value_in_unit(kilojoule_per_mole)
        #
        forces_list[-1].append(forces)
        coords_list[-1].append(position)
        energies.append(energy)
        
        # Perform swap attempt
        simulations, success, ix, iy = swap_attempt(simulations)
        swap_attempt_matrix[iy,ix] += success
        swap_attempt_matrix[ix,iy] += 1

    # Write out
    forces = np.concatenate(forces_list,axis=0)
    positions = np.concatenate(coords_list,axis=0)
    np.save('forces-{}.npy'.format(loop),forces)
    np.save('coords-{}.npy'.format(loop),positions)
    np.save('energy-{}.npy'.format(loop),energies)
