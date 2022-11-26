import numpy as np
import openmm as mm
from openmm import unit
import os
from openmmtools import states, mcmc
import openmmtools
import os
import os.path as osp
from ReplicaExchangeProtocol import ReplicaExchange
import matplotlib.pyplot as plt

if osp.exists('output-1.nc'): os.system('rm output-1.nc')
# System creation
n_atoms = 2
mass = 1.0*unit.dalton
timestep = 1e-15*unit.second # (fs)
gamma =  0.5 # (1/ps)
spacing = 0.0
start_pos = np.zeros((n_atoms,3))
start_pos[0,0] = -0.5*spacing
start_pos[1,0] = 0.5*spacing
# 
system = mm.System()
for ix in range(0,n_atoms):
    system.addParticle(mass)
# Specify bonding connectivity
big_k = 500*unit.kilocalorie_per_mole/unit.angstrom**2
big_r0 = 0*spacing*unit.angstrom
harmonic_force = mm.HarmonicBondForce()
harmonic_force.addBond(0,1,big_r0,big_k)

system.addForce(harmonic_force)
#

# Replica setup
platform = mm.Platform.getPlatformByName("CUDA")
platform_properties = {"DeviceIndex": "0", "Precision": "mixed"}

n_replicas = 2

protocol = {'temperature': [1e-2,1,5,10] * unit.kelvin}
thermodynamic_states = states.create_thermodynamic_state_protocol(system,protocol)
sampler_states = list()
for i_t,_ in enumerate(thermodynamic_states):
    pos = unit.Quantity(start_pos,unit=unit.angstrom)
    sampler_states.append(openmmtools.states.SamplerState(positions=pos))

langevin_move = mcmc.LangevinSplittingDynamicsMove(
    timestep = timestep,
    collision_rate = gamma/unit.picosecond,
    n_steps = 50,
    reassign_velocities=False,
    n_restart_attempts=20,
    splitting="V R O R V"
)

parallel_tempering = ReplicaExchange(
    thermodynamic_states=thermodynamic_states, 
    sampler_states=sampler_states, 
    mcmc_move=langevin_move
)
positions, forces, acceptance = parallel_tempering.run(5000,save=True)

def compute_forces(coords,forces):
    n_frames = coords.shape[0]
    dists = []
    for fr in range(n_frames):
        dists.append(np.linalg.norm(coords[fr,1]-coords[fr,0]))
        F = np.zeros(3)
        for ix in range(3):
            tmp = -big_k.value_in_unit(unit.kilocalorie_per_mole/unit.angstrom**2) * (coords[fr,1,ix]-coords[fr,0,ix])
            F[ix] = tmp
        np.testing.assert_allclose(F,forces[fr,1],rtol=1e-4)
    return dists


for ii in range(4):
    dist = compute_forces(positions[ii],forces[ii])
    bins = np.linspace(0,0.02,25)
    count,edges = np.histogram(dist,bins=bins)
    px_dist = count / np.sum(count)
    centers = (edges[1:]+edges[:-1])/2
    plt.plot(centers,px_dist,alpha=0.5,label=ii)
    energy = 0.5*big_k.value_in_unit(unit.kilocalorie_per_mole/unit.angstrom**2) * centers**2
    beta = (1 / protocol['temperature'][ii] / unit.BOLTZMANN_CONSTANT_kB / 6.022e23).value_in_unit(unit.kilocalorie**-1)
    px = np.exp(-beta*energy)
    px = px / np.sum(px)
    plt.plot(centers,px,label='{} exact'.format(ii))
plt.legend()
plt.show()
x = 5