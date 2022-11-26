import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit
import os
from openmm.unit import dalton, kelvin, second, nanometer, kilojoule_per_mole
import os
import os.path as osp

# System creation
n_atoms = 2
mass = 1.0*dalton
temperature = 2.405*kelvin
timestep = 1e-15*second # (fs)
gamma =  0.5 # (1/ps)
mass = 12.0*dalton
zero_mass = 0.0*dalton
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
platform = mm.Platform.getPlatformByName("CUDA")
platform_properties = {"DeviceIndex": "0", "Precision": "mixed"}

n_replicas = 1
T = 2.405 * unit.kelvin

# Build topology
topo = app.Topology()
topo.addChain()
chain = list(topo.chains())[0]
topo.addResidue('C1',chain)
residue = list(topo.residues())[0]
element = app.Element.getBySymbol('C')
for atom in range(n_atoms):
    topo.addAtom('C',element,residue)

simulation = mm.app.Simulation(topo, system, integrator, platform, platform_properties)
# start_pos = unit.Quantity(np.zeros([0,3], np.float), unit.nanometers)
# simulation.context.setPositions(start_pos)
simulation.step(0)
forces = simulation.context.getState(getForces=True).getForces(asNumpy=True)