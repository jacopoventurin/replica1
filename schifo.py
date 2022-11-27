import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit
from openmm.openmm import NonbondedForce
import os
import openmmtools
from openmm.unit import kilocalorie_per_mole, angstrom
import mdtraj as md

forcefield=['amber99sb.xml','tip3p.xml']
forcefield=['amber96.xml','tip3p.xml']
constraints = app.HBonds
nonbonded_cutoff = 0.9*unit.nanometer
switch_distance = 0.75*unit.nanometer
nonbonded_method = app.PME
implicit_solvent = False
solvated = True
hydrogen_mass = None

friction = 0.1 / unit.picoseconds
pressure = 1.0 * unit.atmosphere
temperature = 300*unit.kelvin
timestep = 2.0 * unit.femtosecond

#xsc = np.loadtxt('input.xsc',skiprows=1)
pdb = app.PDBFile('step3_input.pdb')
traj = md.load('step3_input.pdb')
xdist = traj.xyz[:,:,0].max() - traj.xyz[:,:,0].min()
ydist = traj.xyz[:,:,1].max() - traj.xyz[:,:,1].min()
zdist = traj.xyz[:,:,2].max() - traj.xyz[:,:,2].min()
#xdist = unit.Quantity(xdist,unit=unit.nanometers)
#ydist = unit.Quantity(ydist,unit=unit.nanometers)
#zdist = unit.Quantity(zdist,unit=unit.nanometers)

dimensions = np.zeros((3,3))
dimensions[0][0] = xdist
dimensions[1][1] = ydist
dimensions[2][2] = zdist

#v1 = mm.vec3.Vec3(xdist, ydist, zdist)*unit.nanometers
#v2 = mm.vec3.Vec3(0., ydist, zdist)*unit.nanometers
#z = mm.vec3.Vec3(0., 0., zdist)#*unit.nanometers


#pdb.topology.setUnitCellDimensions((xdist,ydist,zdist))
print("Max spans, x, y, z: ", xdist, ydist, zdist)
pdb.topology.setPeriodicBoxVectors(dimensions)
#prmtop = app.AmberPrmtopFile('structure.prmtop')

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
system = ff.createSystem(pdb.topology, **create_system_kwargs)
barostat = mm.MonteCarloBarostat(pressure, temperature)
force_id = system.addForce(barostat)

integrator = mm.LangevinMiddleIntegrator(temperature , friction , timestep)
integrator.setConstraintTolerance(1e-7)

platform = mm.Platform.getPlatformByName("CUDA")
platform_properties = {"DeviceIndex": "0", "Precision": "mixed"}
#platform = mm.Platform.getPlatformByName("CPU")
#platform_properties = {"DeviceIndex": "0", "Precision": "mixed"}


simulation = mm.app.Simulation(pdb.topology, system, integrator, platform, platform_properties)
#simulation = mm.app.Simulation(pdb.topology, system, integrator, platform)
print(simulation)
simulation.context.setPositions(pdb.positions)



#if prmtop.topology.getPeriodicBoxVectors() is not None:
#    simulation.context.setPeriodicBoxVectors(*prmtop.topology.getPeriodicBoxVectors())
#else:


# simulation.context.setPeriodicBoxVectors()

# simulation.minimizeEnergy()
simulation.reporters.append(app.DCDReporter('output.dcd', 1000))
simulation.reporters.append(app.StateDataReporter('log.csv', 5000, step=True,
        potentialEnergy=True, temperature=True, speed=True))

for sim in range(75):
    coords_list = []
    forces_list = []

    for ix in range(10000): 
        simulation.step(1000)
        coords_list.append([])
        forces_list.append([])
        coords = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        coords = coords.value_in_unit(angstrom)
        forces = simulation.context.getState(getForces=True).getForces(asNumpy=True)
        forces = forces.value_in_unit(kilocalorie_per_mole/angstrom)
        forces_list[-1].append(forces)
        coords_list[-1].append(coords)

    coords = np.concatenate(coords_list,axis=0)
    forces = np.concatenate(forces_list,axis=0)
    np.save('coords-{}-.npy'.format(sim),coords)
    np.save('forces-{}-.npy'.format(sim),forces)

    simulation.saveState('state.xml.BAK')