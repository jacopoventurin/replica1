import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit
import pandas as pd
from openmm.openmm import NonbondedForce
import os
import openmmtools
from openmm.unit import kilocalorie_per_mole, angstrom
import mdtraj as md
import matplotlib.pyplot as plt

forcefield=['amber99sbildn.xml','tip3p.xml']
constraints = app.HBonds
nonbonded_cutoff = 0.9*unit.nanometer
switch_distance = 0.75*unit.nanometer
nonbonded_method = app.PME
implicit_solvent = False
solvated = True
hydrogen_mass = None

friction = 0.1 / unit.picoseconds
pressure = 1.0 * unit.atmosphere
temperature = 280*unit.kelvin
timestep = 2.0 * unit.femtosecond

pdb = app.PDBFile('../chi_sys.pdb')
traj = md.load('../chi_sys.pdb')

modeller = app.modeller.Modeller(pdb.topology, pdb.positions)


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

system = ff.createSystem(modeller.topology, **create_system_kwargs)

integrator = mm.LangevinMiddleIntegrator(temperature , friction , timestep)
integrator.setConstraintTolerance(1e-7)

platform = mm.Platform.getPlatformByName("CUDA")
platform_properties = {"DeviceIndex": "0", "Precision": "mixed"}


simulation = mm.app.Simulation(modeller.topology, system, integrator, platform, platform_properties)
print(simulation)
simulation.context.setPositions(modeller.positions)
simulation.context.setVelocitiesToTemperature(280*unit.kelvin)

simulation.minimizeEnergy()
simulation.reporters.append(app.DCDReporter('output.dcd', 1000))
simulation.reporters.append(app.StateDataReporter('log.csv', 500, step=True,
        potentialEnergy=True, temperature=True, speed=True)) # Save every ps

simulation.step(20000) # run for 0.2 ns 





simulation.setTemperature(400)








# Plot results
#data = pd.read_csv("log.csv")

#data.plot(0, 1)
#plt.title('Energy over MD timesteps')
#plt.savefig('energy_without_rescaling')

#data.plot(0, 2)
#plt.title('Temperature over MD timesteps')
#plt.savefig('temperature_without_rescaling')
