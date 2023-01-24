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
from Reporters import ReplicaStateReporter
import time
from mpi4py import MPI
np.random.seed(12345)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

system = mm.System()
system.addParticle(2.0)
system.addParticle(0.0)
dt = 0.01
harmonic_force = mm.HarmonicBondForce()
r0 = 1.75
k = 1
harmonic_force.addBond(0,1,r0,k)
system.addForce(harmonic_force)
start_positions = np.array(((-1,0,0),(1,0,0)),dtype=np.float64)

constraint_tolerance = 1e-20
random_number_seed = 12345

class VerletMove(openmmtools.mcmc.BaseIntegratorMove):
    def __init__(self, timestep, n_steps, **kwargs):
        super(VerletMove, self).__init__(n_steps, **kwargs)
        self.timestep = timestep
        self.constraint_tolerance = 1e-20
    def _get_integrator(self, thermodynamic_state):
        integrator = mm.VerletIntegrator(self.timestep)
        integrator.setConstraintTolerance(self.constraint_tolerance)
        return integrator
    def _before_integration(self, context, thermodynamic_state):
        # pass
        # print('Setting velocities')
        context.setVelocitiesToTemperature(thermodynamic_state.temperature,random_number_seed)
        # context.setVelocities(np.zeros((2,3)))
    def _after_integration(self, context, thermodynamic_state):
        pass
        # print('Reading statistics')
verlet_move = VerletMove(timestep=1*unit.femtosecond, n_steps=1)

# Replica setup
n_replicas = 3
min_T = 1
max_T = 10
temperatures = np.geomspace(min_T, max_T, n_replicas)

## Run propagation outside of class for comparison
# Build topology
topo = app.Topology()
topo.addChain()
chain = list(topo.chains())[0]
topo.addResidue('C1',chain)
residue = list(topo.residues())[0]
element = app.Element.getBySymbol('C')
for atom in range(2):
    topo.addAtom('C',element,residue)

simulations = []
integrators = []
test_positions = np.zeros((2,n_replicas,2,3))
md_step = 1

for i_t,temperature in enumerate(temperatures):
    integrators.append(mm.VerletIntegrator(1*unit.femtosecond))
    integrators[-1].setConstraintTolerance(constraint_tolerance)
    platform = mm.Platform.getPlatformByName("CUDA")
    platform_properties={"Precision": "mixed"}
    simulation = app.Simulation(topo,system,integrators[-1],platform=platform,platformProperties=platform_properties)
    simulations.append(simulation)
    simulation.context.setPositions(start_positions)
    # First step
    simulation.context.setVelocitiesToTemperature(temperature,random_number_seed)
    simulation.step(md_step)
    # simulations.append(simulation)
    test_positions[0,i_t] = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    
    # Second step (0 and 1 have swapped)
    rep = i_t
    if i_t == 0:
        simulation.context.setVelocitiesToTemperature(temperatures[1],random_number_seed)
        rep = 1
    elif i_t == 1:
        simulation.context.setVelocitiesToTemperature(temperatures[0],random_number_seed)
        rep = 0
    else:
        simulation.context.setVelocitiesToTemperature(temperature,random_number_seed)

    simulation.step(md_step)
    test_positions[1,rep] = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)

protocol = {'temperature': temperatures * unit.kelvin}
thermodynamic_states = states.create_thermodynamic_state_protocol(system,protocol)

sampler_states = list()
for i_t,_ in enumerate(thermodynamic_states):
    sampler_states.append(openmmtools.states.SamplerState(positions=start_positions))

# Define class for replica exchange
parallel_tempering = ReplicaExchange(
    thermodynamic_states=thermodynamic_states, 
    sampler_states=sampler_states, 
    mcmc_move=verlet_move,
    rescale_velocities=True,
    save_temperatures_history=True
)

# Define and load reporter 
reporter = ReplicaStateReporter('state.csv', reportInterval=50, time=True, potentialEnergy=True,
                                kineticEnergy=True, totalEnergy=True, volume=True, speed=True, elapsedTime=True)

parallel_tempering.load_reporter(reporter)

# Run symulation and save position and forces every 100 timesteps

sim_params ={
    'n_iterations' : 2,
    'n_attempts': 3, #int(n_replicas*np.log(n_replicas)), 
    'md_timesteps': md_step, 
    'equilibration_timesteps': 0, 
    'save': True, 
    'save_interval': 1, 
    'checkpoint_simulations': True, 
    'mixing': 'all',  
    'save_atoms': 'all',   
    'reshape_for_TICA': 'False', 
}


print('-' * 50)
start = time.time()
print(f'Simulation started at time {time.ctime()}')
print('-' * 50)

positions, forces, acceptance_matrix, temperature_history = parallel_tempering.run(**sim_params)  
    
end = time.time()
print(f'Simulation ended at {time.ctime()}')
print(f'Total simulation time {end-start}')

if rank == 0:
    np.save('positions.npy',positions)

    for i_t,temperature in enumerate(temperatures):
        ## After first MD step
        np.testing.assert_allclose(positions[:,0],test_positions[0],rtol=1e-12)
        ## After swap and second MD step
        np.testing.assert_allclose(positions[:,1],test_positions[1],rtol=1e-12)

    print(test_positions[1],positions[:,1])
