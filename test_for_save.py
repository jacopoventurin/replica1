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
n_replicas = 10
min_T = 1
max_T = 10000
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
#test_positions = np.zeros((2,n_replicas,2,3))
md_step = 60
test_positions = np.zeros((2,n_replicas,md_step,2,3))


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
reporter = ReplicaStateReporter('state.csv', reportInterval=1, time=True, potentialEnergy=True,
                                kineticEnergy=True, totalEnergy=True, volume=True, speed=True, elapsedTime=True)

parallel_tempering.load_reporter(reporter)

# Run symulation and save position and forces every 100 timesteps

sim_params ={
    'n_iterations' : 100,
    'n_attempts': 23, #int(n_replicas*np.log(n_replicas)), 
    'md_timesteps': md_step, 
    'equilibration_timesteps': 0, 
    'save': True, 
    'save_interval': 1, 
    'checkpoint_simulations': False, 
    'mixing': 'all',  
    'save_atoms': 'all',   
    'reshape_for_TICA': 'False', 
}


print('-' * 50)
start = time.time()
print(f'Simulation started at time {time.ctime()}')
print('-' * 50)

positions, forces, acceptance_matrix = parallel_tempering.run(**sim_params)  
    
end = time.time()
print(f'Simulation ended at {time.ctime()}')
print(f'Total simulation time {end-start}')

temperature_history = parallel_tempering.get_temperature_history()

parallel_tempering.save_temperature_history(filename=f'temp_rank{rank}.npy')

print(f'Rank: {rank} --- Position shape: {np.array(positions).shape} --- Acceptance Matrix shape: {np.array(acceptance_matrix).shape} --- temperature_history shape: {np.array(temperature_history).shape}')



if rank == 0:
    np.save('positions.npy',positions)
    np.save('temperature_history.npy', temperature_history)
    np.save('acceptance', acceptance_matrix)
    positions = np.array(positions)

