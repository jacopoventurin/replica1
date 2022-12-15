import numpy as np
from openmmtools import cache
import openmm.unit as unit
import openmm as mm

class ReplicaExchange:

    def __init__(
        self, 
        thermodynamic_states=None,
        sampler_states=None,
        mcmc_move=None,
        rescale_velocities=False,
    ):
        self._thermodynamic_states = thermodynamic_states
        self._replicas_sampler_states = sampler_states
        self._mcmc_move = mcmc_move
        self.n_replicas = len(thermodynamic_states)
        self._temperature_list = [self._thermodynamic_states[i].temperature for i in range(self.n_replicas)]
        self.rescale_velocities = rescale_velocities

    def run(
        self, 
        n_iterations=1,
        n_md_steps=100, 
        n_swap_attempts_per_iteration=1,
        equilibration_period=1,
        collection_interval=1,
        save=False, 
        checkpoint_simulations=False,

    ):
        '''
            There are three different timescales here
            1.) Simulation timestep, how long each MD step is taken for
            2.) Replica swaps, how often to attempt replica exchange
            3.) Output saving, how frequently to save coords/forces
        '''

        self.positions = []
        self.forces = []
        self.acceptance_matrix = np.zeros((self.n_replicas, self.n_replicas))

        for iteration in range(n_iterations):
            # if iteration % 10 == 0: print('iteration',iteration)

            ## Propagate dynamics
            for md_step in range(n_md_steps):
                # if md_step % 100 == 0: print('md_step',md_step)

                self._propagate_replicas()
                if md_step > equilibration_period:
                    if save and (md_step % collection_interval == 0):
                        self.positions.append([])
                        self.forces.append([])
                        self.positions[-1], self.forces[-1] = self._grab_forces()

            ## Replica attempts
            for _ in range(n_swap_attempts_per_iteration):
                self._mix_replicas()

            if checkpoint_simulations:
                self._save_contexts()

        if save:
            ## Output in the format shape
            # replica, frames, n_atoms, xyz
            positions = np.swapaxes(np.array(self.positions),0,1)
            forces = np.swapaxes(np.array(self.forces),0,1)
            return positions,forces,self.acceptance_matrix
        else:
            return _,_,self.acceptance_matrix
            
    def _propagate_replicas(self):
        # _thermodynamic_state[i] is associated to the replica configuration in _replicas_sampler_states[i].
        for thermo_state, sampler_state in zip(self._thermodynamic_states, self._replicas_sampler_states):
            self._mcmc_move.apply(thermo_state, sampler_state)

    def _mix_replicas(self, n_attempts=1):
        '''
            Attempt replica exchange swap.
                If many swap attempts compute energies a priori
                Rescale velocities if flag set
        '''
        # If more than several swaps more efficient to compute energies first
        energy_matrix = None
        if n_attempts > self.n_replicas:
            ## Each row is a state and each column is the temperature
            energy_matrix = np.zeros((self.n_replicas,self.n_replicas))
            for i,thermo_state in enumerate(self._thermodynamic_states):
                for j,sampler_state in enumerate(self._replicas_sampler_states):
                    energy_matrix[j,i] = self._compute_reduced_potential(sampler_state, thermo_state)

        # Attempt to switch two replicas at random. 
        random_number_list = np.random.rand(n_attempts)
        for attempt in range(n_attempts):
            # Select two replicas at random.
            i,j = np.sort(np.random.choice(range(len(self._thermodynamic_states)),2,replace=False))
            self.acceptance_matrix[j,i] += 1
            sampler_state_i, sampler_state_j = (self._replicas_sampler_states[k] for k in [i, j])
            thermo_state_i, thermo_state_j = (self._thermodynamic_states[k] for k in [i, j])

            # Compute the energies.
            if energy_matrix is None:
                energy_ii = self._compute_reduced_potential(sampler_state_i, thermo_state_i)
                energy_jj = self._compute_reduced_potential(sampler_state_j, thermo_state_j)
                energy_ij = self._compute_reduced_potential(sampler_state_i, thermo_state_j)
                energy_ji = self._compute_reduced_potential(sampler_state_j, thermo_state_i)
            else:
                energy_ii,energy_jj = energy_matrix[i,i], energy_matrix[j,j]
                energy_ij,energy_ji = energy_matrix[i,j], energy_matrix[j,i]

            # Accept or reject the swap.
            log_p_accept = -(energy_ij + energy_ji) + energy_ii + energy_jj
            if random_number_list[attempt] < np.exp(log_p_accept):
                # Swap states in replica slots i and j.
                self._thermodynamic_states[i] = thermo_state_j
                self._thermodynamic_states[j] = thermo_state_i
                self.acceptance_matrix[i,j] += 1

        # Reset velocities 
        if self.rescale_velocities == True:
            self._rescale_velocities()

    def _compute_reduced_potential(self, sampler_state, thermo_state):
        # Obtain a Context to compute the energy with OpenMM. Any integrator will do.
        context, _ = cache.global_context_cache.get_context(thermo_state)
        # Compute the reduced potential of the sampler_state configuration
        # in the given thermodynamic state.
        sampler_state.apply_to_context(context)
        return thermo_state.reduced_potential(context)

    def _rescale_velocities(self):
        '''
            Put velocities to desired temperature
        '''
        for thermo_state, sampler_state in zip(self._thermodynamic_states, self._replicas_sampler_states):
            context, _ = cache.global_context_cache.get_context(thermo_state)
            sampler_state.apply_to_context(context)
            context.setVelocitiesToTemperature(thermo_state.temperature)

    def _grab_forces(self):
        '''
            Grab positions and forces from context
        '''
        forces = np.zeros((self.n_replicas,*self._replicas_sampler_states[0].positions.shape))
        positions = np.zeros(forces.shape)
        for thermo_state, sampler_state in zip(self._thermodynamic_states, self._replicas_sampler_states):
            context, _ = cache.global_context_cache.get_context(thermo_state)
            sampler_state.apply_to_context(context)
            #
            i_t = self._temperature_list.index(thermo_state.temperature)
            position = context.getState(getPositions=True).getPositions(asNumpy=True)
            position = position.value_in_unit(unit.angstrom)
            force = context.getState(getForces=True).getForces(asNumpy=True)
            force = force.value_in_unit(unit.kilocalorie_per_mole/unit.angstrom)
            positions[i_t] = position
            forces[i_t] = force
        return positions, forces

    def _save_contexts(self):
        for i_t,(thermo_state, sampler_state) in enumerate(zip(self._thermodynamic_states, self._replicas_sampler_states)):
            context, _ = cache.global_context_cache.get_context(thermo_state)
            sampler_state.apply_to_context(context)
            state = context.getState(getPositions=True, getVelocities=True, getParameters=True)
            state_xml = mm.XmlSerializer.serialize(state)
            with open('checkpoint-{}.xml'.format(i_t), 'w') as output:
                output.write(state_xml)

    def _load_contexts(self):
        for i_t,(thermo_state, sampler_state) in enumerate(zip(self._thermodynamic_states, self._replicas_sampler_states)):
            context, _ = cache.global_context_cache.get_context(thermo_state)
            with open('checkpoint-{}.xml'.format(i_t), 'r') as input:
                state = mm.XmlSerializer.deserialize(input.read())
                sampler_state.positions = state.getPositions()
                sampler_state.velocities = state.getVelocities()
            sampler_state.apply_to_context(context)


            
            


            