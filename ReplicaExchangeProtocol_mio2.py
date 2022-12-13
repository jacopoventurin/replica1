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
    ):
        self._thermodynamic_states = thermodynamic_states
        self._replicas_sampler_states = sampler_states
        self._mcmc_move = mcmc_move
        self.n_replicas = len(thermodynamic_states)
        self._temperature_list = [self._thermodynamic_states[i].temperature for i in range(self.n_replicas)]

    def run(self, 
            n_iterations:int = 1, 
            n_attempts:int = 1, 
            save:bool = False, 
            save_interval:int = 1, 
            checkpoint_simulations:bool = False, 
            mixing:str = 'all'):
        """
        Run a simulation with replica exchange protocol. Is possible to try exchange between 
        all replicas or just between neighbors.

        Params
        ------
        n_iteration:
            number of iteration to perform during the all symulation, default is 1.
        n_attempts:
            number of attempts to exchange a pair of replicas for each iteration, default is 1.
        save:
            if set to True position and forces are saved every save_interval steps.
            Position and forces are returned as numpy array.
        save_interval:
            interval for save position and forces, default is 1.
        checkpoint_simulations:
            if set to True a checkpoint configuration is saved every iteration.
        mixing:
            can be 'all' or 'neighbors'. If 'all' exchange between all possible replicas is attempted,
            while if 'neighbors' exchange between neighbors replicas only is attempted. If 'all' then 
            n_attempts should be at least n_replicas*log(n_replicas), where n_replicas is the total 
            number of replicas in the simulation. See https://aip.scitation.org/doi/10.1063/1.3660669
            for more information.
        """
        self.positions = []
        self.forces = []
        self.acceptance_matrix = np.zeros((self.n_replicas, self.n_replicas))
        if mixing == 'neighbors':
            self._define_neighbors()
        for iteration in range(n_iterations):
            if iteration % 100 == 0: print(iteration)
            self._propagate_replicas()
            self._mix_replicas(mixing=mixing, n_attempts=n_attempts)

            # Save positions and forces
            if save:
                if iteration % save_interval == 0:
                    self.positions.append([])
                    self.forces.append([])
                    positions, forces = self._grab_forces()
                    self.positions[-1] = positions
                    self.forces[-1] = forces

            if checkpoint_simulations:
                self._save_contexts()

        if save:
            ## Output in the format shape
            # replica, frames, n_atoms, xyz
            positions = np.swapaxes(np.array(self.positions),0,1)
            forces = np.swapaxes(np.array(self.forces),0,1)
            return positions,forces,self.acceptance_matrix
        else:
            return self.acceptance_matrix

    def _propagate_replicas(self):
        # _thermodynamic_state[i] is associated to the replica configuration in _replicas_sampler_states[i].
        for thermo_state, sampler_state in zip(self._thermodynamic_states, self._replicas_sampler_states):
            self._mcmc_move.apply(thermo_state, sampler_state)

    def _mix_replicas(self, mixing:str = 'all', n_attempts=1,):
        """
        Mix replicas using two possible strategy: try to exchange two replicas randomly choosen 
        between all possible states, or try to exchange a randomly choosen couple of neighbors of
        neighbor states only. If mixing = 'all' the matrix of energyes corresponding to all possible 
        configuration is pre computed in order to save time.
        
        Params:
        ------  
        mixing:
            can be 'all' or 'neighbors'. If 'all' exchange between all possible replicas is attempted,
            while if 'neighbors' exchange between neighbors replicas only is attempted. If 'all' then 
            n_attempts should be at least n_replicas*log(n_replicas), where n_replicas is the total 
            number of replicas in the simulation. See https://aip.scitation.org/doi/10.1063/1.3660669
            for more information.
        n_attempts:
            number of attempts to exchange a pair of replicas for each iteration, default is 1.
        """
        random_number_list = np.random.rand(n_attempts)
        if mixing == 'neighbors':
            for attempt in range(n_attempts):
                #select two random neighbors
                i,j = np.sort(self.couples[np.random.randint(len(self.couples))])
                self.acceptance_matrix[j,i] += 1
                sampler_state_i, sampler_state_j = (self._replicas_sampler_states[k] for k in [i, j])
                thermo_state_i, thermo_state_j = (self._thermodynamic_states[k] for k in [i, j])
    
                # Compute the energies.
                energy_ii = self._compute_reduced_potential(sampler_state_i, thermo_state_i)
                energy_jj = self._compute_reduced_potential(sampler_state_j, thermo_state_j)
                energy_ij = self._compute_reduced_potential(sampler_state_i, thermo_state_j)
                energy_ji = self._compute_reduced_potential(sampler_state_j, thermo_state_i)
    
                # Accept or reject the swap.
                log_p_accept = -(energy_ij + energy_ji) + energy_ii + energy_jj
                if random_number_list[attempt] < np.exp(log_p_accept):
                    # Swap states in replica slots i and j.
                    self._thermodynamic_states[i] = thermo_state_j
                    self._thermodynamic_states[j] = thermo_state_i
                    self.acceptance_matrix[i,j] += 1
        elif mixing == 'all':
            U = self._compute_reduced_pottential_matrix()
            for attempt in range(n_attempts):
                # Select two replicas at random
                i,j = np.sort(np.random.choice(range(len(self._thermodynamic_states)),2,replace=False))
                self.acceptance_matrix[j,i] += 1

                # Get precomputed energy from the energy matrix
                energy_ii = U[i,i]
                energy_jj = U[j,j]
                energy_ij = U[i,j]
                energy_ji = U[j,i]

                # Accept or reject the swap.
                log_p_accept = -(energy_ij + energy_ji) + energy_ii + energy_jj
                if random_number_list[attempt] < np.exp(log_p_accept):
                    # Swap states in replica slots i and j.
                    self._thermodynamic_states[i] , self._thermodynamic_states[j] = self._thermodynamic_states[j], self._thermodynamic_states[i]
                    # Swap i and j row in reduced_potential_matrix
                    U[[i, j]] = U[[j, i]]
                    self.acceptance_matrix[i,j] += 1


    def _compute_reduced_potential_matrix(self):
        # Compute the reduced potential matrix between all possible couples
        dimension = len(self._thermodynamic_states)
        U = np.zeros((dimension,dimension))
        for i in range(dimension):
            sampler_state_i = self._replicas_sampler_states[i]
            for j in range(dimension):
                thermo_state_j = self._thermodynamic_states[j]
                U[i,j] = self._compute_reduced_potential(sampler_state_i, thermo_state_j)
        return U



    def _define_neighbors(self):
        # Define all possible couples of neighbors
        l = np.arange(len(self._thermodynamic_states))
        couples = []
        for i in zip(l, l[1:]):
            couples.append(i)

        self.couples = np.array(couples)



    def _compute_reduced_potential(self, sampler_state, thermo_state):
        # Obtain a Context to compute the energy with OpenMM. Any integrator will do.
        context, _ = cache.global_context_cache.get_context(thermo_state)
        # Compute the reduced potential of the sampler_state configuration
        # in the given thermodynamic state.
        sampler_state.apply_to_context(context)
        return thermo_state.reduced_potential(context)

    def _grab_forces(self):
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