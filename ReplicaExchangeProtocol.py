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

    def run(self, 
            n_iterations:int = 1, 
            n_attempts:int = 1, 
            md_timesteps:int =1,
            equilibration_timesteps:int =0,  
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
        md_timesteps:
            how many md timesteps propagate the system for each iteration. 
        equilibraton_timesteps:
            how long is the equilibration phase.       
        save:
            if set to True position and forces are saved every save_interval steps during 
            the production phase. Position and forces are returned as numpy array.
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

        Return
        ------
            If save is set to True position, forces, acceptance_matrix are returned.
            If save is set to False acceptance_matrix only is returned
        """
        self.positions = []
        self.forces = []
        self.acceptance_matrix = np.zeros((self.n_replicas, self.n_replicas))
        if mixing == 'neighbors':
            self._define_neighbors()
        for iteration in range(n_iterations):
            ## Propagate dynamics
            self._propagate_replicas(md_timesteps=md_timesteps, equilibration_timesteps=equilibration_timesteps, 
                                        save=save, save_interval=save_interval)
            ## Mix replicas
            self._mix_replicas(mixing=mixing, n_attempts=n_attempts)

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

    def _propagate_replicas(self, 
                            md_timesteps:int = 1, 
                            equilibration_timesteps:int = 0, 
                            save:bool = False, 
                            save_interval:int = 1):
        """
        Apply _mcmc_move to all the replicas md_timesteps times. If equilibration_timesteps > 0,
        an equilibration phase is considered before try to save position and forces.
        If save is set to True position and forces are saved every save_interval time.
        _thermodynamic_state[i] is associated to the replica configuration in _replicas_sampler_states[i].
        """
        for md_step in range(md_timesteps):
            for thermo_state, sampler_state in zip(self._thermodynamic_states, self._replicas_sampler_states):
                self._mcmc_move.apply(thermo_state, sampler_state)
            if md_step > equilibration_timesteps:
                    if save and (md_step % save_interval == 0):
                        self.positions.append([])
                        self.forces.append([])
                        self.positions[-1], self.forces[-1] = self._grab_forces()


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
        # If more than several swaps more efficient to compute energies first
        if n_attempts > self.n_replicas:
            energy_matrix = self._compute_reduced_potential_matrix()
        else:
            energy_matrix = None

        # Attempts to exchnge n_attempts times 
        for attempt in range(n_attempts):
            if mixing == 'neighbors':
                i,j = np.sort(self.couples[np.random.randint(len(self.couples))])
            elif mixing == 'all':
                i,j = np.sort(np.random.choice(range(len(self._thermodynamic_states)),2,replace=False))
            self.acceptance_matrix[j,i] += 1

            # Compute the energies.
            if energy_matrix is None:
                sampler_state_i, sampler_state_j = (self._replicas_sampler_states[k] for k in [i, j])
                thermo_state_i, thermo_state_j = (self._thermodynamic_states[k] for k in [i, j])
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
                self._thermodynamic_states[i] , self._thermodynamic_states[j] = self._thermodynamic_states[j], self._thermodynamic_states[i]
                # Swap i and j row in reduced_potential_matrix
                energy_matrix[[i, j]] = energy_matrix[[j, i]]
                self.acceptance_matrix[i,j] += 1
            
            # Reset velocities 
            if self.rescale_velocities == True:
               self._rescale_velocities()


    def _compute_reduced_potential_matrix(self):
        # Compute the reduced potential matrix between all possible couples
        energy_matrix = np.zeros((self.n_replicas,self.n_replicas))
        for i,thermo_state in enumerate(self._thermodynamic_states):
                for j,sampler_state in enumerate(self._replicas_sampler_states):
                    energy_matrix[j,i] = self._compute_reduced_potential(sampler_state, thermo_state)
        return energy_matrix

    def _rescale_velocities(self):
        '''
            Put velocities to desired temperature
        '''
        for thermo_state, sampler_state in zip(self._thermodynamic_states, self._replicas_sampler_states):
            context, _ = cache.global_context_cache.get_context(thermo_state)
            sampler_state.apply_to_context(context)
            context.setVelocitiesToTemperature(thermo_state.temperature)


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