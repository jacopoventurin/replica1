import numpy as np
from openmmtools import cache
import openmm.unit as unit

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
            n_iterations=1, 
            save=False, 
            save_interval:int = 1, 
            forces_path:str = None, 
            coords_path:str = None,
            mode:str = 'all'):

        """
        Perform integration of the all system, and attempt to exchange 
        replicas.

        Params
        ------
        n_iteration:
            total number of timesteps for the evolution of the system
        save:
            if set to True position, forces and acceptance matrix are saved, default is False
        save_interval:
            length of position and forces array saved, default is 100.
            Must be a multiple of n_interaction
        coords_path:
            path where to save forces
        forces_path: 
            path where to save coords
        mode: 
            can be 'all', then exchange between any possible couple 
            of replicas is attempted, or 'neighbors', then exchange 
            between replicaswith neighbors temperatures only 
            is attempted
        Return
        ------
            acceptance matrix
        """

        if n_iterations % save_interval != 0:
            raise ValueError(f"save_interval must be intere multipole of n_iterations, you set {save_interval} and {n_iterations}")

        if forces_path is None:
            forces_path = 'forces_'
        else:
            forces_path = forces_path + '/forces'
        if coords_path is None:
            coords_path = 'coords_'
        else:
            coords_path = coords_path + '/coords'

        positions = []
        forces = []
        self.acceptance_matrix = np.zeros((self.n_replicas, self.n_replicas))
        count = 0

        print('Start symulation')
        for iteration in range(n_iterations):               
            self._propagate_replicas()
            self._mix_replicas(mode = mode)

            # Save positions and forces
            if save:
                positions.append([])
                forces.append([])
                pos, forc = self._grab_forces()
                positions[-1] = pos
                forces[-1] = forc

            if (iteration+1) % save_interval == 0: 
                print(f'Interaction {iteration} of {n_iterations}')
                if save:    
                    np.save(f'{coords_path}_{count}.npy',np.swapaxes(np.array(positions),0,1))
                    np.save(f'{forces_path}_{count}.npy',np.swapaxes(np.array(forces),0,1))
                    # Empty memory
                    positions = []
                    forces = []
                    count += 1
        
        return self.acceptance_matrix


    def _propagate_replicas(self):
        # _thermodynamic_state[i] is associated to the replica configuration in _replicas_sampler_states[i].
        for thermo_state, sampler_state in zip(self._thermodynamic_states, self._replicas_sampler_states):
            self._mcmc_move.apply(thermo_state, sampler_state)

    def _mix_replicas(self, n_attempts:int =1, mode:str = 'all'):
        """
        Attempt to switch two replicas at random or a couple of 
        adiacent temperature replicas. 
        Obviously, this scheme can be improved.
        Params:
        ------
        n_attempts:
            number of attempts to exchange a couple of replicas
        mode:
            can be 'all', then exchange between any possible couple 
            of replicas is attempted, or 'neighbors', then exchange 
            between replicaswith neighbors temperatures only 
            is attempted
        """

        random_number_list = np.random.rand(n_attempts)

        for attempt in range(n_attempts):
            # Select two replicas at random.
            if mode == 'all':
                i,j = np.sort(np.random.choice(range(len(self._thermodynamic_states)),2,replace=False))
            elif mode == 'neighbors':
                i,j = self._adiacent()
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

    def _adiacent(self):
        """
        Find random indeces of adiacent replicas
        """
        temps = len(self._thermodynamic_states)-1
        i = np.random.randint(0, temps)
        j = i + 1
        k = i - 1

        if j  > temps:
            return i, k
        elif k < 0:
            return i, j
        else:
            jk = np.random.choice([j,k],1)
            return i, jk

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

            