import numpy as np
import mdtraj as md
from openmmtools import cache
import openmm.unit as unit
import openmm as mm
import mpiplus
from numba import njit

class ReplicaExchange:
    def __init__(
        self,
        thermodynamic_states=None,
        sampler_states=None,
        mcmc_move=None,
        rescale_velocities=False,
        save_temperatures_history=False

    ):
        """
        Class to perform Replica Exchange simulation.

        Parameters:
        ----------
        thermodynamic_states:
            should be a list of openmmtools.states.ThermodynamicState representing different 
            contexts of the simulation that doesn't changes during the integration. The full
            list can be obtained using openmmtools.states.create_thermodynamic_state_protocol
        sampler_states:
            should be a list of openmmtools.states.SamplerState representing different portion of 
            the system that changes during the integration.
        mcmc_move:
            should be openmmtools compatible move (e.g. openmmtools.mcmc.LangevinSplittingDynamicsMove)
        rescale_velocities:
            if set to True velocities are rescaled after every attempt to exchange replicas.
        save_temperatures_history:
            if set to True the history of exchnage is saved in a format (n_iteration, n_replicas).
            The history can be obtained by get_temperature_history() method.
        """
        self._thermodynamic_states = thermodynamic_states
        self._replicas_sampler_states = sampler_states
        self._mcmc_move = mcmc_move
        self.n_replicas = len(thermodynamic_states)

        self._topology = None
        self._dimension = None
        self._reporter = None

        self._temperature_list = [self._thermodynamic_states[i].temperature for i in range(self.n_replicas)]
        self.rescale_velocities = rescale_velocities
        if save_temperatures_history: 
            self._temperature_history = []
            temperature = [self._thermodynamic_states[i].temperature.value_in_unit(unit.kelvin) for i in range(self.n_replicas)]
            self._temperature_history.append(temperature)
        else:
            self._temperature_history = None   

    def run(self, 
            n_iterations:int = 1, 
            n_attempts:int = 1, 
            md_timesteps:int =1,
            equilibration_timesteps:int =0,  
            save:bool = False, 
            save_interval:int = 1, 
            checkpoint_simulations:bool = False, 
            mixing:str = 'all',
            save_atoms:str = 'all',
            reshape_for_TICA:bool = False):
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
        save_atoms:
            can be 'all' or any mdtraj compatible string. For example if set to 'all' positions and
            forces of all toms in the system are saved, while if set to 'protein' positions and forces 
            of protein's atoms only are saved.
        reshape_for_TICA:
            if set to True position and forces are returned with shape 
            (n_iteration, n_replicas, md_timesteps-equilibration_timesteps)/save_interval, any, 3),
            else (n_replicas, n_iteration*(md_timesteps-equilibration_timesteps)/save_interval, any, 3)
            is returned. Default is False

        Return
        ------
            If save is set to True position, forces, acceptance_matrix are returned.
            If save is set to False acceptance_matrix only is returned
        """
        self.positions = []
        self.forces = []
        self.acceptance_matrix = np.zeros((self.n_replicas, self.n_replicas))
        self._define_target_elements_and_dimension(save_atoms)

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
            positions = np.swapaxes(np.array(self.positions), 0, 1)
            forces = np.swapaxes(np.array(self.forces),0 , 1)
            if reshape_for_TICA:
                positions_list = np.split(positions, n_iterations, axis=1)
                forces_list = np.split(forces, n_iterations, axis=1)
                return positions_list, forces_list, self.acceptance_matrix
            else:    
                return positions, forces, self.acceptance_matrix
        else:
            return self.acceptance_matrix


    def load_topology(self, topology: md.Topology):
        """
        This method allow to add topology object to the class.
        Is necessary to specify a topology if only protein's atoms positions are saved during
        the simulation.
        """
        if self._topology is None:
            self._topology = topology
        else:
            raise AttributeError('Topology is already defined for this class')

    
    def load_reporter(self, reporter):
        """
        This method add a reporter object to the class, used to save some interesting variables 
        during the simulation (e.g. temperature, total energy ...)
        """
        if self._reporter is None:
            self._reporter = reporter
        else:
            raise AttributeError('Multiple reporters not jet supported for this class')
    
    def get_temperature_history(self, asNpy:bool = True):
        """
        Return temperature history with shape
        (n_iteration, n_replicas), where n_iterations is the number of 
        iterations specified in run.
        If asNpy is set to True temperature history is returned as npy array
        """
        if asNpy:
            return np.array(self._temperature_history)
        else:
            return self._temperature_history
    

    def save_checkpoint(self, code:str = None):
        """
        Save checkpoint in n_replicas number of file of the form 
        {code}_checkpoint-{idx}.xml, where idx is the index of a specific replica.
        If code is not specified the output files are of the form checkpoint-{idx}.xml
        """
        if code == None:
            checkpoint_string = 'checkpoint'
        else:
            checkpoint_string = f'{code}_checkpoint'

        for i_t,(thermo_state, sampler_state) in enumerate(zip(self._thermodynamic_states, self._replicas_sampler_states)):
            context, _ = cache.global_context_cache.get_context(thermo_state)
            sampler_state.apply_to_context(context)
            state = context.getState(getPositions=True, getVelocities=True, getParameters=True)
            state_xml = mm.XmlSerializer.serialize(state)
            with open(f'{checkpoint_string}-{i_t}.xml', 'w') as output:
                output.write(state_xml)

    def load_context_from_checkpoint(self, code:str = None):
        """
        Load context from a checkpoint files of the form {code}_checkpoint-{idx}.xml
        If code is None then context is loaded from checkpoint-{idx}.xml
        """
        if code == None:
            checkpoint_string = 'checkpoint'
        else:
            checkpoint_string = f'{code}_checkpoint'

        for i_t,(thermo_state, sampler_state) in enumerate(zip(self._thermodynamic_states, self._replicas_sampler_states)):
            context, _ = cache.global_context_cache.get_context(thermo_state)
            with open(f'{checkpoint_string}-{i_t}.xml', 'r') as input:
                state = mm.XmlSerializer.deserialize(input.read())
                sampler_state.positions = state.getPositions()
                sampler_state.velocities = state.getVelocities()
            sampler_state.apply_to_context(context)


    def _define_target_elements_and_dimension(self, save_atoms):
        """
        This method define target elements we whant to save during the simulation
        and define dimension of the system.
        """
        if save_atoms == 'all':
            self.target_elements = list(range(len(self._replicas_sampler_states[0].positions)))
        else:
            try:
                self.target_elements = self._topology.select(save_atoms)
            except:
                if self._topology is None:
                    print('Topology not loaded: need to be loaded using load_topology()')
                else:
                    print(f'Seems {save_atoms} is not compatible with mdtraj.topology.select()')
                print('Position and forces of all the atoms are saved')
                self.target_elements = list(range(len(self._replicas_sampler_states[0].positions)))           

        # get the dymension of the space  
        if self._dimension is None:
            self._dimension = self._replicas_sampler_states[0].positions.shape[-1]

    @njit
    def _propagate_replicas(self, 
                            md_timesteps:int = 1, 
                            equilibration_timesteps:int = 0, 
                            save:bool = False, 
                            save_interval:int = 1
                            ):
        """
        Apply _mcmc_move to all the replicas md_timesteps times. If equilibration_timesteps > 0,
        an equilibration phase is considered before try to save position and forces.
        If save is set to True position and forces are saved every save_interval time.
        _thermodynamic_state[i] is associated to the replica configuration in _replicas_sampler_states[i].
        If reporter was loaded, reports is also saved.

        Params:
        -------
        md_timesteps:
            number of MD timesteps, menas how many times apply self.mcmc_move to each state
        equilibration_timesteps:
            number of timesteps for equilibration. During equilibration position, forces and 
            state of the system are not saved
        save:
            if set to True position and forces are saved every save_interval timesteps
        save_interval:
            if save is set to True, position and forces are saved every save_interval timesteps 
        """
        for md_step in range(md_timesteps):
            # for thermo_state, sampler_state in zip(self._thermodynamic_states, self._replicas_sampler_states):
            #     self._mcmc_move.apply(thermo_state, sampler_state)

            mpiplus.distribute(self._run_replica, range(self.n_replicas), send_results_to=0)

            # verify if reporter was loaded
            if self._reporter is not None:
                report_interval = self._reporter.get_report_interval()
            if md_step > equilibration_timesteps:
                    if save and (md_step % save_interval == 0):
                        self.positions.append([])
                        self.forces.append([])
                        self.positions[-1], self.forces[-1] = self._grab_forces()
                    if self._reporter is not None:
                        if (md_step % report_interval) == 0:
                            self._reporter.report(self._thermodynamic_states, self._replicas_sampler_states)
                    

    def _run_replica(self, replica_id):
        self._mcmc_move.apply(self._thermodynamic_states[replica_id], self._replicas_sampler_states[replica_id])
        
    @njit 
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
            self._compute_reduced_potential_matrix()
        else:
            self.energy_matrix = None

        premix_temperatures = []
        for i_t,thermo_state in enumerate(self._thermodynamic_states):
            premix_temperatures.append(thermo_state.temperature)
        

        # Attempts to exchnge n_attempts times 
        for attempt in range(n_attempts):
            if mixing == 'neighbors':
                i,j = np.sort(self.couples[np.random.randint(len(self.couples))])
            elif mixing == 'all':
                i,j = np.sort(np.random.choice(range(len(self._thermodynamic_states)),2,replace=False))
            self.acceptance_matrix[j,i] += 1

            # Compute the energies.
            if self.energy_matrix is None:
                sampler_state_i, sampler_state_j = (self._replicas_sampler_states[k] for k in [i, j])
                thermo_state_i, thermo_state_j = (self._thermodynamic_states[k] for k in [i, j])
                energy_ii = self._compute_reduced_potential(sampler_state_i, thermo_state_i)
                energy_jj = self._compute_reduced_potential(sampler_state_j, thermo_state_j)
                energy_ij = self._compute_reduced_potential(sampler_state_i, thermo_state_j)
                energy_ji = self._compute_reduced_potential(sampler_state_j, thermo_state_i)
            else:
                energy_ii,energy_jj = self.energy_matrix[i,i], self.energy_matrix[j,j]
                energy_ij,energy_ji = self.energy_matrix[i,j], self.energy_matrix[j,i]

            # Accept or reject the swap.
            log_p_accept = -(energy_ij + energy_ji) + energy_ii + energy_jj
            if random_number_list[attempt] < np.exp(log_p_accept):
                # Swap states in replica slots i and j.
                self._thermodynamic_states[i] , self._thermodynamic_states[j] = self._thermodynamic_states[j], self._thermodynamic_states[i]
                self.acceptance_matrix[i,j] += 1
                if self.energy_matrix is not None:
                    # Swap i and j row in reduced_potential_matrix
                    self.energy_matrix[[i, j]] = self.energy_matrix[[j, i]]

            # Update temperature history after swap
            if self._temperature_history is not None:
                temperatures = [self._thermodynamic_states[i].temperature.value_in_unit(unit.kelvin) for i in range(self.n_replicas)]
                self._temperature_history.append(temperatures)
    

            # Reset velocities 
            if self.rescale_velocities == True:
               self._rescale_velocities(premix_temperatures)


    def _compute_reduced_potential_matrix(self):
        # Compute the reduced potential matrix between all possible couples
        self.energy_matrix = np.zeros((self.n_replicas,self.n_replicas))
        for i,thermo_state in enumerate(self._thermodynamic_states):
                for j,sampler_state in enumerate(self._replicas_sampler_states):
                    self.energy_matrix[j,i] = self._compute_reduced_potential(sampler_state, thermo_state)


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
        """
        Return position and forces of the target elements as np.array
        """
        forces = np.zeros((self.n_replicas,len(self.target_elements), self._dimension))
        positions = np.zeros(forces.shape)
        for thermo_state, sampler_state in zip(
            self._thermodynamic_states, self._replicas_sampler_states
        ):
            context, _ = cache.global_context_cache.get_context(thermo_state)
            sampler_state.apply_to_context(context)
            #
            i_t = self._temperature_list.index(thermo_state.temperature)
            position = context.getState(getPositions=True).getPositions(asNumpy=True)[self.target_elements]
            position = position.value_in_unit(unit.angstrom)
            force = context.getState(getForces=True).getForces(asNumpy=True)[self.target_elements]
            force = force.value_in_unit(unit.kilocalorie_per_mole/unit.angstrom)
            positions[i_t] = position
            forces[i_t] = force
        return positions, forces

    def _save_contexts(self):
        for i_t, (thermo_state, sampler_state) in enumerate(
            zip(self._thermodynamic_states, self._replicas_sampler_states)
        ):
            context, _ = cache.global_context_cache.get_context(thermo_state)
            sampler_state.apply_to_context(context)
            state = context.getState(
                getPositions=True, getVelocities=True, getParameters=True
            )
            state_xml = mm.XmlSerializer.serialize(state)
            with open("checkpoint-{}.xml".format(i_t), "w") as output:
                output.write(state_xml)

    def _load_contexts(self):
        for i_t, (thermo_state, sampler_state) in enumerate(
            zip(self._thermodynamic_states, self._replicas_sampler_states)
        ):
            context, _ = cache.global_context_cache.get_context(thermo_state)
            with open("checkpoint-{}.xml".format(i_t), "r") as input:
                state = mm.XmlSerializer.deserialize(input.read())
                sampler_state.positions = state.getPositions()
                sampler_state.velocities = state.getVelocities()
            sampler_state.apply_to_context(context)


    def _rescale_velocities(self,original_temperature=None):
        '''
            Rescale velocities to desired temperature from thermodynamic state after swap attempt
        '''
        for i_t,(thermo_state, sampler_state) in enumerate(zip(self._thermodynamic_states, self._replicas_sampler_states)):
            context, _ = cache.global_context_cache.get_context(thermo_state)
            sampler_state.apply_to_context(context)
            if original_temperature is not None:
                context.setVelocities(context.getState(getVelocities=True).getVelocities()*np.sqrt(thermo_state.temperature/original_temperature[i_t]))
            else:
                context.setVelocitiesToTemperature(thermo_state.temperature)
