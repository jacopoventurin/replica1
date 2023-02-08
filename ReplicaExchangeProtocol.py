import numpy as np
import mdtraj as md
from openmmtools import cache
import openmm.unit as unit
import openmm as mm
import mpiplus
import os
from mpi4py import MPI
import logging
from time import time

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def get_available_gpus():
    """
    See how many gpu available to use
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        result = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    else:
        try:
            import torch
            torch.cuda.is_available()
            result = list(range(torch.cuda.device_count()))
        except:
            raise RuntimeError("Unable to count GPU devices")
    return len(result)


class ReplicaExchange:
    def __init__(
        self,
        thermodynamic_states=None,
        sampler_states=None,
        mcmc_move=None,
        rescale_velocities=False,
        save_temperatures_history=False,
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
            velocities are rescaled after every attempt to exchange replicas according to sqrt(Tnew/Told).
        save_temperatures_history:
            the history of exchange is saved in a format (n_iteration, n_replicas).
                obtained by get_temperature_history() method.
        """
        self._thermodynamic_states = thermodynamic_states
        self._replicas_sampler_states = sampler_states
        self._mcmc_move = mcmc_move
        self.n_replicas = len(thermodynamic_states)

        self._topology = None
        self._dimension = None
        self._reporter = None

        self._temperature_list = [
            self._thermodynamic_states[i].temperature for i in range(self.n_replicas)
        ]
        self.rescale_velocities = rescale_velocities
        if save_temperatures_history:
            self._temperature_history = []
            self._temperature_history.append(
                [tl.value_in_unit(unit.kelvin) for tl in self._temperature_list]
            )
        else:
            self._temperature_history = None

        platform = mm.Platform.getPlatformByName("CUDA")
        self.n_gpus = get_available_gpus()
        self._local_cache = []
        for i_gpu in range(self.n_gpus):
            self._local_cache.append(
                cache.ContextCache(
                    capacity=None,
                    time_to_live=None,
                    platform=platform,
                    platform_properties={
                        "DeviceIndex": "{}".format(i_gpu),
                        "Precision": "mixed",
                    },
                )
            )

    def run(
        self,
        n_iterations: int = 1,
        n_attempts: int = 1,
        md_timesteps: int = 1,
        equilibration_timesteps: int = 0,
        save: bool = False,
        save_interval: int = 1,
        checkpoint_simulations: bool = False,
        mixing: str = "all",
        save_atoms: str = "all",
        reshape_for_TICA: bool = False,
    ):
        """
        Run replica exchange protocol.
            Propagate dynamics
            Attempt Replica Switch
            Save coordinates

        Params
        ------
        n_iteration:
            number of times to perform the first two steps listed above
        n_attempts:
            number of attempts to exchange a pair of replicas for each iteration
        md_timesteps:
            total number of md timesteps to propagate the system
        equilibraton_timesteps:
            timelength in which to not save coordinates as system is equilibrating
        save:
            position and forces are saved every save_interval steps during
                the production phase. Position and forces are returned as numpy array.
        save_interval:
            interval for save position and forces
            total number of points saved are thus
                (md_timesteps - equilibration_timesteps)/save_interval
        checkpoint_simulations:
            a checkpoint configuration is saved every iteration.
        mixing:
            can be 'all' or 'neighbors'. If 'all' exchange between all possible replicas is attempted,
            while if 'neighbors' exchange between neighbors replicas only is attempted. If 'all' then
            n_attempts should be at least n_replicas*log(n_replicas), where n_replicas is the total
            number of replicas in the simulation. See https://aip.scitation.org/doi/10.1063/1.3660669
            for more information.
        save_atoms:
            any mdtraj compatible string
            if set to 'all' positions and forces for all atoms in the system are saved
        reshape_for_TICA:
            position and forces are returned with shape
                (n_iteration, n_replicas, md_timesteps-equilibration_timesteps)/save_interval, any, any)

        Return
        ------
            positions:
            forces:
            acceptance_matrix:
                upper half is number of successful swaps between thermodynamic state i and j
                lower half is number of attempted swaps between thermodynamic state i and j
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        self.positions = []
        self.forces = []
        self.energies = []
        self.acceptance_matrix = np.zeros((self.n_replicas, self.n_replicas))
        self._define_target_elements_and_dimension(save_atoms)

        if mixing == "neighbors":
            self._define_neighbors()

        for i_t in range(n_iterations):
            ## Propagate dynamics
            start = time()
            self._propagate_replicas(
                md_timesteps=md_timesteps,
                equilibration_timesteps=equilibration_timesteps,
                save=save,
                save_interval=save_interval,
            )

            end = time()
            if rank == 0:
                logger.warning(f"Propagate replicas iter:{i_t} took {end-start} [sec]")

            ## Energy matrix computation
            start = time()
            if n_attempts > self.n_replicas:
                self._compute_reduced_potential_matrix()
            else:
                self.energy_matrix = None

            end = time()
            if rank == 0:
                logger.debug(f"Energy matrix computation took {end-start} [sec]")

            ## Mix replicas
            start = time()
            self._thermodynamic_states = self._mix_replicas(
                mixing=mixing, n_attempts=n_attempts
            )
            end = time()
            if rank == 0:
                logger.warning(f"Mix replicas iter: {i_t} took {end-start} [sec]")
                for ii in range(self.n_replicas):
                    logger.debug(
                        f"Check position post-mix replica {ii} {self._grab_forces()[0][ii][0]}"
                    )

        if rank == 0:
            if checkpoint_simulations:
                logger.debug(f"checkpoint_simulations")
                self._save_context_checkpoints()
            
            if self._reporter is not None:
                # Save report to file 
                self._reporter.report()

            if save:
                logger.debug(f"save")
                ## Output in the format shape
                #    replica, frames, n_atoms, xyz
                #  Note that the final position will not be at the final swapped temperature
                positions = np.swapaxes(np.array(self.positions), 0, 1)
                forces = np.swapaxes(np.array(self.forces), 0, 1)
                energies = np.swapaxes(np.array(self.energies), 0, 1)
                if reshape_for_TICA:
                    # if reshape_for_TICA output is in the format shape 
                    # n_iteration, n_replicas, md_timesteps-equilibration_timesteps)/save_interval, any, any
                    positions = np.split(positions, n_iterations, axis=1)
                    forces = np.split(forces, n_iterations, axis=1)
                #return positions_list, forces_list, self.acceptance_matrix

                return (
                    positions,
                    forces,
                    energies,
                    self.acceptance_matrix,
                )
            else:
                return None, None, None, self.acceptance_matrix
        else:
            return None, None, None, None

    def load_topology(self, topology: md.Topology):
        """
        This method allow to add topology object to the class.
        Is necessary to specify a topology if only protein's atoms positions are saved during
        the simulation.
        """
        if self._topology is None:
            self._topology = topology
        else:
            raise AttributeError("Topology is already defined for this class")

    def load_reporter(self, reporter):
        """
        This method add a reporter object to the class, used to save some interesting variables
        during the simulation (e.g. temperature, total energy ...)
        """
        if self._reporter is None:
            self._reporter = reporter
        else:
            raise AttributeError("Multiple reporters not jet supported for this class")

    def save_temperature_history(self, filename: str = "temperature_history.npy", asNpy: bool = True):
        """
        Save temperature_history in a file. If asNpy = True then temperature_history 
        is automatically saved as .npy array. Only the 0 rank process saves.
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank ==0:
            if asNpy:
                np.save(filename, self._temperature_history)
            else:
                with open(filename, "w") as f:
                    f.write(self._temperature_history)
        else:
            pass

    def get_temperature_history(self, asNpy: bool = True):
        """
        Return temperature history with shape
        (n_iteration, n_replicas), where n_iterations is the number of
        iterations specified in run.
        If asNpy is set to True temperature history is returned as npy array
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            if asNpy:
                return np.array(self._temperature_history)
            else:
                return self._temperature_history
        else:
            return None

    def _save_context_checkpoints(self, filename: str = "checkpoint"):
        """
        Save checkpoint in n_replicas number of file of the form
        {filename}-{i_t}.xml, where i_t is the replica at index of temperature.
        """

        checkpoint_string = f"{filename}"

        for (thermo_state, sampler_state) in zip(
            self._thermodynamic_states, self._replicas_sampler_states
        ):
            i_t = self._temperature_list.index(thermo_state.temperature)
            context = self._get_context(i_t, thermo_state)
            sampler_state.apply_to_context(context)
            state = context.getState(
                getPositions=True, getVelocities=True, getParameters=True
            )
            state_xml = mm.XmlSerializer.serialize(state)
            with open(f"{checkpoint_string}-{i_t}.xml", "w") as output:
                output.write(state_xml)

    def _load_context_checkpoints(self, filename: str = "checkpoint"):
        """
        Load context from a checkpoint files of the form {filename}-{idx}.xml
        """

        checkpoint_string = f"{filename}"

        for idx, (thermo_state, sampler_state) in enumerate(
            zip(self._thermodynamic_states, self._replicas_sampler_states)
        ):
            context = self._get_context(idx, thermo_state)
            with open(f"{checkpoint_string}-{idx}.xml", "r") as input:
                state = mm.XmlSerializer.deserialize(input.read())
                sampler_state.positions = state.getPositions()
                sampler_state.velocities = state.getVelocities()
            sampler_state.apply_to_context(context)

    def _define_target_elements_and_dimension(self, save_atoms):
        """
        This method define target elements we whant to save during the simulation
        and define dimension of the system.
        """
        if save_atoms == "all":
            self.target_elements = list(
                range(len(self._replicas_sampler_states[0].positions))
            )
        else:
            try:
                self.target_elements = self._topology.select(save_atoms)
            except:
                if self._topology is None:
                    print(
                        "Topology not loaded: need to be loaded using load_topology()"
                    )
                else:
                    print(
                        f"Seems {save_atoms} is not compatible with mdtraj.topology.select()"
                    )
                print("Position and forces of all the atoms are saved")
                self.target_elements = list(
                    range(len(self._replicas_sampler_states[0].positions))
                )

        # get the dymension of the space
        if self._dimension is None:
            self._dimension = self._replicas_sampler_states[0].positions.shape[-1]

    def _propagate_replicas(
        self,
        md_timesteps: int = 1,
        equilibration_timesteps: int = 0,
        save: bool = False,
        save_interval: int = 1,
    ):
        """
        Apply _mcmc_move to all the replicas md_timesteps times.
        If equilibration_timesteps > 0,
            an equilibration phase is considered before try to save position and forces.
        If save is set to True position and forces are saved every save_interval time.
        _thermodynamic_state[i] is associated to the replica configuration in _replicas_sampler_states[i].

        Params:
        -------
        md_timesteps:
            how many times apply self.mcmc_move to each state
        equilibration_timesteps:
            During equilibration position, forces and state of the system are not saved
        save:
            position and forces saving every save_interval timesteps
        save_interval:
            frequency to save position and forces
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        for md_step in range(md_timesteps):
            propagated_states = mpiplus.distribute(
                self._run_replica, range(self.n_replicas), send_results_to='all'
            )

            # Update all sampler states.
            for replica_id, propagated_state in enumerate(propagated_states):
                self._replicas_sampler_states[replica_id].__setstate__(
                    propagated_state, ignore_velocities=False
            )

            if rank == 0:
                # verify if reporter was loaded
                if self._reporter is not None:
                    report_interval = self._reporter.get_report_interval()
                if md_step >= equilibration_timesteps:
                    if save and (md_step % save_interval == 0):
                        self.positions.append([])
                        self.forces.append([])
                        self.energies.append([])
                        self.positions[-1], self.forces[-1], self.energies[-1] = self._grab_forces()
                    if self._reporter is not None and (md_step % report_interval) == 0:
                            self._reporter.store_report(
                                self._thermodynamic_states,
                                self._replicas_sampler_states,
                                self._temperature_list,
                            )

    def _run_replica(self, replica_id):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        # context, _ = context_cache.get_context(thermodynamic_state)

        thermodynamic_state = self._thermodynamic_states[replica_id]
        sampler_state = self._replicas_sampler_states[replica_id]
        context_id = self._get_context_gpu_id(replica_id)
        context_cache = self._get_context_cache(context_id)
        # logger.debug(f"_propagate_replica ID:{replica_id} from RANK:{rank} and {platform.getName()}:{platform.getPropertyValue(context, 'DeviceIndex')}")

        self._mcmc_move.apply(
            thermodynamic_state, sampler_state, context_cache=context_cache
        )
        return sampler_state.__getstate__(ignore_velocities=False)

    @mpiplus.on_single_node(0, broadcast_result=True)
    def _mix_replicas(
        self,
        mixing: str = "all",
        n_attempts=1,
    ):
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
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        random_number_list = np.random.rand(n_attempts)

        premix_temperatures = []
        for _, thermo_state in enumerate(self._thermodynamic_states):
            premix_temperatures.append(thermo_state.temperature)

        start = time()
        # Attempts swaps
        # Keep track of temperature history
        # Rescale velocities
        for attempt in range(n_attempts):
            # logger.debug(f"_mix_replicas attempt:{attempt} from RANK:{rank}")
            self._mix_states(mixing, random_number_list[attempt])

        if self._temperature_history is not None:
            temperatures = [
                self._thermodynamic_states[i].temperature.value_in_unit(unit.kelvin)
                for i in range(self.n_replicas)
            ]
            self._temperature_history.append(temperatures)

        if self.rescale_velocities == True:
            self._rescale_velocities(premix_temperatures)

        end = time()
        logger.warning(f"_mix_replicas Exchange took {end-start} [sec]")
        return self._thermodynamic_states

    def _mix_states(self, mixing, random_number):
        """
        Perform mixing
        Decorator runs only on first node and all process halted until finished and then broadcast
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if mixing == "neighbors":
            i, j = np.sort(self.couples[np.random.randint(len(self.couples))])
        elif mixing == "all":
            i, j = np.sort(
                np.random.choice(
                    range(len(self._thermodynamic_states)), 2, replace=False
                )
            )
        self.acceptance_matrix[j, i] += 1

        if self.energy_matrix is None:
            ## Since the temperatures are swapping through time, we have to index correctly
            sampler_state_i, sampler_state_j = (
                self._replicas_sampler_states[k] for k in [i, j]
            )
            temperatures = [self._thermodynamic_states[k].temperature for k in range(self.n_replicas)]
            temp_i = temperatures.index(self._temperature_list[i])
            temp_j = temperatures.index(self._temperature_list[j])
            thermo_state_i, thermo_state_j = (
                self._thermodynamic_states[k] for k in [temp_i, temp_j]
            )

            # Compute the energies.
            energy_ii = self._compute_reduced_potential(
                [sampler_state_i, thermo_state_i]
            )
            energy_jj = self._compute_reduced_potential(
                [sampler_state_j, thermo_state_j]
            )
            energy_ij = self._compute_reduced_potential(
                [sampler_state_i, thermo_state_j]
            )
            energy_ji = self._compute_reduced_potential(
                [sampler_state_j, thermo_state_i]
            )
        else:
            energy_ii, energy_jj = self.energy_matrix[i, i], self.energy_matrix[j, j]
            energy_ij, energy_ji = self.energy_matrix[i, j], self.energy_matrix[j, i]

        # Accept or reject the swap.
        log_p_accept = -(energy_ij + energy_ji) + energy_ii + energy_jj
        if random_number < np.exp(log_p_accept):
            # Swap states in replica slots i and j. Update energy matrix if needed
            self._thermodynamic_states[i], self._thermodynamic_states[j] = (
                self._thermodynamic_states[j],
                self._thermodynamic_states[i],
            )
            self.acceptance_matrix[i, j] += 1

            if self.energy_matrix is not None:
                self.energy_matrix[:,[i,j]] = self.energy_matrix[:,[j,i]]

    def _compute_reduced_potential_matrix(self):
        # Compute the reduced potential matrix between all possible couples
        self.energy_matrix = np.zeros((self.n_replicas, self.n_replicas))
        inps = []
        for i_s, sampler_state in enumerate(self._replicas_sampler_states):
            for replica_j, thermo_state in enumerate(self._thermodynamic_states):
                # Grab the context associated to thermodynamic state so from correct gpu
                context = self._get_context(replica_j, thermo_state)
                inps.append((sampler_state, thermo_state, context, (i_s,replica_j)))

        outs = mpiplus.distribute(
            self._compute_reduced_potential, inps, send_results_to="all"
        )
        self.energy_matrix = np.array(outs).reshape(self.n_replicas,self.n_replicas)
        # logger.debug(f"_energy_computation from RANK:{rank}  LEN:{len(outs)}")

    def _compute_reduced_potential(
        self, args
    ):  
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        sampler_state = args[0]
        thermo_state = args[1]
        if len(args) == 3:
            context = args[2]
        else:
            context, _ = cache.global_context_cache.get_context(thermo_state)

        # logger.debug(f"_compute_reduced_potential from RANK:{rank} and {platform.getName()}:{platform.getPropertyValue(context, 'DeviceIndex')}")

        # Compute the reduced potential of the sampler_state configuration
        #  in the given thermodynamic state.
        sampler_state.apply_to_context(context)
        return thermo_state.reduced_potential(context)

    def _grab_forces(self):
        """
        Return position and forces of the target elements

        Outputs
            positions : np.array
            forces : np.array
        """
        forces = np.zeros((self.n_replicas, len(self.target_elements), self._dimension))
        positions = np.zeros(forces.shape)
        energies = np.zeros(self.n_replicas)
        for (thermo_state, sampler_state) in zip(
            self._thermodynamic_states, self._replicas_sampler_states
        ):
            i_t = self._temperature_list.index(thermo_state.temperature)
            context = self._get_context(i_t, thermo_state)
            sampler_state.apply_to_context(context)
            #
            position = context.getState(getPositions=True).getPositions(asNumpy=True)[
                self.target_elements
            ]
            position = position.value_in_unit(unit.angstrom)
            force = context.getState(getForces=True).getForces(asNumpy=True)[
                self.target_elements
            ]
            force = force.value_in_unit(unit.kilocalorie_per_mole / unit.angstrom)
            energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
            positions[i_t] = position
            forces[i_t] = force
            energies[i_t] = energy
        return positions, forces, energies

    def _define_neighbors(self):
        """ Define all possible couples of neighbors """
        l = np.arange(len(self._thermodynamic_states))
        couples = []
        for i in zip(l, l[1:]):
            couples.append(i)
        self.couples = np.array(couples)

    def _get_context_gpu_id(self, replica_id):
        """
        Return the gpu id that context is loaded onto
        """
        return replica_id % self.n_gpus

    def _get_context_cache(self, context_gpu_id):
        """
        Retrieve context cache.
            This are set at beginning of run to spread simulations across available gpus
        """
        return self._local_cache[context_gpu_id]

    def _get_context(self, replica_id, thermo_state):
        """
        Return associated context based on gpu it has been assigned to and the thermodynamic_state
        """
        context_gpu_id = self._get_context_gpu_id(replica_id)
        context_cache = self._get_context_cache(context_gpu_id)
        context, _ = context_cache.get_context(thermo_state)
        return context

    def _rescale_velocities(self, original_temperature=None):
        """
        Rescale velocities to desired temperature from thermodynamic state after swap attempt
        """
        for (thermo_state, sampler_state) in zip(self._thermodynamic_states, self._replicas_sampler_states):
            i_t = self._temperature_list.index(thermo_state.temperature)
            context = self._get_context(i_t, thermo_state)
            sampler_state.apply_to_context(context)
            if original_temperature is not None:
                context.setVelocities(
                    context.getState(getVelocities=True).getVelocities()
                    * np.sqrt(thermo_state.temperature / original_temperature[i_t])
                )
            else:
                context.setVelocitiesToTemperature(thermo_state.temperature)
